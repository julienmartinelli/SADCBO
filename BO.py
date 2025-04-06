import copy

import torch
from botorch.acquisition.analytic import (ExpectedImprovement,
                                          UpperConfidenceBound)
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from sensitivity import sensitivity
from synthetic_functions import Ackley, EggHolder, Hartmann
from utils import extract_method_hp


class BayesianOptimization:
    def __init__(
        self, func, method, str_acqf, budget, seed, train_X=None, train_Y=None, train_Y_nonoise=None
    ):
        self.func = func
        self.budget = budget
        (
            self.method,
            self.eta,
            self.batch_diversity,
            self.gamma,
        ) = extract_method_hp(method)

        self.data = {"regrets": torch.zeros(0), "bestvals": torch.zeros(0)}
        self.max_retries_gp = 5  # GP hyperparameters optimization restarts
        self.num_restarts = 10
        self.raw_samples = 256
        self.beta = 0.2  # default value for UCB
        self.str_acqf = str_acqf

        self.tkwargs = {
            "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.double,
        }

        # variable selection attributes
        self.kept_z_idx = None
        self.kept_idx = None

        self.sampler = SobolQMCNormalSampler(torch.Size([1024]))

        self.synthetic_functions()

        self.candidate_cost = 0
        self.fixed_cost = 1 * self.x_dim  # hardcoded cost, 1 per x dim
        self.cost_vector = 1 * torch.ones(self.z_dim)  # hardcoded cost, 1 per z dim
        self.mapzidx = {s: i for i, s in enumerate(self.z_idx)}

    def synthetic_functions(self):
        """
        Instantiate the given function to optimize together
        with some useful dimensions values for relevant and irrelevant variables.
        """

        funcname, dims = self.func.split("_")
        self.dims_and_bounds(dims)
        self.funcname = funcname

        if funcname == "Hartmann":
            if self.rel_dim not in [3, 4, 6]:
                raise ValueError(
                    "Hartmann only defined for 3, 4 and 6 dimensional relevant variables"
                )
            self.problem = Hartmann(dim=self.rel_dim, negate=True)
        elif funcname == "Ackley":
            self.problem = Ackley(
                dim=self.rel_dim,
                negate=True,
                bounds=[(-5.0, 5.0) for _ in range(self.rel_dim)],
            )
        elif funcname == "EggHolder":
            self.problem = EggHolder(
                dim=self.rel_dim,
                negate=False,
                bounds=[(-512.0, 512.0) for _ in range(self.rel_dim)],
            )
        self.problem.opt = 1.0
        self.problem.noise = 0.01

    def dims_and_bounds(self, dims):
        """Compute dimension of the relevant, irrelevant problem etc."""

        # rel_dim, x_idx, irr_dim = re.findall(r"\d+", dims)
        rel_dim, x_idx, irr_dim = dims.split("-")
        if x_idx == "@":
            x_idx = []

        self.rel_dim = int(rel_dim)
        self.irr_dim = int(irr_dim)
        self.embpbdim = self.rel_dim + self.irr_dim
        x_idx = x_idx.split("&")
        self.x_dim = len(x_idx)
        self.z_dim = self.embpbdim - self.x_dim

        self.x_idx = list(map(int, list(x_idx)))
        self.z_rel_idx = [i for i in range(self.rel_dim) if i not in self.x_idx]
        self.rel_idx = list(range(self.rel_dim))
        self.irr_idx = list(range(self.rel_dim, self.embpbdim))
        self.z_idx = [i for i in range(self.embpbdim) if i not in self.x_idx]

        self.set_method()

        # bounds for acqf optim in diversity set computation
        self.bounds_acqf_full = torch.cat(
            (torch.zeros(1, self.embpbdim), torch.ones(1, self.embpbdim))
        )
        self.boundsacqf = torch.cat((torch.zeros(1, self.dim), torch.ones(1, self.dim)))

        # map context indicies in [x,z] to their indices in z only
        self.mapzidxdict = {i: j for j, i in enumerate(self.z_idx)}

    def choose_kernel(self, str_K, custom_dim=None, t=1):
        """Instantiate the given kernel."""
        dim = self.dim if custom_dim is None else custom_dim
        # ScaleKernel adds the amplitude hyperparameter
        if str_K == "RBF":
            self.K = ScaleKernel(RBFKernel(ard_num_dims=dim))
        elif str_K == "Matern":
            self.K = ScaleKernel(MaternKernel(ard_num_dims=dim))
        self.str_K = str_K

    def fit_gp(self):

        inputs = self.data["train_X"][..., self.kept_idx]
        self.choose_kernel(
            self.str_K,
            custom_dim=len(self.kept_z_idx) + self.x_dim
        )
        self.gpr = SingleTaskGP(
            inputs,
            self.data["train_Y"],
            covar_module=self.K,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(self.gpr.likelihood, self.gpr)
        fit_gpytorch_mll(mll, max_retries=5)

    def fit_full_gp(self):
        """
        Fits a GP with all variables for SADCBO and projpred method
        Necessary as these baseline deals with 2 GPs:
            - the full one for sensitivity analysis / projpred batch acqf
            - the reduced one for acqf optimization
        """

        if self.str_K == "RBF":
            self.Kfull = ScaleKernel(RBFKernel(ard_num_dims=self.embpbdim))
        elif self.str_K == "Matern":
            self.Kfull = ScaleKernel(MaternKernel(ard_num_dims=self.embpbdim))
        self.gprfull = SingleTaskGP(
            self.data["train_X"],
            self.data["train_Y"],
            covar_module=self.Kfull,
            outcome_transform=Standardize(m=1),
        )
        mllfull = ExactMarginalLogLikelihood(self.gprfull.likelihood, self.gprfull)
        fit_gpytorch_mll(mllfull, max_retries=self.max_retries_gp)

    def pick_acqf(self):
        "Instantiate the given acqf."

        if "UCB" in self.str_acqf:
            if len(self.str_acqf) > 3:
                self.beta = float(self.str_acqf[3:])
            self.acqf = UpperConfidenceBound(self.gpr, self.beta)

        elif self.str_acqf == "EI":
            self.acqf = ExpectedImprovement(self.gpr, self.data["train_Y"].max())
        elif self.str_acqf == "MES":
            self.candidate_set = torch.rand(
                1000,
                self.boundsacqf.size(1),
                device=self.boundsacqf.device,
                dtype=self.boundsacqf.dtype,
            )
            self.candidate_set = (
                self.boundsacqf[0]
                + (self.boundsacqf[1] - self.boundsacqf[0]) * self.candidate_set
            )
            self.acqf = qMaxValueEntropy(self.gpr, self.candidate_set)

    def get_candidate(self, z):
        self.less_xz = 0

        if self.batch_diversity:
            self.get_diverse_candidates(z)
        self.kept_z_idx, res = self.select_variables_sensitivity()
        self.vs_settings(res)
        self.fix_z_acqf(z)

        self.xz, _ = optimize_acqf(
            acq_function=self.acqf,
            bounds=self.boundsacqf,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            fixed_features=self.fix_dims,
        )
        self.compute_cost()
        self.create_query(z)


    def fix_z_acqf(self, z):
        """Fix z to given values when maximizing acqf"""
        zz = z.flatten()

        if self.method == "Sensitivitycontextual":
            self.fix_dims = {
                self.mapdic_kept_idx[s]: zz[self.mapzidx[s]] for s in self.kept_z_idx
            }
            self.opt_dims = [
                i for i in range(len(self.kept_idx)) if i not in self.fix_dims.keys()
            ]
        else:
            self.fix_dims = None

    def mapping(self):
        self.map = {}
        add = copy.copy(self.x_dim)
        for k in self.kept_z_idx:
            if k < self.x_dim:
                self.map[k] = k
            else:
                self.map[k] = add
                add += 1
        return self.map

    def vs_settings(self, res):
        self.kept_idx = sorted(self.kept_z_idx + self.x_idx)
        self.mapdic_kept_idx = {
            k: i for i, k in enumerate(self.kept_idx) if k in self.kept_z_idx
        }
        self.data["sensitivity_path"] = torch.cat((self.data["sensitivity_path"], res))
        self.data["idx_path"].append(self.kept_z_idx)
        self.choose_kernel(
            self.str_K, custom_dim=len(self.kept_z_idx) + self.x_dim, t=1
        )
        self.fit_gp()
        self.boundsacqf = torch.cat(
            (
                torch.zeros(1, len(self.kept_z_idx) + self.x_dim),
                torch.ones(1, len(self.kept_z_idx) + self.x_dim),
            )
        )
        self.pick_acqf()

    def update_dataset(self, train_X=None, train_Y=None, train_Y_nonoise=None, y=None, y_nonoise=None):
        if "train_X" not in self.data.keys():
            self.data["train_Y"] = train_Y
            self.data["train_Y_nonoise"] = train_Y_nonoise
            self.data["train_X"] = train_X
            self.data["sensitivity_path"] = torch.zeros((0, self.z_dim))
            self.data["ard_path"] = torch.zeros((0, self.z_dim))
            self.data["idx_path"] = []
        else:
            self.data["train_X"] = torch.cat((self.data["train_X"], self.xz))
            self.data["train_Y"] = torch.cat((self.data["train_Y"], y))
            self.data["train_Y_nonoise"] = torch.cat((self.data["train_Y_nonoise"], y_nonoise))


    def update_regret(self, ):
        """Compute simple regret at current iteration."""

        maxval = self.data["train_Y_nonoise"].max()
        regret = torch.tensor([self.problem.opt - maxval])
        self.data["regrets"] = torch.cat((self.data["regrets"], regret))
        self.data["bestvals"] = torch.cat(
            (self.data["bestvals"], torch.tensor([maxval]))
        )

    def set_method(self):
        """Set the input dimension relatively to selected baseline."""

        self.dim = self.x_dim  # just a placeholder, dim is computed on the fly

    def compute_cost(self):
        """Compute the cost of selected sample to evaluate."""

        if "costs" not in self.data.keys():
            self.data["costs"] = [0]
        if self.method == "Sensitivitycontextual":
            self.candidate_cost = self.fixed_cost
        if self.method == "Sensitivity":
            self.candidate_cost = float(
                self.fixed_cost
                + torch.sum(
                    self.cost_vector[[self.mapzidx[i] for i in self.kept_z_idx]]
                )
            )
        self.data["costs"].append(self.data["costs"][-1] + float(self.candidate_cost))

    def create_query(self, z):
        """Assemble the final query from acqf optimizer and sampled contextual variables."""

        if "contextual" in self.method:
            self.less_xz = torch.clone(self.xz)
            almost_xz = torch.zeros((1, self.embpbdim))
            almost_xz[:, self.x_idx] = self.xz[:, self.opt_dims]
            almost_xz[:, self.z_idx] = z.flatten()
        else:
            self.less_xz = torch.clone(self.xz)
            almost_xz = torch.zeros((1, self.embpbdim))
            notsensi_dim = [
                i for i in range(self.embpbdim) if i not in self.kept_idx
            ]
            almost_xz[:, self.kept_idx], almost_xz[:, notsensi_dim] = (
                self.less_xz.flatten(),
                z[[d - self.x_dim for d in notsensi_dim]].flatten(),
            )
        self.xz = torch.clone(almost_xz)

    def init_sensi_matrix(self):
        """Compute initial sensitivity matrix from initial dataset."""

        self.sensitivity_matrix = sensitivity(
            self.data["train_X"],
            self.z_idx,
            self.cost_vector,
            self.gprfull,
        )

    def update_sensi_matrix(self):
        """Update sensitivity matrix with new query."""

        new_sensi = sensitivity(
            self.xz,
            self.z_idx,
            self.cost_vector,
            self.gprfull,
        )
        self.sensitivity_matrix = torch.cat((self.sensitivity_matrix, new_sensi))

    def select_variables_sensitivity(self):
        """Append sensi to avoid recomputing indices for whole dataset."""
        idx = self.cut_dataset()
        truncated_matrix = self.sensitivity_matrix[idx]
        if len(truncated_matrix.shape) == 1:
            truncated_matrix = truncated_matrix.view(1, -1)
        if self.batch_diversity:
            sensi_batchacqf = sensitivity(
                self.sensicand, self.z_idx, self.cost_vector, self.gprfull
            )
            truncated_matrix = torch.cat((truncated_matrix, sensi_batchacqf))
        bestidx, res = self.compute_sum_indices(truncated_matrix)
        return bestidx, res

    def compute_sum_indices(self, sensimat):
        res = sensimat.nanmean(dim=0)
        res[torch.isnan(res)] = 0
        res = res / sum(res)
        final, idx = torch.sort(res, descending=True)
        final = torch.cumsum(final, 0)
        select = torch.argwhere(final > self.eta)[0]
        bestidx = idx[: select + 1]
        return sorted([self.z_idx[b] for b in bestidx]), res.view(1, -1)

    def get_diverse_candidates(self, z=None):
        """Obtain a set of diverse candidates to compute sensitivity analysis on."""

        qUCB = qUpperConfidenceBound(self.gprfull, self.beta, self.sampler)
        self.sensicand, _ = optimize_acqf(
            acq_function=qUCB,
            bounds=self.bounds_acqf_full,
            q=self.batch_diversity,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            fixed_features=dict(zip(self.z_idx, z.squeeze())),
        )

    def cut_dataset(self):
        """Cut the dataset to only keep the 100*gamma% best samples."""

        idx = torch.sort(-self.data["train_Y"], dim=0)[1]
        cut = int(self.gamma * len(idx))
        idx = idx[:cut].squeeze()
        return idx