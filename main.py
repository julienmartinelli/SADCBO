import copy
import functools
import multiprocessing as mp
import warnings

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from BO import BayesianOptimization
from stopping_criterion import compute_stopping_criterion
from utils import (build_combinations, embed_test_function, parser_bo)

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)
torch.set_default_tensor_type(torch.DoubleTensor)


def eval_model(combi, budget, savefolder):
    exp, method, str_K, str_acqf, n_init, seed = combi
    path = f"{savefolder}/{exp}_{method}_{str_K}_{str_acqf}_{seed}.pt"
    torch.manual_seed(seed)
    np.random.seed(seed)
    BO = BayesianOptimization(exp, method, str_acqf, budget, seed)

    # initiate contexts in advance for reproducibility
    Z, i = torch.rand((1000, BO.z_dim, 1)), 0
    n = n_init * BO.rel_dim  # number of initial training points
    # fixing seed=0 SAME TRAIN_X ACROSS SEEDS EG RANDOMNESS CANNOT COME FROM INIT DATASET
    train_X = SobolEngine(dimension=BO.embpbdim, scramble=True, seed=seed).draw(n)
    train_Y = embed_test_function(BO.problem, train_X, BO.rel_dim).unsqueeze(-1)
    train_Y_nonoise = torch.clone(train_Y) # saving these to compute the simple regret
    train_Y += BO.problem.noise * torch.randn_like(train_Y)

    cc = []
    B = budget * BO.rel_dim
    ##### INITIAL DATASET AND GP FITTING
    BO.choose_kernel(str_K, t=0)
    BO.update_dataset(train_X=train_X, train_Y=train_Y, train_Y_nonoise=train_Y_nonoise)

    BO.fit_full_gp()
    previousmodel = copy.deepcopy(BO.gprfull)
    BO.init_sensi_matrix()
    BO.update_regret()

    ##### BO LOOP
    while B > 0:
        z = Z[i]
        BO.get_candidate(z)
        y = embed_test_function(BO.problem, BO.xz.squeeze(), BO.rel_dim).view(1, -1)
        y_nonoise = torch.clone(y)
        y += BO.problem.noise * torch.randn_like(y)
        BO.update_dataset(train_X=None, train_Y=None, y=y, y_nonoise=y_nonoise)
        previousmodel = copy.deepcopy(BO.gprfull)
        BO.fit_full_gp()
        BO.update_sensi_matrix()
        if "contextual" in BO.method:
            status, deltart, st, kappa, bru = compute_stopping_criterion(
                BO.gprfull, previousmodel, BO.data, BO.bounds_acqf_full, cc, method
            )
            if not status:
                BO.method = BO.method.replace(
                    "contextual", ""
                )  # switching from phase 1 to phase 2
                BO.data["stopped"] = i
        BO.update_regret()
        B -= BO.candidate_cost
        i += 1
    torch.save(BO.data, path)


def parallel_eval(combi, budget, savefolder, x):
    return eval_model(combi[x], budget, savefolder)


def main(N_REP, N_INIT, budget, kernels, acqfs, experiments, seed, methods, savefolder):
    combi = build_combinations(
        N_REP, experiments, methods, kernels, acqfs, N_INIT, seed
    )
    with mp.Pool() as p:
        p.map(
            functools.partial(parallel_eval, combi, budget, savefolder),
            range(len(combi)),
        )
    p.close()

if __name__ == "__main__":
    parser = parser_bo()
    main(**vars(parser.parse_args()))
