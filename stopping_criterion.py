import math
from typing import Optional, Union

import numpy as np
import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class CustomUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        lcb=False,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.lcb = lcb
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        factor = -self.beta.sqrt() if self.lcb else self.beta.sqrt()
        return (mean if self.maximize else -mean) + factor * sigma


def compute_stopping_criterion(model, previousmodel, data, bounds, li, method):
    argmax, argoldmax = (data["train_Y"]).argmax(), (data["train_Y"][:-1]).argmax()
    curr_max = data["train_X"][argmax].view(1, -1)
    old_max = data["train_X"][argoldmax].view(1, -1)

    modeleval, previousmodeleval = model(curr_max), previousmodel(old_max)
    previousmodelcurreval = previousmodel(data["train_X"][-1].view(1, -1))

    noise = model.likelihood.noise.item()
    covar = model(torch.cat((curr_max, old_max))).covariance_matrix[0][1]
    modelevalvar = modeleval.variance
    previousmodelevalvar = previousmodeleval.variance
    previousmodelevalmean = previousmodeleval.mean
    modelevalmean = modeleval.mean

    previousmodelcurrevalmean = previousmodelcurreval.mean
    previousmodelcurrevalvar = previousmodelcurreval.variance
    previousmodelargmaxvar = previousmodel(curr_max).variance

    clip = 1e-16  # avoid taking square root of negative...
    clipsqrt = max(modelevalvar + previousmodelevalvar - 2 * covar, clip)
    v = math.sqrt(clipsqrt)
    g = ((previousmodelevalmean) - (modelevalmean)) / v

    dist = torch.distributions.normal.Normal(0, 1)
    phig = torch.exp(dist.log_prob(g))
    Phig = dist.cdf(g)

    noiseinv = noise ** (-1)
    firstkl = torch.log(1 + noise * previousmodelcurrevalvar)
    secondkl = previousmodelcurrevalvar / (previousmodelcurrevalvar + noiseinv)
    thirdkl = (
        previousmodelcurrevalvar
        * (data["train_Y"][-1] - previousmodelcurrevalmean) ** 2
    ) / (previousmodelcurrevalvar + noiseinv) ** 2
    kl = 0.5 * max((firstkl - secondkl + thirdkl), clip)
    beta = 0.2

    UCB = CustomUpperConfidenceBound(model, beta)
    LCB = CustomUpperConfidenceBound(model, beta, lcb=True)

    _, val = optimize_acqf(
        acq_function=UCB,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
    )

    kappa = val - LCB(data["train_X"][:-1].unsqueeze(-2)).max()
    deltart = (
        torch.abs((previousmodelevalmean) - (modelevalmean))
        + v * (phig + g * Phig)
        + kappa * math.sqrt(1 / 2 * kl)
    )

    ### COMPUTE st
    delta = 0.05  # w/ prob 1-delta
    c = math.sqrt(-2 * math.log(delta))
    numst = (
        (torch.sqrt(previousmodelargmaxvar) + kappa / 2)
        * torch.sqrt(previousmodelcurrevalvar)
        * c
    )
    denomst = math.sqrt(noise) * (previousmodelcurrevalvar + noiseinv)
    st = numst / denomst
    # status = deltart >= st

    # parametric stopping criterion, see end of sec 4.3 from Ishibashi et al.
    li.append(float(deltart))
    med = np.median(li[:20])
    status = deltart >= 0.1 * med or len(li) < 20
    return status, deltart, st, kappa, 0.1 * med
