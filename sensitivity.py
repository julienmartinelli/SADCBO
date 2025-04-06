"""Functions necessary for implementing the sensitivity analysis BO baseline."""

import torch


def sensitivity(X, z_idx, cost_vector, model):

    n, c = len(X), len(z_idx)
    KLs = torch.zeros((n, c))
    for k, j in enumerate(z_idx):
        Delta_j = torch.zeros((1, X.shape[1]))
        Delta_j[0, j] += 1
        for i in range(n):
            x1 = X[i].unsqueeze(-1).T
            x2 = x1 * (1 - Delta_j)
            KLs[i, k] = d_KL(x1, x2, model)
    return KLs / cost_vector


def d_KL(x1, x2, model):
    """
    Compute d(p||q) = sqrt(D_KL(p||q)) with:
    p = p(y*|x1,y), the univariate gaussian posterior of a GP at predictive point x1
    q = p(y*|x2, y), the univariate gaussian posterior of a GP at predictive point x2
    Since both p and q are gaussian, there is closed-form formula.
    x1 and x2 should be (n,1)-dim tensors.
    """

    model.eval()
    mx1, mx2 = model.posterior(x1), model.posterior(x2)
    mu1, mu2 = mx1.mean, mx2.mean
    sigma1, sigma2 = mx1.variance, mx2.variance
    kl = (
        torch.log(torch.sqrt(sigma2) / torch.sqrt(sigma1))
        + (sigma1 + (mu1 - mu2) ** 2) / (2 * sigma2)
        - 1 / 2
    )
    return torch.sqrt(kl)
