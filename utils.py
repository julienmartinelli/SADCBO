import argparse
import itertools
import math

import matplotlib as mpl
import numpy as np
import torch


def set_matplotlib_params():
    """Set matplotlib params."""

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rc("font", family="serif")
    mpl.rcParams.update(
        {
            "font.size": 24,
            "lines.linewidth": 2,
            "axes.labelsize": 24,  # fontsize for x and y labels
            "axes.titlesize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "axes.linewidth": 2,
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": True,  # use LaTeX to write all text
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            # "axes.grid": True,
            "legend.shadow": True,
            "legend.fancybox": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


def adapt_save_fig(fig, filename="test.pdf"):
    """Remove right and top spines, set bbox_inches and dpi."""

    for ax in fig.get_axes():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def parser_bo():
    """
    Parser used to run the algorithm from an already known crn.
    - Output:
        * parser: ArgumentParser object.
    """

    parser = argparse.ArgumentParser(description="Command description.")

    parser.add_argument(
        "-n", "--N_REP", help="int, number of reps for stds", type=int, default=1
    )
    parser.add_argument(
        "-ni", "--N_INIT", help="int, size of initial dataset", type=int, default=1
    )
    parser.add_argument(
        "-se", "--seed", default=None, help="int, random seed", type=int
    )
    parser.add_argument(
        "-s", "--savefolder", default=None, type=str, help="Name of saving directory."
    )
    parser.add_argument(
        "-b",
        "--budget",
        help="BO Budget",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-k",
        "--kernels",
        nargs="*",
        type=str,
        default=["RBF"],
        help="list of kernels to try.",
    )
    parser.add_argument(
        "-m",
        "--methods",
        nargs="*",
        type=str,
        default=["Random"],
        help="list of baselines to try.",
    )
    parser.add_argument(
        "-a",
        "--acqfs",
        nargs="*",
        type=str,
        default=["UCB"],
        help="list of BO acquisition function to try.",
    )
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="*",
        type=str,
        default=["Hartmann"],
        help="list of test functions to optimize.",
    )
    return parser


def build_combinations(N_REP, experiments, methods, kernels, acqfs, n_init, seed):
    """Construct the list of combination settings to run."""

    li = [
        experiments,
        methods,
        kernels,
        acqfs,
        [n_init],
        [seed + n for n in range(N_REP)],
    ]
    combi = [list(itertools.product(*li))]
    return sum(combi, [])


def embed_test_function(testfunc, x, rel_dim):
    lb, ub = torch.tensor(testfunc._bounds).T
    return testfunc(lb + (ub - lb) * x[..., :rel_dim])


def mean_regrets(results, grid, metric):
    nrep = len(results["costs"])
    grid_val = np.zeros((len(grid), nrep))
    for i, (cost, regret) in enumerate(zip(results["costs"], results[metric])):
        idx = np.searchsorted(cost, grid, side="right") - 1
        grid_val[:, i] = regret[idx]
    return np.nanmean(grid_val, axis=1), 1.96 * np.nanstd(grid_val, axis=1) / math.sqrt(
        nrep
    )


def stats_sensitivity(paths, exp):
    """Compute percentage a contextual variable being selected by sensitivity analysis."""

    scheme = exp.split("_")[1]
    rel_dim, x_idx, irr_dim = scheme.split("-")
    if x_idx == "@":
        x_idx = []
    rel_dim = int(rel_dim)
    irr_dim = int(irr_dim)
    x_idx = x_idx.split("&")
    x_idx = list(map(int, list(x_idx)))
    embpbdim = rel_dim + irr_dim
    z_idx = [i for i in range(embpbdim) if i not in x_idx]
    z_rel_dim = rel_dim - len(x_idx)
    pp = torch.zeros((0, len(z_idx)))
    for i in range(len(paths)):
        pp = torch.cat((pp, paths[i]))
    mean, std = torch.nanmean(pp, axis=0), torch.std(pp, axis=0)
    return mean.detach().numpy(), std.detach().numpy(), z_idx, z_rel_dim


def extract_method_hp(method):
    info = method.split("-")
    if len(info) < 2:
        method, hp1, hp2, hp3 = info[0], None, None, None
    elif len(info) == 2:
        method, hp1 = info
        hp1, hp2, hp3 = int(hp1), None, None
    else:
        method, hp1, hp2, hp3 = info
        hp1, hp2, hp3 = float(hp1), int(hp2), float(hp3)
    return method, hp1, hp2, hp3
