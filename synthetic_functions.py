import math
from typing import List, Optional, Tuple

import torch
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor


class SyntheticTestFunction(BaseTestProblem):
    r"""Base class for synthetic test functions."""

    _optimal_value: float
    _optlowval: float
    _optimizers: Optional[List[Tuple[float, ...]]] = None
    num_objectives: int = 1

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if bounds is not None:
            self._bounds = bounds
        super().__init__(noise_std=noise_std, negate=negate)
        if self._optimizers is not None:
            if bounds is not None:
                # Ensure at least one optimizer lies within the custom bounds
                def in_bounds(
                    optimizer: Tuple[float, ...], bounds: List[Tuple[float, float]]
                ) -> bool:
                    for i, xopt in enumerate(optimizer):
                        lower, upper = bounds[i]
                        if xopt < lower or xopt > upper:
                            return False

                    return True

                if not any(
                    in_bounds(optimizer=optimizer, bounds=bounds)
                    for optimizer in self._optimizers
                ):
                    raise ValueError(
                        "No global optimum found within custom bounds. Please specify "
                        "bounds which include at least one point in "
                        f"`{self.__class__.__name__}._optimizers`."
                    )
            self.register_buffer(
                "optimizers", torch.tensor(self._optimizers, dtype=torch.float)
            )

    @property
    def optimal_value(self) -> float:
        r"""The global minimum (maximum if negate=True) of the function."""
        return -self._optimal_value if self.negate else self._optimal_value

    @property
    def optlowval(self) -> float:
        r"""The global minimum (maximum if negate=True) of the function."""
        return -self._optlowval if self.negate else self._optlowval


class Hartmann(SyntheticTestFunction):
    r"""Hartmann synthetic test function.

    Most commonly used is the six-dimensional version (typically evaluated on
    `[0, 1]^6`):

        H(x) = - sum_{i=1}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )

    H has a 6 local minima and a global minimum at

        z = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

    with `H(z) = -3.32237`.
    """

    def __init__(
        self,
        dim=6,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if dim not in (3, 4, 6):
            raise ValueError(f"Hartmann with dim {dim} not defined")
        self.dim = dim
        if bounds is None:
            bounds = [(0.0, 1.0) for _ in range(self.dim)]
            # self._bounds = bounds
        # optimizers and optimal values for dim=4 not implemented
        optvals = {3: -3.86278, 4: -3.13370, 6: -3.32237}
        optlowvals = {3: 0.0, 4: 1.30943, 6: 0.0}
        optimizers = {
            3: [(0.114614, 0.555649, 0.852547)],
            6: [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)],
        }
        self._optlowval = optlowvals.get(self.dim)
        self._optimal_value = optvals.get(self.dim)
        self._optimizers = optimizers.get(self.dim)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        if dim == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dim == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        self.register_buffer("A", torch.tensor(A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float))

    @property
    def optimal_value(self) -> float:
        return super().optimal_value

    @property
    def optimizers(self) -> Tensor:
        if self.dim == 4:
            raise NotImplementedError()
        return super().optimizers

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = -(torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        if self.dim == 4:
            H = (1.1 + H) / 0.839
            H = -H if self.negate else H
        H = (H - self.optlowval) / (self.optimal_value - self.optlowval)
        return -H if self.dim == 4 and self.negate else H


class Ackley(SyntheticTestFunction):
    r"""Ackley synthetic test function.

    d-dimensional function (usually evaluated on `[-5, 10]^d`):

        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_i) = 0.0`.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim=2,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        if bounds is None:
            bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        if not self.negate:
            raise NotImplementedError(
                "Standardization not implemented for non negated case"
            )
        self._min_ = -14.30267
        self._max_ = 0.0
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate_true(self, X: Tensor) -> Tensor:
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.norm(X, dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
        H = part1 + part2 + a + math.e
        if self.negate:
            H = -H
        return -(H - self._min_) / (self._max_ - self._min_)

class EggHolder(SyntheticTestFunction):
    r"""Eggholder test function.

    Two-dimensional function (usually evaluated on `[-512, 512]^2`):

        E(x) = (x_2 + 47) sin(R1(x)) - x_1 * sin(R2(x))

    where `R1(x) = sqrt(|x_2 + x_1 / 2 + 47|)`, `R2(x) = sqrt|x_1 - (x_2 + 47)|)`.
    """

    def __init__(
        self,
        dim=2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:

        self.dim = dim
        bounds = [(-512.0, 512.0), (-512.0, 512.0)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        if self.negate:
            raise NotImplementedError(
                "Standardization not implemented for negated case"
            )
        self._min_ = -1048.9454345703125
        self._max_ = 959.5001220703125

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = -(x2 + 47.0) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47.0))))
        result = - (part1 + part2)
        return (result - self._min_) / (self._max_ - self._min_)
