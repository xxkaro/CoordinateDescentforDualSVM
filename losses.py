from abc import ABC, abstractmethod

import numpy as np


class SVMLoss(ABC):

    @abstractmethod
    def dual_params(self, C: float) -> tuple[float, float]:
        ...

    @abstractmethod
    def primal_loss(self, margins: np.ndarray) -> float:
        ...


class L1Loss(SVMLoss):
    """Hinge loss: max(1 - y * w^T x, 0)."""

    def dual_params(self, C: float) -> tuple[float, float]:
        U = C
        Dii = 0
        return U, Dii

    def primal_loss(self, margins: np.ndarray) -> float:
        return np.maximum(1.0 - margins, 0.0)


class L2Loss(SVMLoss):
    """Squared hinge loss: max(1 - y * w^T x, 0)^2."""

    def dual_params(self, C: float) -> tuple[float, float]:
        U = np.inf
        Dii = 1.0 / (2.0 * C)
        return U, Dii

    def primal_loss(self, margins: np.ndarray) -> float:
        return np.maximum(1.0 - margins, 0.0) ** 2


LOSS_REGISTRY: dict[str, type[SVMLoss]] = {
    "l1": L1Loss,
    "l2": L2Loss,
}