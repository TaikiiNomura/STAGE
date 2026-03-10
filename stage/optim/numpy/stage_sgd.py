# stage\optim\numpy\base_optimizer.py

import numpy as np

from .base_optimizer import MyNumpyOptimizer

class StageSGD(MyNumpyOptimizer):
    def __init__(
            self,
            lr: float = 0.01,
            tau: float = 1.0,
            eps: float = 1e-3,
    ):
        super().__init__(lr=lr)
        self.tau = tau
        self.eps = eps

    def update(
            self,
            params: np.ndarray,
            grads: np.ndarray,
    ) -> None:
        with np.errstate(over='ignore'):
            scaled_grads = 1.0 / np.cosh(grads * self.tau)
        
        scale = np.maximum(scaled_grads, self.eps)

        params -= self.lr * grads * scale