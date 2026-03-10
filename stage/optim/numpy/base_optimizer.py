# stage\optim\numpy\base_optimizer.py

import numpy as np

class MyNumpyOptimizer:
    def __init__(self, lr: float = 0.01):
        self.lr = lr
    
    def my_update(
            self,
            params: np.ndarray,
            grads: np.ndarray,
    ) -> None:
        """
            params: 更新する重みパラメータ
            grads: 勾配
        """
        raise NotImplementedError('This method should be implemented by subclasses.')