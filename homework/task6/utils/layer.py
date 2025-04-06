from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self):
        self.params = {}
        self.grads = {}

    @abstractmethod
    def forward(self, x): ...

    @abstractmethod
    def backward(self, d_out): ...


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, d_out):
        dx = d_out * (self.cache > 0)
        return dx


class FullyConnected(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._init_params(input_size, output_size)
        self.cache = None

    def _init_params(self, input_size, output_size):
        self.params["W"] = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.params["b"] = np.zeros((1, output_size))

    def forward(self, x):
        self.cache = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, d_out):
        x = self.cache
        self.grads["W"] = x.T @ d_out
        self.grads["b"] = np.sum(d_out, axis=0, keepdims=True)
        return d_out @ self.params["W"].T


class BatchNorm(Layer):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.params["gamma"] = np.ones((1, num_features))
        self.params["beta"] = np.zeros((1, num_features))
        self.grads["gamma"] = np.zeros_like(self.params["gamma"])
        self.grads["beta"] = np.zeros_like(self.params["beta"])
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.zeros((1, num_features))
        self.epsilon = epsilon
        self.momentum = momentum
        self.cache = None

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        std = np.sqrt(var + self.epsilon)
        x_hat = (x - mean) / std
        out = self.params["gamma"] * x_hat + self.params["beta"]

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        self.cache = (x, x_hat, mean, var, std, self.params["gamma"])

        return out

    def backward(self, d_out):
        if self.cache is None:
            raise RuntimeError("Backward called before forward or cache cleared.")

        x, x_hat, mean, var, std, gamma = self.cache
        N = x.shape[0]

        self.grads["beta"] = np.sum(d_out, axis=0, keepdims=True)
        self.grads["gamma"] = np.sum(d_out * x_hat, axis=0, keepdims=True)

        dx_hat = d_out * gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * std**-3, axis=0, keepdims=True)
        dmean = np.sum(dx_hat * -1 / std, axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=0, keepdims=True)
        dx = dx_hat / std + dvar * 2 * (x - mean) / N + dmean / N

        return dx
