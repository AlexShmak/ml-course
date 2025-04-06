from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self):
        self.params = {}
        self.grads = {}

    @abstractmethod
    def forward(self, x):
        ...

    @abstractmethod
    def backward(self, d_out):
        ...


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, d_out):
        # inputs = self.cache
        # d_out[inputs <= 0] = 0
        # return d_out
        if self.cache is None:
            raise RuntimeError("Backward called before forward or cache cleared.")
        inputs = self.cache
        d_local = np.zeros_like(inputs)
        d_local[inputs > 0] = 1  # Gradient is 1 where input > 0
        dx = d_out * d_local  # Apply chain rule
        return dx


class FullyConnected(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.__initialize_parameters(input_size, output_size)
        self.cache = None

    def __initialize_parameters(self, input_size, output_size):
        self.params["W"] = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.params["b"] = np.zeros((1, output_size))

    def forward(self, x):
        self.cache = x
        return np.dot(x, self.params["W"]) + self.params["b"]

    def backward(self, d_out):
        inputs = self.cache
        self.grads["W"] = np.dot(inputs.T, d_out)
        # Fix: Reshape to match the shape of self.params["b"]
        self.grads["b"] = np.sum(d_out, axis=0, keepdims=True)
        dx = np.dot(d_out, self.params["W"].T)
        return dx


class BatchNorm(Layer):
    def __init__(self, num_features):
        super().__init__()
        # Initialize parameters (gamma, beta) and running stats (mean, var)
        self.params['gamma'] = np.ones((1, num_features))
        self.params['beta'] = np.zeros((1, num_features))
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.zeros((1, num_features))
        self.epsilon = 1e-5  # For numerical stability
        self.momentum = 0.9  # For running mean/variance update
        self.cache = None  # To store values needed for backward pass
        # Initialize grads structure
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])

    def forward(self, x):
        if x.ndim == 1:  # Handle single sample case
            x = x.reshape(1, -1)

        # 1. Calculate batch mean
        batch_mean = np.mean(x, axis=0, keepdims=True)
        # 2. Calculate batch variance
        batch_var = np.var(x, axis=0, keepdims=True)
        # 3. Normalize
        x_hat = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
        # 4. Scale and shift
        out = self.params['gamma'] * x_hat + self.params['beta']

        # Update running mean and variance (for inference)
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

        # Store intermediate values needed for backward pass
        self.cache = (x, x_hat, batch_mean, batch_var, self.epsilon, self.params['gamma'])

        return out

    def backward(self, d_out):

        if self.cache is None:  # If forward was never called in training

            raise RuntimeError("Cannot backprop through BatchNorm without prior training pass cache.")

        if self.cache is None:
            raise RuntimeError("Backward called before forward or cache cleared.")

        x, x_hat, batch_mean, batch_var, epsilon, gamma = self.cache
        N, D = d_out.shape

        # Gradients w.r.t. scale (gamma) and shift (beta)
        self.grads['beta'] = np.sum(d_out, axis=0, keepdims=True)
        self.grads['gamma'] = np.sum(d_out * x_hat, axis=0, keepdims=True)

        # Gradient w.r.t. normalized input x_hat
        dx_hat = d_out * gamma

        # Intermediate gradients for chain rule (leading to dx)
        # Gradient w.r.t variance
        dvar = np.sum(dx_hat * (x - batch_mean) * -0.5 * (batch_var + epsilon) ** (-1.5), axis=0, keepdims=True)
        # Gradient w.r.t mean
        dmean = np.sum(dx_hat * -1.0 / np.sqrt(batch_var + epsilon), axis=0, keepdims=True) + \
                dvar * np.mean(-2.0 * (x - batch_mean), axis=0, keepdims=True)

        # Gradient w.r.t. input x
        dx = dx_hat / np.sqrt(batch_var + epsilon) + \
             dvar * 2.0 * (x - batch_mean) / N + \
             dmean / N

        return dx
