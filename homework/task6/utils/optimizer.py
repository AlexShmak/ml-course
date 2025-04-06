import numpy as np


class Optimizer:
    """Base optimizer class."""

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def init_param_state(self, layers):
        """Initialize optimizer-specific state (if needed)."""
        pass

    def update(self, layer_idx, layer, param_name):
        """Update parameters — must be implemented by subclass."""
        raise NotImplementedError("Optimizer subclasses must implement update method")


class SGD(Optimizer):
    """Standard Stochastic Gradient Descent optimizer."""

    def update(self, layer_idx, layer, param_name):
        layer.params[param_name] -= self.learning_rate * layer.grads[param_name]


class Momentum(Optimizer):
    """SGD with Momentum optimizer."""

    def __init__(self, learning_rate=1e-3, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def init_param_state(self, layers):
        """Initialize velocity tensors for all layer parameters."""
        self.velocity = {
            i: {name: np.zeros_like(param)
                for name, param in layer.params.items()}
            for i, layer in enumerate(layers)
            if hasattr(layer, 'params')
        }

    def update(self, layer_idx, layer, param_name):
        v = self.velocity[layer_idx][param_name]
        g = layer.grads[param_name]

        v[:] = self.momentum * v - self.learning_rate * g
        layer.params[param_name] += v


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer."""

    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m, self.v = {}, {}
        self.t = 0

    def init_param_state(self, layers):
        """Initialize first and second moment vectors."""
        self.m = {}
        self.v = {}
        self.t = 0
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params'):
                self.m[i] = {}
                self.v[i] = {}
                for name, param in layer.params.items():
                    self.m[i][name] = np.zeros_like(param)
                    self.v[i][name] = np.zeros_like(param)

    def update(self, layer_idx, layer, param_name):
        self.t += 1

        grad = layer.grads[param_name]
        m, v = self.m[layer_idx][param_name], self.v[layer_idx][param_name]

        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v[:] = self.beta2 * v + (1 - self.beta2) * np.square(grad)

        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)

        layer.params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
