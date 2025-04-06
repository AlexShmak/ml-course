import numpy as np


class Optimizer:
    """Base optimizer class."""

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def init_param_state(self, layers):
        """Initialize optimizer state for all parameters."""
        pass

    def update(self, layer, param_name):
        """Update a parameter based on its gradient."""
        raise NotImplementedError("Optimizer subclasses must implement update method")


class SGD(Optimizer):
    """Standard Stochastic Gradient Descent optimizer."""

    def update(self, layer, param_name):
        layer.params[param_name] -= self.learning_rate * layer.grads[param_name]


class Momentum(Optimizer):
    """Momentum optimizer."""

    def __init__(self, learning_rate=1e-3, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}  # Will store {layer_id: {param_name: velocity}}

    def init_param_state(self, layers):
        """Initialize velocity for all parameters."""
        self.velocity = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and layer.params:
                self.velocity[i] = {}
                for param_name, param in layer.params.items():
                    self.velocity[i][param_name] = np.zeros_like(param)

    def update(self, layer_idx, layer, param_name):
        """Update parameters using momentum."""
        # Update velocity
        self.velocity[layer_idx][param_name] = (
                self.momentum * self.velocity[layer_idx][param_name] -
                self.learning_rate * layer.grads[param_name]
        )
        # Update parameters
        layer.params[param_name] += self.velocity[layer_idx][param_name]


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0  # Timestep

    def init_param_state(self, layers):
        """Initialize moment estimates for all parameters."""
        self.m = {}
        self.v = {}
        self.t = 0
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params') and layer.params:
                self.m[i] = {}
                self.v[i] = {}
                for param_name, param in layer.params.items():
                    self.m[i][param_name] = np.zeros_like(param)
                    self.v[i][param_name] = np.zeros_like(param)

    def update(self, layer_idx, layer, param_name):
        """Update parameters using Adam."""
        # Increment timestep
        self.t += 1

        # Update biased first moment estimate
        self.m[layer_idx][param_name] = (
                self.beta1 * self.m[layer_idx][param_name] +
                (1 - self.beta1) * layer.grads[param_name]
        )

        # Update biased second raw moment estimate
        self.v[layer_idx][param_name] = (
                self.beta2 * self.v[layer_idx][param_name] +
                (1 - self.beta2) * np.square(layer.grads[param_name])
        )

        # Compute bias-corrected first moment estimate
        m_corrected = self.m[layer_idx][param_name] / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_corrected = self.v[layer_idx][param_name] / (1 - self.beta2 ** self.t)

        # Update parameters
        layer.params[param_name] -= (
                self.learning_rate * m_corrected /
                (np.sqrt(v_corrected) + self.epsilon)
        )
