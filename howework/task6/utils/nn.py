import numpy as np

from howework.task6.utils.layer import Layer
from howework.task6.utils.optimizer import SGD, Optimizer
from howework.task6.utils.utils import softmax_with_cross_entropy, l2_regularization


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_history = []
        self.accuracy_history = []
        self.optimizer = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        # Initialize optimizer state if needed
        if hasattr(self.optimizer, 'init_param_state'):
            self.optimizer.init_param_state(self.layers)

    def build(self, *args: Layer):
        for layer in args:
            self.layers.append(layer)

    def forward(self, X):
        inputs = X.copy()
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def fit(self, X, y, learning_rate=1e-3, epochs=100, batch_size=32, reg_strength=1e-5, optimizer: Optimizer = None):
        if optimizer is not None:
            self.set_optimizer(optimizer)
        elif self.optimizer is None:
            self.set_optimizer(SGD(learning_rate))

        num_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            loss_epoch = 0
            correct_preds = 0

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                scores = self.forward(X_batch)
                loss, dscores = softmax_with_cross_entropy(scores, y_batch)

                for layer in self.layers:
                    if hasattr(layer, 'params') and "W" in layer.params:
                        reg_loss, reg_grad = l2_regularization(layer.params["W"], reg_strength)
                        loss += reg_loss
                        # Store the regularization gradient into the layer's gradients
                        layer.grads["W"] = layer.grads.get("W", 0) + reg_grad

                loss_epoch += loss * (end_idx - start_idx) / num_samples

                # Calculate accuracy
                predicted_classes = np.argmax(scores, axis=1)
                correct_preds += np.sum(predicted_classes == y_batch)

                # Backward pass
                self.backward(dscores)

                # Update parameters using SGD
                # for layer in self.layers:
                #     if hasattr(layer, 'params') and layer.params:
                #         for param_name in layer.params:
                #             layer.params[param_name] -= learning_rate * layer.grads[param_name]
                for layer_idx, layer in enumerate(self.layers):
                    if hasattr(layer, 'params') and layer.params:
                        for param_name in layer.params:
                            # For SGD, layer_idx isn't used
                            if isinstance(self.optimizer, SGD):
                                self.optimizer.update(layer, param_name)
                            else:
                                self.optimizer.update(layer_idx, layer, param_name)

            self.loss_history.append(loss_epoch)
            accuracy = correct_preds / num_samples
            self.accuracy_history.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss_epoch:.4f}, accuracy = {accuracy:.4f}")

    def predict(self, X):
        scores = self.forward(X)
        return np.argmax(scores, axis=1)
