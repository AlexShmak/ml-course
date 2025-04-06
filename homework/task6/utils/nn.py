import numpy as np

from homework.task6.utils.layer import Layer
from homework.task6.utils.optimizer import SGD, Optimizer
from homework.task6.utils.utils import softmax_with_cross_entropy, l2_regularization


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_history = []
        self.accuracy_history = []
        self.optimizer = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        if hasattr(self.optimizer, "init_param_state"):
            self.optimizer.init_param_state(self.layers)

    def build(self, *layers: Layer):
        self.layers.extend(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def _apply_regularization(self, reg_strength):
        total_reg_loss = 0
        for layer in self.layers:
            if "W" in getattr(layer, "params", {}):
                reg_loss, reg_grad = l2_regularization(layer.params["W"], reg_strength)
                total_reg_loss += reg_loss
                layer.grads["W"] = layer.grads.get("W", 0) + reg_grad
        return total_reg_loss

    def fit(
            self,
            X,
            y,
            learning_rate=1e-3,
            epochs=100,
            batch_size=32,
            reg_strength=1e-5,
            optimizer: Optimizer = None,
    ):
        self.set_optimizer(optimizer or SGD(learning_rate))
        num_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            loss_epoch, correct_preds = 0, 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]

                scores = self.forward(X_batch)
                loss, dscores = softmax_with_cross_entropy(scores, y_batch)

                loss += self._apply_regularization(reg_strength)
                loss_epoch += loss * (end - start) / num_samples

                predicted_classes = np.argmax(scores, axis=1)
                correct_preds += np.sum(predicted_classes == y_batch)

                self.backward(dscores)

                for idx, layer in enumerate(self.layers):
                    if layer.params:
                        for name in layer.params:
                            update_fn = self.optimizer.update
                            update_fn(layer if isinstance(self.optimizer, SGD) else idx, layer, name)

            self.loss_history.append(loss_epoch)
            accuracy = correct_preds / num_samples
            self.accuracy_history.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss_epoch:.4f}, accuracy = {accuracy:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
