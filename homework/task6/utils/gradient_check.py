import numpy as np

from homework.task6.utils.optimizer import SGD, Optimizer
from homework.task6.utils.utils import softmax_with_cross_entropy, l2_regularization


def gradient_check(model, X, y, epsilon=1e-7, threshold=1e-3):
    """
    Performs gradient checking on a neural network model.
    """
    original_scores = model.forward(X)
    original_loss, dscores = softmax_with_cross_entropy(original_scores, y)
    model.backward(dscores)

    results = {}

    for layer_idx, layer in enumerate(model.layers):
        if not hasattr(layer, 'params') or not layer.params:
            continue

        layer_name = f"{layer.__class__.__name__}_{layer_idx}"
        results[layer_name] = {}

        for param_name, param in layer.params.items():
            analytical_grad = layer.grads[param_name]
            assert param.shape == analytical_grad.shape, \
                f"Shape mismatch: param {param.shape} vs grad {analytical_grad.shape}"

            numerical_grad = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])

            while not it.finished:
                idx = it.multi_index
                orig_value = param[idx]

                param[idx] = orig_value + epsilon
                loss_plus = softmax_with_cross_entropy(model.forward(X), y)[0]

                param[idx] = orig_value - epsilon
                loss_minus = softmax_with_cross_entropy(model.forward(X), y)[0]

                numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                param[idx] = orig_value
                it.iternext()

            abs_diff = np.abs(analytical_grad - numerical_grad)
            rel_error = abs_diff / (np.maximum(np.abs(analytical_grad), np.abs(numerical_grad)) + 1e-8)

            results[layer_name][param_name] = {
                'max_error': np.max(rel_error),
                'avg_error': np.mean(rel_error),
                'passed': np.max(rel_error) < threshold
            }

    return results


def print_gradient_check_results(results):
    """
    Print gradient checking results in a structured format.
    """
    print("\n===== Gradient Check Results =====")
    all_passed = True

    for layer_name, layer_results in results.items():
        print(f"\nLayer: {layer_name}")
        for param_name, res in layer_results.items():
            status = "PASSED" if res['passed'] else "FAILED"
            all_passed &= res['passed']
            print(f"  Parameter: {param_name}")
            print(f"    Max Relative Error: {res['max_error']:.2e}")
            print(f"    Avg Relative Error: {res['avg_error']:.2e}")
            print(f"    Status: {status}")

    print("\nOverall Status:", "PASSED" if all_passed else "FAILED")
    print("=============================")
    return all_passed


def modify_fit_with_gradient_check(nn_model):
    """
    Wraps the fit method to add gradient checking at configurable intervals.
    """
    original_fit = nn_model.fit

    def fit_with_gradient_check(X, y, learning_rate=1e-3, epochs=100, batch_size=32,
                                reg_strength=1e-5, check_freq=None, optimizer: Optimizer = None):

        if optimizer is not None:
            nn_model.set_optimizer(optimizer)
        elif nn_model.optimizer is None:
            nn_model.set_optimizer(SGD(learning_rate))

        num_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            # Gradient checking condition
            do_check = (check_freq and epoch % check_freq == 0) or (check_freq is None and epoch == 0)
            if do_check:
                print(f"\nPerforming gradient check at epoch {epoch}...")
                idx = np.random.choice(num_samples, min(batch_size, num_samples), replace=False)
                results = gradient_check(nn_model, X[idx], y[idx])
                print_gradient_check_results(results)

            loss_epoch, correct_preds = 0.0, 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]

                scores = nn_model.forward(X_batch)
                loss, dscores = softmax_with_cross_entropy(scores, y_batch)

                for layer in nn_model.layers:
                    if hasattr(layer, 'params') and "W" in layer.params:
                        reg_loss, reg_grad = l2_regularization(layer.params["W"], reg_strength)
                        loss += reg_loss
                        layer.grads["W"] = layer.grads.get("W", 0) + reg_grad

                loss_epoch += loss * (end - start) / num_samples
                correct_preds += np.sum(np.argmax(scores, axis=1) == y_batch)

                nn_model.backward(dscores)

                for idx, layer in enumerate(nn_model.layers):
                    if hasattr(layer, 'params'):
                        for pname in layer.params:
                            if isinstance(nn_model.optimizer, SGD):
                                nn_model.optimizer.update(idx, layer, pname)
                            else:
                                nn_model.optimizer.update(idx, layer, pname)

            nn_model.loss_history.append(loss_epoch)
            accuracy = correct_preds / num_samples
            nn_model.accuracy_history.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss_epoch:.4f}, accuracy = {accuracy:.4f}")

    nn_model.fit = fit_with_gradient_check
    return original_fit
