import numpy as np

from howework.task6.utils.optimizer import SGD, Optimizer
from howework.task6.utils.utils import softmax_with_cross_entropy, l2_regularization


def gradient_check(model, X, y, epsilon=1e-7, threshold=1e-3):
    """
    Performs gradient checking on a neural network model.

    Parameters:
    model - Neural network model to check
    X - Input data
    y - Target labels
    epsilon - Small value for computing numerical gradient

    Returns:
    Dictionary containing the results of the check for each parameter
    """
    # Forward pass to compute loss
    original_scores = model.forward(X)
    original_loss, dscores = softmax_with_cross_entropy(original_scores, y)

    # Backward pass to compute analytical gradients
    model.backward(dscores)

    results = {}

    # Check gradients for each layer with parameters
    for layer_idx, layer in enumerate(model.layers):
        if not hasattr(layer, 'params') or not layer.params:
            continue

        layer_name = f"{layer.__class__.__name__}_{layer_idx}"
        results[layer_name] = {}

        for param_name, param in layer.params.items():
            # Get analytical gradient
            analytical_grad = layer.grads[param_name]

            # Verify dimensions match
            assert param.shape == analytical_grad.shape, \
                f"Shapes don't match: param {param.shape} vs grad {analytical_grad.shape}"

            # Compute numerical gradient
            numerical_grad = np.zeros_like(param)

            # Iterate through each parameter element
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index

                # Save the original value
                orig_value = param[idx]

                # Compute loss with parameter + epsilon
                param[idx] = orig_value + epsilon
                scores_plus = model.forward(X)
                loss_plus, _ = softmax_with_cross_entropy(scores_plus, y)

                # Compute loss with parameter - epsilon
                param[idx] = orig_value - epsilon
                scores_minus = model.forward(X)
                loss_minus, _ = softmax_with_cross_entropy(scores_minus, y)

                # Compute numerical gradient
                numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

                # Restore original value
                param[idx] = orig_value

                it.iternext()

            # Compare analytical and numerical gradients
            abs_diff = np.abs(analytical_grad - numerical_grad)
            rel_error = abs_diff / (np.maximum(np.abs(analytical_grad), np.abs(numerical_grad)) + 1e-8)

            # Compute max and average relative error
            max_error = np.max(rel_error)
            avg_error = np.mean(rel_error)

            results[layer_name][param_name] = {
                'max_error': max_error,
                'avg_error': avg_error,
                'passed': max_error < threshold
            }

    return results


def print_gradient_check_results(results):
    """
    Print the results of gradient checking in a readable format.
    """
    print("\n===== Gradient Check Results =====")
    all_passed = True

    for layer_name, layer_results in results.items():
        print(f"\nLayer: {layer_name}")
        for param_name, check_result in layer_results.items():
            status = "PASSED" if check_result['passed'] else "FAILED"
            all_passed = all_passed and check_result['passed']

            print(f"  Parameter: {param_name}")
            print(f"    Max Relative Error: {check_result['max_error']:.2e}")
            print(f"    Avg Relative Error: {check_result['avg_error']:.2e}")
            print(f"    Status: {status}")

    print("\nOverall Status:", "PASSED" if all_passed else "FAILED")
    print("=============================")

    return all_passed


def modify_fit_with_gradient_check(nn_model):
    """
    Modify the fit method to include gradient checking during training.
    This is a wrapper function that modifies the fit method temporarily.
    """
    original_fit = nn_model.fit

    def fit_with_gradient_check(X, y, learning_rate=1e-3, epochs=100, batch_size=32,
                                reg_strength=1e-5, check_freq=None, optimizer: Optimizer = None):
        """
        Modified fit method that includes gradient checking.

        Parameters:
        check_freq - If provided, check gradients every 'check_freq' epochs.
                    If None, check only at the beginning.
        """
        if optimizer is not None:
            nn_model.set_optimizer(optimizer)
        elif nn_model.optimizer is None:
            nn_model.set_optimizer(SGD(learning_rate))
        num_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Perform gradient check at specified frequency
            do_check = (check_freq is not None and epoch % check_freq == 0) or (check_freq is None and epoch == 0)
            if do_check:
                print(f"\nPerforming gradient check at epoch {epoch}...")
                # Use a small subset for gradient checking to make it faster
                check_size = min(batch_size, X.shape[0])
                check_indices = np.random.choice(X.shape[0], check_size, replace=False)
                X_check, y_check = X[check_indices], y[check_indices]

                results = gradient_check(nn_model, X_check, y_check)
                passed = print_gradient_check_results(results)

            # Rest of the original fit method
            loss_epoch = 0
            correct_preds = 0

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                scores = nn_model.forward(X_batch)
                loss, dscores = softmax_with_cross_entropy(scores, y_batch)

                for layer in nn_model.layers:
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
                nn_model.backward(dscores)

                # Update parameters using SGD
                for layer_idx, layer in enumerate(nn_model.layers):
                    if hasattr(layer, 'params') and layer.params:
                        for param_name in layer.params:
                            # For SGD, layer_idx isn't used
                            if isinstance(nn_model.optimizer, SGD):
                                nn_model.optimizer.update(layer, param_name)
                            else:
                                nn_model.optimizer.update(layer_idx, layer, param_name)

            nn_model.loss_history.append(loss_epoch)
            accuracy = correct_preds / num_samples
            nn_model.accuracy_history.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss_epoch:.4f}, accuracy = {accuracy:.4f}")

    nn_model.fit = fit_with_gradient_check
    return original_fit  # Return original to restore later if needed
