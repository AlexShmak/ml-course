import numpy as np


def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(np.square(W[:-1]))
    grad = 2 * reg_strength * W
    grad[-1] = 0

    return loss, grad


def softmax(predictions):
    """
    Compute softmax values for each set of scores in predictions.
    """
    # Subtract max for numerical stability
    m = np.max(predictions, axis=1, keepdims=True)
    exps = np.exp(predictions - m)
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    """
    Compute cross entropy loss.
    """
    batch_size = probs.shape[0]
    # Get probability corresponding to the target class
    correct_probs = probs[np.arange(batch_size), target_index]
    # Compute log loss
    return -np.sum(np.log(correct_probs)) / batch_size


def softmax_with_cross_entropy(predictions, target_index):
    """
    Compute softmax and cross entropy loss.

    Arguments:
        predictions: array of shape (batch_size, num_classes)
        target_index: array of shape (batch_size,) with class indices

    Returns:
        loss: scalar value
        dprediction: gradient of shape (batch_size, num_classes)
    """
    batch_size = predictions.shape[0]
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    # Compute gradient
    dprediction = probs.copy()
    dprediction[np.arange(batch_size), target_index] -= 1
    # Normalize gradient
    dprediction /= batch_size

    return loss, dprediction
