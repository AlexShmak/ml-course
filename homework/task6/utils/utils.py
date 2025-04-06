import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss and gradient.

    Arguments:
        W: weights matrix
        reg_strength: regularization strength (lambda)

    Returns:
        loss: L2 loss value
        grad: gradient of L2 loss w.r.t. W
    """
    loss = reg_strength * np.sum(W * W)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    """
    Computes softmax probabilities.

    Arguments:
        predictions: array of shape (N, C) or (C,) — raw scores

    Returns:
        probs: same shape, softmax probabilities
    """
    predictions = np.atleast_2d(predictions)
    max_scores = np.max(predictions, axis=1, keepdims=True)
    exps = np.exp(predictions - max_scores)
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    return probs if predictions.shape[0] > 1 else probs.squeeze()


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss.

    Arguments:
        probs: probabilities after softmax (N, C)
        target_index: true class indices (int or array)

    Returns:
        loss: scalar
    """
    probs = np.atleast_2d(probs)
    target_index = np.atleast_1d(target_index)
    correct_log_probs = -np.log(probs[np.arange(len(probs)), target_index])
    return np.mean(correct_log_probs)


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax, cross-entropy loss, and gradient.

    Arguments:
        predictions: raw model outputs (N, C)
        target_index: ground-truth class indices

    Returns:
        loss: scalar
        dprediction: gradient w.r.t. predictions (same shape)
    """
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    dprediction = probs.copy()
    batch_size = probs.shape[0]
    dprediction[np.arange(batch_size), target_index] -= 1
    dprediction /= batch_size

    return loss, dprediction
