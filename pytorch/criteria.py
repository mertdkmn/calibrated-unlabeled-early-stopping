"""
Simple math helpers for the BUS-stop reimplementation.

These functions do not need PyTorch or TensorFlow, so we can test them alone.
"""

import numpy as np
from scipy.special import softmax as _softmax


def softmax(logits, axis=1):
    # Turn raw scores (logits) into probabilities that add up to 1.
    return _softmax(np.asarray(logits, dtype=float), axis=axis)


def accuracy(y_true, y_pred):
    # How many predictions are correct (between 0 and 1).
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred):
    # Macro F1: average F1 over all classes (each class counts the same).
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=1))


def class_calibration(unl_dist, smp_bias, val_perf, num_labels):
    """
    Fix the predicted class distribution so it is closer to the true one.

    This is Equation (2) in the paper:
        Cu ~ B + (1 - Acc_min) / (Acc_val - Acc_min) * (Cu_hat - B)

    unl_dist   : class distribution predicted on the unlabeled data
    smp_bias   : the class distribution the model tends to output (a bias)
    val_perf   : validation score (accuracy or F1)
    num_labels : number of classes
    """
    smp_bias = np.asarray(smp_bias, dtype=float)
    unl_dist = np.asarray(unl_dist, dtype=float)

    # Make each vector add up to 1.
    smp_bias = smp_bias / np.sum(smp_bias)
    unl_dist = unl_dist / np.sum(unl_dist)

    # Acc_min is the accuracy of random guessing (0.5 for two classes).
    acc_min = 1.0 / num_labels

    # How far the model is from random guessing. Bigger means more confident.
    frac = (1.0 - acc_min) / (val_perf - acc_min)

    # Move from the bias toward the predicted distribution by this factor.
    cali_dist = smp_bias + frac * (unl_dist - smp_bias)

    # Probabilities cannot be below 0.
    return np.clip(cali_dist, 0.0, 1.0)


def downsample_indices(unl_len, lab_len):
    # Pick evenly spaced indices, so we get about lab_len points out of unl_len.
    # (This copies what the original code does.)
    return np.arange(0, unl_len, unl_len / lab_len).astype("int32")


def euclidean(a, b):
    # Normal Euclidean distance between two vectors.
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def cosine_sim(a, b):
    # Cosine similarity: 1 means same direction, 0 means unrelated.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def evenly_shuffle_indices(y):
    """
    Return indices that mix the classes evenly.

    Example: labels [0, 0, 1, 1] gives order [0, 2, 1, 3],
    so any prefix of the list has about the same number of each class.
    """
    y = [int(v) for v in y]
    classes = sorted(set(y))

    all_idx = []
    marks = []
    for c in classes:
        # Positions where this class appears.
        idxs = [i for i, v in enumerate(y) if v == c]
        all_idx.extend(idxs)
        # Give each position a height from 0 to 1, so classes interleave.
        marks.extend([i / len(idxs) for i in range(len(idxs))])

    # Sort by height to get the even mix.
    order = np.argsort(marks, kind="stable")
    return [all_idx[i] for i in order]


def ema(prev, value, alpha):
    """
    Smooth a value over time with an exponential moving average (EMA).

    prev  : previous smoothed value (None on the first call)
    value : the new raw value
    alpha : how much to trust the new value
            (1.0 = no smoothing, 0.0 = ignore the new value)

    Used to reduce noise in the stopping signals (improvement #1).
    """
    if prev is None:
        return float(value)
    return alpha * float(value) + (1.0 - alpha) * float(prev)
