"""
Small tests for the math helpers. They need only numpy and scipy.

Run from anywhere with:
    python tests/test_criteria.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from criteria import (
    class_calibration,
    cosine_sim,
    downsample_indices,
    ema,
    evenly_shuffle_indices,
    euclidean,
)


def test_class_calibration():
    # Example from the paper (Figure 1):
    # two classes, random accuracy 0.5, validation score 0.8,
    # predicted distribution (0.65, 0.35), bias (0.5, 0.5)
    # -> the calibrated distribution should be (0.75, 0.25).
    cali = class_calibration([0.65, 0.35], [0.5, 0.5], 0.8, 2)
    assert np.allclose(cali, [0.75, 0.25], atol=1e-3), cali
    print("OK class_calibration:", np.round(cali, 4))


def test_downsample_indices():
    # 1822 samples down to about 100 points.
    ids = downsample_indices(1822, 100)
    assert len(ids) == 100, len(ids)
    assert ids[0] == 0 and ids.max() < 1822
    print("OK downsample_indices: len", len(ids), "range", ids.min(), ids.max())


def test_evenly_shuffle():
    # Two of each class should come out mixed.
    order = evenly_shuffle_indices([0, 0, 1, 1])
    assert order == [0, 2, 1, 3], order
    print("OK evenly_shuffle_indices:", order)


def test_similarities():
    # Same vectors -> cosine 1. Different vectors -> cosine 0.
    assert abs(cosine_sim([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine_sim([1, 0], [0, 1])) < 1e-9
    # The distance from (0, 0) to (3, 4) is 5.
    assert abs(euclidean([0, 0], [3, 4]) - 5.0) < 1e-9
    print("OK cosine_sim / euclidean")


def test_ema():
    # First call just returns the value.
    assert ema(None, 10.0, 0.5) == 10.0
    # alpha=0.5 -> halfway between old and new.
    assert abs(ema(10.0, 20.0, 0.5) - 15.0) < 1e-9
    # alpha=1.0 -> no smoothing (raw value).
    assert abs(ema(10.0, 20.0, 1.0) - 20.0) < 1e-9
    # alpha=0.0 -> keep the old value.
    assert abs(ema(10.0, 20.0, 0.0) - 10.0) < 1e-9
    print("OK ema")


if __name__ == "__main__":
    test_class_calibration()
    test_downsample_indices()
    test_evenly_shuffle()
    test_similarities()
    test_ema()
    print("ALL TESTS PASSED")
