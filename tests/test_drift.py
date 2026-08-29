import numpy as np
import pandas as pd

from monitoring.drift import psi, total_variation


def test_psi_zero_for_same_distribution():
    expected = np.array([0.2, 0.3, 0.5])
    assert psi(expected, expected) == 0.0


def test_total_variation_zero_for_same_distribution():
    expected = {"A": 0.6, "B": 0.4}
    actual = pd.Series({"A": 0.6, "B": 0.4})
    assert total_variation(expected, actual) == 0.0
