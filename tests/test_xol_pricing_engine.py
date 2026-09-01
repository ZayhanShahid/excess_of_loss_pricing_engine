"""Small checks for the core XoL calculations."""

import unittest

import numpy as np

from src.xol_pricing_engine import XoLLayer, layer_recovery


class TestLayerRecovery(unittest.TestCase):
    def test_recovery_below_inside_and_above_layer(self):
        layer = XoLLayer(limit=100_000, attachment=50_000)
        losses = np.array([25_000, 50_000, 80_000, 150_000, 250_000])
        expected = np.array([0, 0, 30_000, 100_000, 100_000])
        np.testing.assert_array_equal(layer_recovery(losses, layer), expected)

    def test_recovery_is_never_negative(self):
        layer = XoLLayer(limit=250_000, attachment=100_000)
        recoveries = layer_recovery(np.array([1, 50_000, 100_000]), layer)
        self.assertTrue((recoveries >= 0).all())


if __name__ == "__main__":
    unittest.main()
