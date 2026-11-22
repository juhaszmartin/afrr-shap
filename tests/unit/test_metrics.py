import unittest

import numpy as np

from src.afrr_shap.modeling.metrics import smape, target_groups


class SMAPETests(unittest.TestCase):
    def test_basic_smape(self):
        """SMAPE formula: 200 * mean(|y_pred - y_true| / (|y_true| + |y_pred|))"""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        result = smape(y_true, y_pred)

        # Standard SMAPE: 100 * mean(2 * |y_pred - y_true| / (|y_true| + |y_pred|))
        # = 200 * mean(|y_pred - y_true| / (|y_true| + |y_pred|))
        expected = 200 * np.mean(
            np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
        )
        self.assertAlmostEqual(result, expected)

    def test_smape_perfect_prediction(self):
        """SMAPE is 0 for perfect predictions."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])

        result = smape(y_true, y_pred)
        self.assertEqual(result, 0.0)

    def test_smape_with_zeros(self):
        """SMAPE handles zeros in both true and pred (eps prevents div/0)."""
        y_true = np.array([0.0, 100.0, 0.0])
        y_pred = np.array([0.0, 100.0, 10.0])

        result = smape(y_true, y_pred)
        # First: both 0 -> 0
        # Second: perfect -> 0
        # Third: 200 * |10-0|/(|0|+|10|+eps) ≈ 200 * 10/10 = 200
        # mean ≈ 200/3 ≈ 66.67
        self.assertAlmostEqual(result, 66.666, places=1)

    def test_smape_multidimensional(self):
        """SMAPE works column-wise for 2D arrays (multiple targets)."""
        y_true = np.array([[100.0, 200.0], [150.0, 250.0], [200.0, 300.0]])
        y_pred = np.array([[110.0, 190.0], [140.0, 260.0], [210.0, 290.0]])

        result = smape(y_true, y_pred)

        self.assertEqual(result.shape, (2,))
        # Check each column independently
        for i in range(2):
            expected = 200 * np.mean(
                np.abs(y_pred[:, i] - y_true[:, i])
                / (np.abs(y_true[:, i]) + np.abs(y_pred[:, i]))
            )
            self.assertAlmostEqual(result[i], expected)

    def test_smape_symmetric(self):
        """SMAPE is symmetric: smape(a, b) == smape(b, a)."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([150.0, 180.0, 350.0])

        result1 = smape(y_true, y_pred)
        result2 = smape(y_pred, y_true)

        self.assertAlmostEqual(result1, result2)


class TargetGroupsTests(unittest.TestCase):
    def test_groups_volume_targets(self):
        """Imbalance volume targets get 'volume [MWh]' group."""
        target_names = [
            "imbalance_volumes - value_t+15min",
            "imbalance_volumes - value_t+30min",
        ]
        y_test = np.array([[10.0, 20.0], [15.0, 25.0]])

        groups = target_groups(target_names, y_test)

        self.assertEqual(groups[target_names[0]], "volume [MWh]")
        self.assertEqual(groups[target_names[1]], "volume [MWh]")

    def test_groups_long_price(self):
        """Long price targets get 'price Long [EUR/MWh]' group."""
        target_names = [
            "imbalance_prices - Long_t+15min",
            "imbalance_prices - Long_t+30min",
        ]
        y_test = np.array([[50.0, 60.0], [55.0, 65.0]])

        groups = target_groups(target_names, y_test)

        self.assertEqual(groups[target_names[0]], "price Long [EUR/MWh]")
        self.assertEqual(groups[target_names[1]], "price Long [EUR/MWh]")

    def test_groups_short_price_single_pricing(self):
        """Short price gets 'price Short = Long' when values match Long."""
        target_names = [
            "imbalance_prices - Long_t+15min",
            "imbalance_prices - Short_t+15min",
        ]
        # Short equals Long (single pricing)
        y_test = np.array([[50.0, 50.0], [55.0, 55.0], [60.0, 60.0]])

        groups = target_groups(target_names, y_test)

        self.assertEqual(groups[target_names[0]], "price Long [EUR/MWh]")
        self.assertEqual(groups[target_names[1]], "price Short = Long (single pricing)")

    def test_groups_short_price_dual_pricing(self):
        """Short price gets 'price Short [EUR/MWh]' when values differ from Long."""
        target_names = [
            "imbalance_prices - Long_t+15min",
            "imbalance_prices - Short_t+15min",
        ]
        # Short differs from Long (dual pricing)
        y_test = np.array([[50.0, 40.0], [55.0, 45.0], [60.0, 50.0]])

        groups = target_groups(target_names, y_test)

        self.assertEqual(groups[target_names[0]], "price Long [EUR/MWh]")
        self.assertEqual(groups[target_names[1]], "price Short [EUR/MWh]")

    def test_single_pricing_threshold(self):
        """Single pricing requires >= 99% match."""
        target_names = [
            "imbalance_prices - Long_t+15min",
            "imbalance_prices - Short_t+15min",
        ]
        # 99% match (297/300 = 0.99)
        y_test = np.array([[50.0, 50.0]] * 297 + [[50.0, 51.0]] * 3)

        groups = target_groups(target_names, y_test)

        self.assertEqual(groups[target_names[1]], "price Short = Long (single pricing)")

    def test_dual_pricing_below_threshold(self):
        """Below 99% match -> dual pricing."""
        target_names = [
            "imbalance_prices - Long_t+15min",
            "imbalance_prices - Short_t+15min",
        ]
        # 98% match (294/300 = 0.98)
        y_test = np.array([[50.0, 50.0]] * 294 + [[50.0, 51.0]] * 6)

        groups = target_groups(target_names, y_test)

        self.assertEqual(groups[target_names[1]], "price Short [EUR/MWh]")


if __name__ == "__main__":
    unittest.main()
