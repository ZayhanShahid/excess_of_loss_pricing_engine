"""Checks for the core XoL calculations and their important error paths."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.xol_pricing_engine import (
    SeverityModel,
    XoLLayer,
    calculate_technical_premium,
    fit_frequency_model,
    fit_severity_model,
    historical_burning_costs,
    layer_recovery,
    load_data,
    sample_severities,
    simulate_annual_claim_counts,
    simulate_annual_recoveries,
    validate_data,
)


def valid_claims() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PolicyNum": [1, 2],
            "Year": [2010, 2010],
            "Claim": [75_000.0, 200_000.0],
            "Deduct": [1_000.0, 1_000.0],
        }
    )


def valid_policies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PolicyNum": [1, 2],
            "Year": [2010, 2010],
            "BCcov": [1_000_000.0, 2_000_000.0],
            "Freq": [1, 1],
        }
    )


def deterministic_severity(amount: float) -> SeverityModel:
    return SeverityModel(
        threshold=amount,
        threshold_quantile=0.95,
        tail_probability=0.0,
        shape=0.0,
        scale=1.0,
        body_values=np.array([amount]),
        tail_excesses=np.array([], dtype=float),
        historical_maximum=amount,
    )


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


class TestDataValidation(unittest.TestCase):
    def test_missing_file_error_names_the_required_files(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "CLAIMLEVEL.csv"):
                load_data(Path(directory))

    def test_missing_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing claim columns"):
            validate_data(valid_claims().drop(columns="Claim"), valid_policies())

    def test_non_positive_claim_is_rejected(self):
        claims = valid_claims()
        claims.loc[0, "Claim"] = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_data(claims, valid_policies())

    def test_claim_year_without_exposure_is_rejected(self):
        claims = valid_claims()
        claims.loc[0, "Year"] = 2009
        with self.assertRaisesRegex(ValueError, "no matching policy exposure"):
            validate_data(claims, valid_policies())


class TestFrequencyModel(unittest.TestCase):
    def test_method_of_moments_fit_on_known_data(self):
        policies = pd.DataFrame(
            {
                "Year": [2010, 2010, 2010, 2010],
                "BCcov": [1.0, 1.0, 1.0, 1.0],
                "Freq": [0, 0, 0, 6],
            }
        )
        model = fit_frequency_model(policies)
        self.assertAlmostEqual(model["claim_rate_per_dollar"], 1.5)
        self.assertAlmostEqual(model["dispersion"], 7.0 / 3.0)
        self.assertAlmostEqual(model["expected_annual_claims"], 6.0)

    def test_poisson_branch_has_expected_mean_and_variance(self):
        model = {"target_means": np.array([2.0, 3.0]), "dispersion": 0.0}
        totals = simulate_annual_claim_counts(
            model,
            simulations=50_000,
            rng=np.random.default_rng(7),
            batch_size=500,
        )
        self.assertAlmostEqual(float(totals.mean()), 5.0, delta=0.06)
        self.assertAlmostEqual(float(totals.var()), 5.0, delta=0.15)


class TestSeverityModel(unittest.TestCase):
    def test_gpd_tail_fit_records_selected_exceedances(self):
        severities = np.geomspace(100.0, 1_000_000.0, 1_000)
        model = fit_severity_model(
            severities,
            threshold_quantile=0.90,
            minimum_exceedances=50,
        )
        self.assertEqual(model.exceedance_count, 100)
        self.assertGreater(model.scale, 0)
        self.assertAlmostEqual(model.tail_probability, 0.10)

    def test_parametric_tail_can_exceed_historical_maximum(self):
        model = SeverityModel(
            threshold=100.0,
            threshold_quantile=0.95,
            tail_probability=1.0,
            shape=0.2,
            scale=100.0,
            body_values=np.array([50.0]),
            tail_excesses=np.ones(50),
            historical_maximum=101.0,
        )
        sampled = sample_severities(model, 1_000, np.random.default_rng(9))
        self.assertGreater(float(sampled.max()), model.historical_maximum)
        self.assertTrue((sampled > model.threshold).all())


class TestSimulationAndPricing(unittest.TestCase):
    def test_recovery_aggregation_with_fixed_severity(self):
        layer = XoLLayer(limit=100_000, attachment=50_000)
        recoveries = simulate_annual_recoveries(
            claim_counts=np.array([1, 2, 0]),
            severity_model=deterministic_severity(200_000.0),
            layers=(layer,),
            rng=np.random.default_rng(1),
            batch_size=2,
        )
        np.testing.assert_array_equal(
            recoveries[layer.name],
            np.array([100_000.0, 200_000.0, 0.0]),
        )

    def test_volatility_increases_technical_premium(self):
        stable = calculate_technical_premium(1_000_000, 100_000)
        volatile = calculate_technical_premium(1_000_000, 500_000)
        self.assertGreater(volatile, stable)

    def test_burning_cost_reports_missing_exposure_year(self):
        claims = valid_claims()
        claims.loc[0, "Year"] = 2009
        layer = XoLLayer(limit=100_000, attachment=50_000)
        with self.assertRaisesRegex(ValueError, "exposure is missing"):
            historical_burning_costs(claims, valid_policies(), (layer,))


if __name__ == "__main__":
    unittest.main()
