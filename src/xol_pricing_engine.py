"""Core calculations for the excess-of-loss reinsurance pricing project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class XoLLayer:
    """A per-claim excess-of-loss layer written as limit xs attachment."""

    limit: float
    attachment: float

    @property
    def name(self) -> str:
        return f"{self.limit / 1_000:,.0f}k xs {self.attachment / 1_000:,.0f}k"


DEFAULT_LAYERS = (
    XoLLayer(limit=100_000, attachment=50_000),
    XoLLayer(limit=250_000, attachment=100_000),
    XoLLayer(limit=500_000, attachment=250_000),
)


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the original claim-level and policy-year CSV files."""

    data_dir = Path(data_dir)
    claims = pd.read_csv(data_dir / "CLAIMLEVEL.csv")
    policies = pd.read_csv(data_dir / "PropertyFundInsample.csv")
    return claims, policies


def validate_data(claims: pd.DataFrame, policies: pd.DataFrame) -> None:
    """Raise a clear error if the input files do not match the expected structure."""

    claim_columns = {"PolicyNum", "Year", "Claim", "Deduct"}
    policy_columns = {"PolicyNum", "Year", "BCcov", "Freq"}
    missing_claims = claim_columns.difference(claims.columns)
    missing_policies = policy_columns.difference(policies.columns)

    if missing_claims or missing_policies:
        raise ValueError(
            f"Missing claim columns: {sorted(missing_claims)}; "
            f"missing policy columns: {sorted(missing_policies)}"
        )
    if (claims["Claim"] <= 0).any():
        raise ValueError("Claim amounts must be positive.")
    if (policies["BCcov"] <= 0).any() or (policies["Freq"] < 0).any():
        raise ValueError("Coverage must be positive and frequency cannot be negative.")


def layer_recovery(losses: np.ndarray, layer: XoLLayer) -> np.ndarray:
    """Calculate the reinsurer's recovery for each individual claim."""

    return np.minimum(np.maximum(losses - layer.attachment, 0.0), layer.limit)


def fit_frequency_model(policies: pd.DataFrame, target_year: int = 2010) -> dict:
    """Fit a transparent exposure-adjusted Negative Binomial frequency model.

    Expected claims are proportional to building and contents coverage (BCcov).
    The Negative Binomial dispersion is estimated by the method of moments.
    """

    claim_rate = policies["Freq"].sum() / policies["BCcov"].sum()
    fitted_mean = claim_rate * policies["BCcov"]
    numerator = ((policies["Freq"] - fitted_mean) ** 2 - policies["Freq"]).sum()
    denominator = (fitted_mean**2).sum()
    dispersion = max(float(numerator / denominator), 0.0)

    target = policies.loc[policies["Year"] == target_year].copy()
    target_means = (claim_rate * target["BCcov"]).to_numpy()

    return {
        "claim_rate_per_dollar": float(claim_rate),
        "dispersion": dispersion,
        "target_year": target_year,
        "target_policy_count": int(len(target)),
        "target_means": target_means,
        "expected_annual_claims": float(target_means.sum()),
        "expected_annual_variance": float(
            np.sum(target_means + dispersion * target_means**2)
        ),
    }


def simulate_annual_claim_counts(
    frequency_model: dict,
    simulations: int,
    rng: np.random.Generator,
    batch_size: int = 500,
) -> np.ndarray:
    """Simulate total annual claim counts for the target portfolio."""

    means = frequency_model["target_means"]
    dispersion = frequency_model["dispersion"]
    totals = np.empty(simulations, dtype=np.int64)

    for start in range(0, simulations, batch_size):
        stop = min(start + batch_size, simulations)
        rows = stop - start
        if dispersion == 0:
            draws = rng.poisson(means, size=(rows, len(means)))
        else:
            shape = 1.0 / dispersion
            probability = shape / (shape + means)
            draws = rng.negative_binomial(shape, probability, size=(rows, len(means)))
        totals[start:stop] = draws.sum(axis=1)

    return totals


def simulate_annual_recoveries(
    claim_counts: np.ndarray,
    historical_severities: np.ndarray,
    layers: tuple[XoLLayer, ...],
    rng: np.random.Generator,
    batch_size: int = 500,
) -> dict[str, np.ndarray]:
    """Bootstrap severities and aggregate per-claim recoveries by simulated year."""

    simulations = len(claim_counts)
    recoveries = {layer.name: np.zeros(simulations) for layer in layers}

    for start in range(0, simulations, batch_size):
        stop = min(start + batch_size, simulations)
        counts = claim_counts[start:stop]
        event_count = int(counts.sum())
        sampled_losses = rng.choice(historical_severities, size=event_count, replace=True)
        simulation_ids = np.repeat(np.arange(stop - start), counts)

        for layer in layers:
            event_recovery = layer_recovery(sampled_losses, layer)
            recoveries[layer.name][start:stop] = np.bincount(
                simulation_ids,
                weights=event_recovery,
                minlength=stop - start,
            )

    return recoveries


def historical_burning_costs(
    claims: pd.DataFrame,
    policies: pd.DataFrame,
    layers: tuple[XoLLayer, ...],
    target_year: int = 2010,
) -> pd.DataFrame:
    """Calculate annual layer recoveries and rebase exposure to the target year."""

    annual_exposure = policies.groupby("Year")["BCcov"].sum()
    exposure_factor = annual_exposure.loc[target_year] / annual_exposure
    rows: list[dict] = []

    for year, year_claims in claims.groupby("Year"):
        losses = year_claims["Claim"].to_numpy()
        for layer in layers:
            recovery = float(layer_recovery(losses, layer).sum())
            rows.append(
                {
                    "Year": int(year),
                    "Layer": layer.name,
                    "Historical Recovery": recovery,
                    "Exposure Factor to 2010": float(exposure_factor.loc[year]),
                    "2010 Exposure-Adjusted Recovery": float(
                        recovery * exposure_factor.loc[year]
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarise_pricing(
    simulated_recoveries: dict[str, np.ndarray],
    claims: pd.DataFrame,
    expected_annual_claims: float,
    burning_costs: pd.DataFrame,
    layers: tuple[XoLLayer, ...],
    loading: float = 0.20,
) -> pd.DataFrame:
    """Summarise simulated losses and calculate an illustrative technical premium."""

    losses = claims["Claim"].to_numpy()
    rows: list[dict] = []

    for layer in layers:
        values = simulated_recoveries[layer.name]
        expected_recovery = float(values.mean())
        burn = burning_costs.loc[
            burning_costs["Layer"] == layer.name,
            "2010 Exposure-Adjusted Recovery",
        ].mean()

        rows.append(
            {
                "Layer": layer.name,
                "Limit": layer.limit,
                "Attachment": layer.attachment,
                "Historical Claims Attaching": int((losses > layer.attachment).sum()),
                "Expected Attaching Claims per Year": float(
                    expected_annual_claims * np.mean(losses > layer.attachment)
                ),
                "Historical Burning Cost": float(burn),
                "Simulated Expected Recovery": expected_recovery,
                "Simulation Standard Deviation": float(values.std(ddof=1)),
                "75th Percentile": float(np.quantile(values, 0.75)),
                "90th Percentile": float(np.quantile(values, 0.90)),
                "95th Percentile": float(np.quantile(values, 0.95)),
                "99th Percentile": float(np.quantile(values, 0.99)),
                "Technical Premium": expected_recovery * (1.0 + loading),
                "Premium Loading": loading,
            }
        )

    return pd.DataFrame(rows)


def sensitivity_table(
    claims: pd.DataFrame,
    expected_annual_claims: float,
    attachments: tuple[float, ...],
    limits: tuple[float, ...],
    loading: float = 0.20,
) -> pd.DataFrame:
    """Price a grid of alternative attachments and limits."""

    losses = claims["Claim"].to_numpy()
    rows: list[dict] = []
    for attachment in attachments:
        for limit in limits:
            layer = XoLLayer(limit=limit, attachment=attachment)
            expected_recovery = expected_annual_claims * layer_recovery(losses, layer).mean()
            rows.append(
                {
                    "Attachment": attachment,
                    "Limit": limit,
                    "Expected Recovery": expected_recovery,
                    "Technical Premium": expected_recovery * (1.0 + loading),
                }
            )
    return pd.DataFrame(rows)
