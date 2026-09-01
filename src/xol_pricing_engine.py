"""Core calculations for the excess-of-loss reinsurance pricing project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import genpareto


@dataclass(frozen=True)
class XoLLayer:
    """A per-claim excess-of-loss layer written as limit xs attachment."""

    limit: float
    attachment: float

    @property
    def name(self) -> str:
        return f"{self.limit / 1_000:,.0f}k xs {self.attachment / 1_000:,.0f}k"


@dataclass(frozen=True)
class SeverityModel:
    """Empirical body with a Generalized Pareto tail above a threshold."""

    threshold: float
    threshold_quantile: float
    tail_probability: float
    shape: float
    scale: float
    body_values: np.ndarray
    tail_excesses: np.ndarray
    historical_maximum: float

    @property
    def exceedance_count(self) -> int:
        return int(len(self.tail_excesses))


DEFAULT_LAYERS = (
    XoLLayer(limit=100_000, attachment=50_000),
    XoLLayer(limit=250_000, attachment=100_000),
    XoLLayer(limit=500_000, attachment=250_000),
)


def _batch_slices(total: int, batch_size: int) -> Iterator[tuple[int, int]]:
    """Yield start/stop positions for memory-controlled simulation batches."""

    if total < 0:
        raise ValueError("The total number of simulations cannot be negative.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two source CSVs and report missing files clearly."""

    data_dir = Path(data_dir)
    claim_path = data_dir / "CLAIMLEVEL.csv"
    policy_path = data_dir / "PropertyFundInsample.csv"
    missing = [path.name for path in (claim_path, policy_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required data file(s) in {data_dir}: {', '.join(missing)}. "
            "See data/README.md for download instructions."
        )

    try:
        claims = pd.read_csv(claim_path)
        policies = pd.read_csv(policy_path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"A source CSV could not be parsed: {exc}") from exc
    return claims, policies


def validate_data(claims: pd.DataFrame, policies: pd.DataFrame) -> None:
    """Raise a clear error if the inputs do not match the expected structure."""

    claim_columns = {"PolicyNum", "Year", "Claim", "Deduct"}
    policy_columns = {"PolicyNum", "Year", "BCcov", "Freq"}
    missing_claims = claim_columns.difference(claims.columns)
    missing_policies = policy_columns.difference(policies.columns)

    if missing_claims or missing_policies:
        raise ValueError(
            f"Missing claim columns: {sorted(missing_claims)}; "
            f"missing policy columns: {sorted(missing_policies)}"
        )
    if claims.empty or policies.empty:
        raise ValueError("The claim and policy datasets must both contain records.")
    if claims[list(claim_columns)].isna().any().any():
        raise ValueError("Required claim fields cannot contain missing values.")
    if policies[list(policy_columns)].isna().any().any():
        raise ValueError("Required policy fields cannot contain missing values.")
    if (claims["Claim"] <= 0).any():
        raise ValueError("Claim amounts must be positive.")
    if (policies["BCcov"] <= 0).any() or (policies["Freq"] < 0).any():
        raise ValueError("Coverage must be positive and frequency cannot be negative.")

    claim_years = set(claims["Year"].unique())
    policy_years = set(policies["Year"].unique())
    missing_exposure_years = sorted(claim_years.difference(policy_years))
    if missing_exposure_years:
        raise ValueError(
            "Claims contain year(s) with no matching policy exposure: "
            f"{missing_exposure_years}"
        )


def layer_recovery(losses: np.ndarray, layer: XoLLayer) -> np.ndarray:
    """Calculate the reinsurer's recovery for each individual claim."""

    return np.minimum(np.maximum(losses - layer.attachment, 0.0), layer.limit)


def fit_frequency_model(policies: pd.DataFrame, target_year: int = 2010) -> dict:
    """Fit an exposure-adjusted Negative Binomial frequency model.

    Expected claims are proportional to building and contents coverage (BCcov).
    The Negative Binomial dispersion is estimated by the method of moments.
    """

    if target_year not in set(policies["Year"]):
        raise ValueError(f"Target year {target_year} is absent from the policy data.")

    claim_rate = policies["Freq"].sum() / policies["BCcov"].sum()
    fitted_mean = claim_rate * policies["BCcov"]
    denominator = float((fitted_mean**2).sum())
    if denominator == 0:
        raise ValueError("Frequency model cannot be fitted when all claim counts are zero.")

    numerator = ((policies["Freq"] - fitted_mean) ** 2 - fitted_mean).sum()
    dispersion = max(float(numerator / denominator), 0.0)
    target = policies.loc[policies["Year"] == target_year]
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


def fit_severity_model(
    severities: np.ndarray,
    threshold_quantile: float = 0.95,
    minimum_exceedances: int = 50,
) -> SeverityModel:
    """Fit an empirical-body/GPD-tail severity model."""

    values = np.asarray(severities, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Severities must be a non-empty, finite one-dimensional array.")
    if (values <= 0).any():
        raise ValueError("Severities must be positive.")
    if not 0 < threshold_quantile < 1:
        raise ValueError("Tail threshold quantile must lie between zero and one.")

    threshold = float(np.quantile(values, threshold_quantile))
    body_values = values[values <= threshold]
    tail_excesses = values[values > threshold] - threshold
    if len(tail_excesses) < minimum_exceedances:
        raise ValueError(
            f"Only {len(tail_excesses)} losses exceed the selected threshold; "
            f"at least {minimum_exceedances} are required."
        )

    shape, _, scale = genpareto.fit(tail_excesses, floc=0)
    if scale <= 0 or not np.isfinite([shape, scale]).all():
        raise ValueError("The fitted GPD tail parameters are not valid.")

    return SeverityModel(
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        tail_probability=float(len(tail_excesses) / len(values)),
        shape=float(shape),
        scale=float(scale),
        body_values=body_values,
        tail_excesses=tail_excesses,
        historical_maximum=float(values.max()),
    )


def sample_severities(
    model: SeverityModel,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample losses from the empirical body and fitted GPD tail."""

    if size < 0:
        raise ValueError("Sample size cannot be negative.")
    tail_mask = rng.random(size) < model.tail_probability
    simulated = np.empty(size, dtype=float)
    body_count = int((~tail_mask).sum())
    tail_count = int(tail_mask.sum())

    if body_count:
        simulated[~tail_mask] = rng.choice(model.body_values, size=body_count, replace=True)
    if tail_count:
        simulated[tail_mask] = model.threshold + genpareto.rvs(
            model.shape,
            loc=0,
            scale=model.scale,
            size=tail_count,
            random_state=rng,
        )
    return simulated


def severity_exceedance_probability(model: SeverityModel, amount: float) -> float:
    """Return the modelled probability that an individual loss exceeds an amount."""

    if amount <= model.threshold:
        body_probability = float(np.mean(model.body_values > amount))
        return (1.0 - model.tail_probability) * body_probability + model.tail_probability
    return float(
        model.tail_probability
        * genpareto.sf(amount - model.threshold, model.shape, loc=0, scale=model.scale)
    )


def simulate_annual_claim_counts(
    frequency_model: dict,
    simulations: int,
    rng: np.random.Generator,
    batch_size: int = 500,
) -> np.ndarray:
    """Simulate total annual claim counts for the target portfolio."""

    means = np.asarray(frequency_model["target_means"], dtype=float)
    dispersion = float(frequency_model["dispersion"])
    totals = np.empty(simulations, dtype=np.int64)

    for start, stop in _batch_slices(simulations, batch_size):
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
    severity_model: SeverityModel,
    layers: tuple[XoLLayer, ...],
    rng: np.random.Generator,
    batch_size: int = 500,
) -> dict[str, np.ndarray]:
    """Simulate severities and aggregate per-claim recoveries by year.

    Exact random draws are reproducible for a fixed seed, batch size and code
    version. A different batch size can produce statistically equivalent but
    not necessarily identical draws because it changes the RNG call order.
    """

    simulations = len(claim_counts)
    recoveries = {layer.name: np.zeros(simulations) for layer in layers}

    for start, stop in _batch_slices(simulations, batch_size):
        counts = claim_counts[start:stop]
        event_count = int(counts.sum())
        sampled_losses = sample_severities(severity_model, event_count, rng)
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
    if target_year not in annual_exposure.index:
        raise ValueError(f"Target year {target_year} has no policy exposure.")
    missing_years = sorted(set(claims["Year"]).difference(annual_exposure.index))
    if missing_years:
        raise ValueError(
            "Cannot calculate burning costs because policy exposure is missing for "
            f"claim year(s): {missing_years}"
        )

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


def calculate_technical_premium(
    expected_recovery: float,
    standard_deviation: float,
    expense_loading: float = 0.05,
    volatility_factor: float = 0.25,
) -> float:
    """Apply an expense load and a standard-deviation risk load."""

    if min(expected_recovery, standard_deviation, expense_loading, volatility_factor) < 0:
        raise ValueError("Premium inputs and loading parameters cannot be negative.")
    return (
        expected_recovery * (1.0 + expense_loading)
        + volatility_factor * standard_deviation
    )


def summarise_pricing(
    simulated_recoveries: dict[str, np.ndarray],
    claims: pd.DataFrame,
    expected_annual_claims: float,
    severity_model: SeverityModel,
    burning_costs: pd.DataFrame,
    layers: tuple[XoLLayer, ...],
    expense_loading: float = 0.05,
    volatility_factor: float = 0.25,
) -> pd.DataFrame:
    """Summarise simulated losses and calculate risk-sensitive premiums."""

    historical_losses = claims["Claim"].to_numpy()
    rows: list[dict] = []
    for layer in layers:
        values = simulated_recoveries[layer.name]
        expected_recovery = float(values.mean())
        standard_deviation = float(values.std(ddof=1))
        burn = burning_costs.loc[
            burning_costs["Layer"] == layer.name,
            "2010 Exposure-Adjusted Recovery",
        ].mean()
        premium = calculate_technical_premium(
            expected_recovery,
            standard_deviation,
            expense_loading,
            volatility_factor,
        )

        rows.append(
            {
                "Layer": layer.name,
                "Limit": layer.limit,
                "Attachment": layer.attachment,
                "Historical Claims Attaching": int(
                    (historical_losses > layer.attachment).sum()
                ),
                "Modelled Attaching Claims per Year": float(
                    expected_annual_claims
                    * severity_exceedance_probability(severity_model, layer.attachment)
                ),
                "Historical Burning Cost": float(burn),
                "Simulated Expected Recovery": expected_recovery,
                "Simulation Standard Deviation": standard_deviation,
                "Coefficient of Variation": standard_deviation / expected_recovery,
                "75th Percentile": float(np.quantile(values, 0.75)),
                "90th Percentile": float(np.quantile(values, 0.90)),
                "95th Percentile": float(np.quantile(values, 0.95)),
                "99th Percentile": float(np.quantile(values, 0.99)),
                "Expense Loading": expense_loading,
                "Volatility Factor": volatility_factor,
                "Technical Premium": premium,
                "Effective Premium Loading": premium / expected_recovery - 1.0,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_table(
    simulated_recoveries: dict[str, np.ndarray],
    layers: tuple[XoLLayer, ...],
    expense_loading: float = 0.05,
    volatility_factor: float = 0.25,
) -> pd.DataFrame:
    """Summarise a simulated grid of alternative attachments and limits."""

    rows: list[dict] = []
    for layer in layers:
        values = simulated_recoveries[layer.name]
        expected_recovery = float(values.mean())
        standard_deviation = float(values.std(ddof=1))
        rows.append(
            {
                "Attachment": layer.attachment,
                "Limit": layer.limit,
                "Expected Recovery": expected_recovery,
                "Standard Deviation": standard_deviation,
                "Coefficient of Variation": standard_deviation / expected_recovery,
                "Technical Premium": calculate_technical_premium(
                    expected_recovery,
                    standard_deviation,
                    expense_loading,
                    volatility_factor,
                ),
            }
        )
    return pd.DataFrame(rows)
