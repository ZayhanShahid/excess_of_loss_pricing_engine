"""Run the excess-of-loss pricing analysis and save its outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import genpareto

from src.xol_pricing_engine import (
    DEFAULT_LAYERS,
    SeverityModel,
    XoLLayer,
    fit_frequency_model,
    fit_severity_model,
    historical_burning_costs,
    load_data,
    sensitivity_table,
    simulate_annual_claim_counts,
    simulate_annual_recoveries,
    summarise_pricing,
    validate_data,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

SEED = 42
SIMULATIONS = 10_000
BATCH_SIZE = 500
TAIL_THRESHOLD_QUANTILE = 0.95
EXPENSE_LOADING = 0.05
VOLATILITY_FACTOR = 0.25

SENSITIVITY_ATTACHMENTS = (50_000, 100_000, 250_000, 500_000)
SENSITIVITY_LIMITS = (100_000, 250_000, 500_000, 1_000_000)


def layer_palette(layers: tuple[XoLLayer, ...]) -> dict[str, tuple[float, float, float]]:
    """Return one colour per layer without assuming a fixed layer count."""

    colours = sns.color_palette("colorblind", n_colors=len(layers))
    return {layer.name: colour for layer, colour in zip(layers, colours, strict=True)}


def save_figure(figure: plt.Figure, filename: str) -> None:
    """Apply consistent spacing, save a figure and close it."""

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_claim_severity(claims: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    severities = claims["Claim"].to_numpy()
    log_bins = np.geomspace(severities.min(), severities.max(), 60)
    sns.histplot(severities, bins=log_bins, color="#24557a", ax=axis)
    axis.set_xscale("log")
    axis.set_title("Historical claim severity distribution")
    axis.set_xlabel("Claim amount (log scale)")
    axis.set_ylabel("Number of claims")
    save_figure(figure, "claim_severity_distribution.png")


def plot_historical_recovery(
    burning_costs: pd.DataFrame,
    layers: tuple[XoLLayer, ...],
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=burning_costs,
        x="Year",
        y="2010 Exposure-Adjusted Recovery",
        hue="Layer",
        marker="o",
        palette=layer_palette(layers),
        ax=axis,
    )
    axis.set_title("Historical exposure-adjusted layer recovery")
    axis.set_ylabel("Annual recovery ($)")
    years = sorted(burning_costs["Year"].unique())
    axis.set_xticks(years, labels=[str(int(year)) for year in years])
    axis.ticklabel_format(style="plain", axis="y")
    save_figure(figure, "historical_recovery_by_year.png")


def plot_simulated_recovery(
    simulated_recoveries: dict[str, np.ndarray],
    layers: tuple[XoLLayer, ...],
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    colours = layer_palette(layers)
    for layer in layers:
        sns.kdeplot(
            simulated_recoveries[layer.name],
            label=layer.name,
            color=colours[layer.name],
            fill=False,
            ax=axis,
        )
    axis.set_title("Simulated annual recovery distribution")
    axis.set_xlabel("Annual recovery ($)")
    axis.set_ylabel("Density")
    axis.ticklabel_format(style="plain", axis="x")
    axis.legend(title="Layer")
    save_figure(figure, "simulated_recovery_distribution.png")


def plot_pricing_comparison(
    pricing: pd.DataFrame,
    layers: tuple[XoLLayer, ...],
) -> None:
    long = pricing.melt(
        id_vars="Layer",
        value_vars=[
            "Historical Burning Cost",
            "Simulated Expected Recovery",
            "Technical Premium",
        ],
        var_name="Measure",
        value_name="Amount",
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=long,
        x="Layer",
        y="Amount",
        hue="Measure",
        order=[layer.name for layer in layers],
        palette="Blues_d",
        ax=axis,
    )
    axis.set_title("Burning cost, expected recovery and technical premium")
    axis.set_xlabel("")
    axis.set_ylabel("Amount ($)")
    axis.ticklabel_format(style="plain", axis="y")
    save_figure(figure, "layer_pricing_comparison.png")


def plot_sensitivity(sensitivity: pd.DataFrame) -> None:
    matrix = sensitivity.pivot(
        index="Attachment",
        columns="Limit",
        values="Technical Premium",
    ).sort_index(ascending=False)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(
        matrix / 1_000_000,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Technical premium ($m)"},
        ax=axis,
    )
    axis.set_title("Technical premium sensitivity")
    axis.set_xlabel("Layer limit ($)")
    axis.set_ylabel("Attachment ($)")
    save_figure(figure, "premium_sensitivity_heatmap.png")


def plot_deductible_diagnostic(claims: pd.DataFrame) -> None:
    below = int((claims["Claim"] < claims["Deduct"]).sum())
    at_or_above = int(len(claims) - below)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        x=["Below listed deductible", "At or above deductible"],
        y=[below, at_or_above],
        hue=["Below listed deductible", "At or above deductible"],
        palette=["#d4934c", "#24557a"],
        legend=False,
        ax=axis,
    )
    axis.set_title("Claim amount compared with listed deductible")
    axis.set_xlabel("")
    axis.set_ylabel("Number of records")
    save_figure(figure, "deductible_diagnostic.png")


def plot_tail_diagnostic(model: SeverityModel) -> None:
    excesses = np.sort(model.tail_excesses)
    empirical_survival = (len(excesses) - np.arange(len(excesses))) / (
        len(excesses) + 1
    )
    fitted_excesses = np.geomspace(max(float(excesses.min()), 1.0), excesses.max(), 250)
    fitted_survival = genpareto.sf(
        fitted_excesses,
        model.shape,
        loc=0,
        scale=model.scale,
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        model.threshold + excesses,
        empirical_survival,
        s=15,
        alpha=0.65,
        color="#24557a",
        label="Historical exceedances",
    )
    axis.plot(
        model.threshold + fitted_excesses,
        fitted_survival,
        color="#d4934c",
        linewidth=2,
        label="Fitted GPD tail",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(
        f"Severity tail fit above ${model.threshold:,.0f} "
        f"({model.exceedance_count} exceedances)"
    )
    axis.set_xlabel("Claim amount ($, log scale)")
    axis.set_ylabel("Conditional survival probability (log scale)")
    axis.legend()
    save_figure(figure, "severity_tail_diagnostic.png")


def save_figures(
    claims: pd.DataFrame,
    burning_costs: pd.DataFrame,
    simulated_recoveries: dict[str, np.ndarray],
    pricing: pd.DataFrame,
    sensitivity: pd.DataFrame,
    severity_model: SeverityModel,
) -> None:
    """Create the project figures using one focused function per chart."""

    plot_claim_severity(claims)
    plot_historical_recovery(burning_costs, DEFAULT_LAYERS)
    plot_simulated_recovery(simulated_recoveries, DEFAULT_LAYERS)
    plot_pricing_comparison(pricing, DEFAULT_LAYERS)
    plot_sensitivity(sensitivity)
    plot_deductible_diagnostic(claims)
    plot_tail_diagnostic(severity_model)


def main() -> None:
    """Fit the models, run the simulations and write reproducible outputs."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    claims, policies = load_data(DATA_DIR)
    validate_data(claims, policies)
    frequency_model = fit_frequency_model(policies)
    severity_model = fit_severity_model(
        claims["Claim"].to_numpy(),
        threshold_quantile=TAIL_THRESHOLD_QUANTILE,
    )

    sensitivity_layers = tuple(
        XoLLayer(limit=limit, attachment=attachment)
        for attachment in SENSITIVITY_ATTACHMENTS
        for limit in SENSITIVITY_LIMITS
    )
    all_layers = tuple(dict.fromkeys((*DEFAULT_LAYERS, *sensitivity_layers)))

    rng = np.random.default_rng(SEED)
    claim_counts = simulate_annual_claim_counts(
        frequency_model,
        SIMULATIONS,
        rng,
        batch_size=BATCH_SIZE,
    )
    all_recoveries = simulate_annual_recoveries(
        claim_counts,
        severity_model,
        all_layers,
        rng,
        batch_size=BATCH_SIZE,
    )
    selected_recoveries = {
        layer.name: all_recoveries[layer.name] for layer in DEFAULT_LAYERS
    }

    burning_costs = historical_burning_costs(claims, policies, DEFAULT_LAYERS)
    pricing = summarise_pricing(
        selected_recoveries,
        claims,
        frequency_model["expected_annual_claims"],
        severity_model,
        burning_costs,
        DEFAULT_LAYERS,
        expense_loading=EXPENSE_LOADING,
        volatility_factor=VOLATILITY_FACTOR,
    )
    sensitivity = sensitivity_table(
        all_recoveries,
        sensitivity_layers,
        expense_loading=EXPENSE_LOADING,
        volatility_factor=VOLATILITY_FACTOR,
    )

    data_summary = pd.DataFrame(
        {
            "Metric": [
                "Claim records",
                "Policy-year records",
                "Mean claim",
                "Median claim",
                "Maximum claim",
                "Zero-claim policy-years",
                "Expected annual claims (2010 portfolio)",
                "Frequency dispersion",
                "Severity tail threshold",
                "Severity tail exceedances",
                "GPD shape",
                "GPD scale",
                "Simulation seed",
                "Simulation batch size",
            ],
            "Value": [
                len(claims),
                len(policies),
                claims["Claim"].mean(),
                claims["Claim"].median(),
                claims["Claim"].max(),
                int((policies["Freq"] == 0).sum()),
                frequency_model["expected_annual_claims"],
                frequency_model["dispersion"],
                severity_model.threshold,
                severity_model.exceedance_count,
                severity_model.shape,
                severity_model.scale,
                SEED,
                BATCH_SIZE,
            ],
        }
    )
    tail_summary = pd.DataFrame(
        [
            {
                "Threshold Quantile": severity_model.threshold_quantile,
                "Threshold": severity_model.threshold,
                "Tail Probability": severity_model.tail_probability,
                "Exceedance Count": severity_model.exceedance_count,
                "GPD Shape": severity_model.shape,
                "GPD Scale": severity_model.scale,
                "Historical Maximum": severity_model.historical_maximum,
            }
        ]
    )

    burning_costs.to_csv(TABLE_DIR / "annual_burning_costs.csv", index=False)
    data_summary.to_csv(TABLE_DIR / "data_summary.csv", index=False)
    pricing.to_csv(TABLE_DIR / "layer_pricing_results.csv", index=False)
    sensitivity.to_csv(TABLE_DIR / "premium_sensitivity.csv", index=False)
    tail_summary.to_csv(TABLE_DIR / "tail_model_summary.csv", index=False)

    save_figures(
        claims,
        burning_costs,
        selected_recoveries,
        pricing,
        sensitivity,
        severity_model,
    )

    print("Analysis complete.")
    print(pricing[["Layer", "Simulated Expected Recovery", "Technical Premium"]])


if __name__ == "__main__":
    main()
