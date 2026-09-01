"""Run the full excess-of-loss pricing analysis and save tables and figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.xol_pricing_engine import (
    DEFAULT_LAYERS,
    fit_frequency_model,
    historical_burning_costs,
    layer_recovery,
    load_data,
    sensitivity_table,
    simulate_annual_claim_counts,
    simulate_annual_recoveries,
    summarise_pricing,
    validate_data,
)


SEED = 42
SIMULATIONS = 10_000
PREMIUM_LOADING = 0.20


def currency_axis(axis) -> None:
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"${value / 1_000_000:.1f}m")
    )


def save_figures(
    claims: pd.DataFrame,
    burning_costs: pd.DataFrame,
    pricing: pd.DataFrame,
    simulated_recoveries: dict[str, np.ndarray],
    sensitivity: pd.DataFrame,
    figure_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    colours = ["#24557a", "#4f8f8b", "#d4934c"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(np.log10(claims["Claim"]), bins=45, color=colours[0], edgecolor="white")
    for layer, colour in zip(DEFAULT_LAYERS, colours):
        ax.axvline(np.log10(layer.attachment), color=colour, linestyle="--", linewidth=2,
                   label=f"{layer.name} attachment")
    ticks = np.arange(2, 8)
    ax.set_xticks(ticks, [f"${10**tick:,.0f}" for tick in ticks])
    ax.set_title("Historical claim severity and selected attachment points")
    ax.set_xlabel("Claim amount (log scale)")
    ax.set_ylabel("Number of claims")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figure_dir / "claim_severity_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=burning_costs,
        x="Year",
        y="2010 Exposure-Adjusted Recovery",
        hue="Layer",
        palette=colours,
        ax=ax,
    )
    currency_axis(ax)
    ax.set_title("Exposure-adjusted historical recovery by year")
    ax.set_xlabel("")
    ax.set_ylabel("Reinsurance recovery")
    ax.legend(title="Layer")
    fig.tight_layout()
    fig.savefig(figure_dir / "historical_recovery_by_year.png", dpi=180)
    plt.close(fig)

    long_simulation = pd.DataFrame(simulated_recoveries).melt(
        var_name="Layer", value_name="Annual Recovery"
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(
        data=long_simulation,
        x="Layer",
        y="Annual Recovery",
        hue="Layer",
        palette=colours,
        showfliers=False,
        legend=False,
        ax=ax,
    )
    currency_axis(ax)
    ax.set_title(f"Simulated annual recoveries ({SIMULATIONS:,} simulations)")
    ax.set_xlabel("")
    ax.set_ylabel("Annual recovery")
    fig.tight_layout()
    fig.savefig(figure_dir / "simulated_recovery_distribution.png", dpi=180)
    plt.close(fig)

    comparison = pricing.melt(
        id_vars="Layer",
        value_vars=[
            "Historical Burning Cost",
            "Simulated Expected Recovery",
            "Technical Premium",
        ],
        var_name="Measure",
        value_name="Amount",
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=comparison, x="Layer", y="Amount", hue="Measure", ax=ax)
    currency_axis(ax)
    ax.set_title("Burning cost, expected recovery and technical premium")
    ax.set_xlabel("")
    ax.set_ylabel("Annual amount")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(figure_dir / "layer_pricing_comparison.png", dpi=180)
    plt.close(fig)

    pivot = sensitivity.pivot(
        index="Attachment", columns="Limit", values="Technical Premium"
    ) / 1_000_000
    pivot.index = [f"${value / 1_000:,.0f}k" for value in pivot.index]
    pivot.columns = [f"${value / 1_000:,.0f}k" for value in pivot.columns]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "$m"}, ax=ax)
    ax.set_title("Technical premium sensitivity ($m)")
    ax.set_xlabel("Layer limit")
    ax.set_ylabel("Attachment")
    fig.tight_layout()
    fig.savefig(figure_dir / "premium_sensitivity_heatmap.png", dpi=180)
    plt.close(fig)

    deductible_status = np.select(
        [claims["Claim"] < claims["Deduct"], claims["Claim"] == claims["Deduct"]],
        ["Below deductible", "Equal to deductible"],
        default="Above deductible",
    )
    deductible_counts = pd.Series(deductible_status).value_counts().reindex(
        ["Below deductible", "Equal to deductible", "Above deductible"]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(deductible_counts.index, deductible_counts.values, color=colours)
    ax.bar_label(bars, labels=[f"{v:,}" for v in deductible_counts.values], padding=3)
    ax.set_title("Recorded claims compared with policy deductibles")
    ax.set_ylabel("Number of claims")
    ax.set_ylim(0, deductible_counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(figure_dir / "deductible_diagnostic.png", dpi=180)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    table_dir = project_dir / "outputs" / "tables"
    figure_dir = project_dir / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    claims, policies = load_data(data_dir)
    validate_data(claims, policies)

    frequency_model = fit_frequency_model(policies)
    rng = np.random.default_rng(SEED)
    claim_counts = simulate_annual_claim_counts(frequency_model, SIMULATIONS, rng)
    simulated_recoveries = simulate_annual_recoveries(
        claim_counts,
        claims["Claim"].to_numpy(),
        DEFAULT_LAYERS,
        rng,
    )

    burning_costs = historical_burning_costs(claims, policies, DEFAULT_LAYERS)
    pricing = summarise_pricing(
        simulated_recoveries,
        claims,
        frequency_model["expected_annual_claims"],
        burning_costs,
        DEFAULT_LAYERS,
        PREMIUM_LOADING,
    )
    sensitivity = sensitivity_table(
        claims,
        frequency_model["expected_annual_claims"],
        attachments=(50_000, 100_000, 250_000, 500_000),
        limits=(100_000, 250_000, 500_000, 1_000_000),
        loading=PREMIUM_LOADING,
    )

    data_summary = pd.DataFrame(
        {
            "Metric": [
                "Claims",
                "Policy-years",
                "Mean claim",
                "Median claim",
                "Maximum claim",
                "Mean policy-year frequency",
                "Sample variance of frequency",
                "Zero-claim policy-years",
                "Claims below listed deductible",
                "Expected 2010-base annual claims",
                "Frequency dispersion",
                "Monte Carlo simulations",
            ],
            "Value": [
                len(claims),
                len(policies),
                claims["Claim"].mean(),
                claims["Claim"].median(),
                claims["Claim"].max(),
                policies["Freq"].mean(),
                policies["Freq"].var(ddof=1),
                (policies["Freq"] == 0).mean(),
                (claims["Claim"] < claims["Deduct"]).mean(),
                frequency_model["expected_annual_claims"],
                frequency_model["dispersion"],
                SIMULATIONS,
            ],
        }
    )

    pricing.to_csv(table_dir / "layer_pricing_results.csv", index=False)
    burning_costs.to_csv(table_dir / "annual_burning_costs.csv", index=False)
    sensitivity.to_csv(table_dir / "premium_sensitivity.csv", index=False)
    data_summary.to_csv(table_dir / "data_summary.csv", index=False)
    pd.DataFrame({"Simulated Claim Count": claim_counts, **simulated_recoveries}).to_csv(
        table_dir / "simulation_results.csv", index=False
    )

    save_figures(
        claims,
        burning_costs,
        pricing,
        simulated_recoveries,
        sensitivity,
        figure_dir,
    )

    print(f"Completed {SIMULATIONS:,} simulations with seed {SEED}.")
    print(pricing[["Layer", "Historical Burning Cost", "Simulated Expected Recovery", "Technical Premium"]].to_string(index=False))


if __name__ == "__main__":
    main()
