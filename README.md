# Excess-of-Loss Reinsurance Pricing Engine

This project prices per-claim excess-of-loss (XoL) reinsurance using five years of historical property claims and policy exposure data. It combines historical burning-cost analysis with 10,000 Monte Carlo simulations to compare alternative attachment points and limits.

The aim was to answer a practical question: how does changing an XoL layer alter the insurer's retention, the reinsurer's expected recovery and the resulting technical premium?

## Key results

| Layer | Historical burning cost | Simulated expected recovery | 95th percentile | Technical premium |
|---|---:|---:|---:|---:|
| 100k xs 50k | $2.732m | $2.757m | $4.042m | $3.308m |
| 250k xs 100k | $3.058m | $3.072m | $4.861m | $3.686m |
| 500k xs 250k | $2.874m | $2.850m | $5.063m | $3.420m |

The technical premium is illustrative and applies a 20% loading to the simulated expected recovery. This is intended to cover expenses, uncertainty and profit; it is an explicit project assumption rather than a quoted market rate.

![Layer pricing comparison](outputs/figures/layer_pricing_comparison.png)

## Data

The source is the Wisconsin Local Government Property Insurance Fund case study published in the open-access [Loss Data Analytics](https://openacttexts.github.io/Loss-Data-Analytics/ChapIntro.html) textbook.

- 6,258 claims from 2006–2010
- 5,639 policy-years
- Mean claim: $15,585.90
- Median claim: $1,837
- Maximum claim: $12,922,217.84
- 70.2% of policy-years had no claims
- Claim-frequency mean: 1.109; sample variance: 73.088

The large gap between the mean and median claim and the maximum loss of almost $13m show why XoL pricing is driven by the right tail rather than the typical claim.

## Method

### 1. Reinsurance recovery

For a claim amount `X`, attachment `A` and layer limit `L`, the recovery is:

```text
recovery = min(max(X - A, 0), L)
```

For example, a $180,000 claim under a 100k xs 50k layer produces a $100,000 recovery. The insurer retains the first $50,000 and the portion above $150,000.

### 2. Historical burning cost

I calculated the actual recovery for each claim and layer, aggregated it by year and rebased each year to the 2010 portfolio exposure using total building and contents coverage (`BCcov`). The average exposure-adjusted annual recovery is the historical burning cost.

### 3. Frequency model

Claim frequency is heavily overdispersed: its variance is much larger than its mean. I therefore used a Negative Binomial model rather than a Poisson model. Expected claim counts are proportional to `BCcov`, with the dispersion parameter estimated by the method of moments across all policy-years.

This gives an expected 1,362 claims for the 2010-base portfolio.

### 4. Severity and simulation

Each simulated year follows two steps:

1. Simulate policy-level claim counts from the fitted Negative Binomial model.
2. Bootstrap individual claim amounts from the 6,258 historical positive severities and apply each XoL layer.

The same simulated claims are used across the three layers so the comparison is consistent. A fixed random seed makes the results reproducible.

### 5. Technical premium

For this project:

```text
technical premium = simulated expected recovery × 1.20
```

I also tested 16 attachment-and-limit combinations to show how the price responds as the reinsurer takes more or less of the tail.

![Technical premium sensitivity](outputs/figures/premium_sensitivity_heatmap.png)

## Findings

- The 250k xs 100k layer has the largest expected recovery of the three selected structures because its wider limit captures more loss despite attaching higher.
- The 500k xs 250k layer is the most volatile: it has fewer expected attaching claims but the highest standard deviation and 95th-percentile recovery.
- Historical burning costs and simulated expected recoveries differ by less than 1% for all three layers, providing a useful reasonableness check.
- Increasing the attachment reduces the technical premium, while increasing the layer limit increases it. The sensitivity grid ranges from approximately $0.58m to $9.28m.
- 2,810 claims (44.9%) are below their listed deductible. I retain them as recorded ground-up/informational losses and document this rather than treating them as a small data error.

## Project structure

```text
├── data/
│   └── README.md
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   └── xol_pricing_engine.py
├── tests/
│   └── test_xol_pricing_engine.py
├── run_analysis.py
├── requirements.txt
└── README.md
```

## Running the project

1. Download the two source CSVs described in `data/README.md` and place them in `data/`.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the analysis:

```bash
python run_analysis.py
```

4. Run the calculation checks:

```bash
python -m unittest discover tests
```

## Limitations

- The dataset contains only five years, so a few large losses have a strong influence on the price.
- Empirical severity bootstrapping does not generate losses above the historical maximum.
- Frequency and severity are modelled independently.
- No explicit claims inflation, reinstatements, aggregate limits or brokerage adjustments are included.
- The 20% premium loading is illustrative and would need to be replaced with commercial pricing assumptions in practice.

The engine is an educational pricing model, not a production quotation tool.
