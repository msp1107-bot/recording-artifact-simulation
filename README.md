# Recording-artifact simulation

Simulation code accompanying the article *"When the record makes the result:
exposure-linked recording of a continuous outcome can manufacture a spurious
treatment effect in real-world evidence"* (Mao & Chen).

## What this shows

Starting from a **fully known, null data-generating process** — two treatment
arms drawn from the *same* distribution, so the true treatment effect is exactly
zero — the script applies a plausible **differential recording** rule (routine
cases in the treated arm are often written down as a pinned low template value,
e.g. 30 or 50 mL) and then analyses the *recorded* field exactly as a routine
study would.

Under this known null it reproduces the entire empirical signature:

- **(A)** a large apparent multiplicative "reduction" in a naive log-linear model;
- **(B)** instability of the effect size across standard estimators (median-ratio,
  log-linear, Gamma-log, arithmetic mean);
- **(C)** a between-arm exceedance gap that is large at low thresholds, collapses
  at the clinically meaningful threshold (500 mL), and can reverse at the tail;
- **(D)** a per-arm digit-preference (heaping) signature.

Because the generating truth is zero, **every part of the apparent effect is
manufactured by the recording process.** This is a demonstration of *sufficiency*
— differential recording alone can produce the whole pattern — not a claim that
any particular real dataset arose this way.

## Requirements

```
pip install -r requirements.txt
```
(numpy, pandas, statsmodels, matplotlib)

## Run

```
python3 artifact_simulation.py
```

This prints a summary table to stdout and writes `figure3_simulation.png`.
It also runs 200 replications to show the pattern is a regularity, not a single
lucky draw.

## Expected output (representative)

- Naive log-linear apparent "reduction": ~58% (true effect: 0%)
- Estimator spread: ~37%–75% from one null dataset
- Exceedance gap: large at 100 mL, ~0 at 500 mL, ~0 (noisy) at 1000 mL
- Digit preference at 30/50 mL: treated ~66% vs control ~7%
- 200-replication median naive reduction ~58% (IQR ~57–59%); median 500 mL gap ~0 pp

## Reproducibility / seed

The main run uses a fixed seed (`SEED` in the script) and is deterministic on a
given NumPy version. Minor version differences in NumPy's random `Generator` can
shift the last digit; the **qualitative** pattern is stable across seeds, as the
200-replication summary confirms. All parameters are exposed at the top of the
script and can be changed — for example, set `TRUE_EFFECT_RATIO = 0.90` to inject
a genuine 10% effect and confirm the diagnostics behave differently.

## Files

- `artifact_simulation.py` — the simulation and figure
- `requirements.txt` — Python dependencies
- `CITATION.cff` — citation metadata (add the Zenodo DOI after archiving)
- `LICENSE` — MIT

## How to obtain a DOI

1. Push this folder to a public GitHub repository.
2. In Zenodo, enable the repository and create a release (or upload this folder
   directly). Zenodo will mint a DOI.
3. Put that DOI in `CITATION.cff` and in the manuscript's Data Availability
   statement (replacing "will be deposited in a public repository upon
   acceptance").

## License

MIT (see `LICENSE`).
