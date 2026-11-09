# Final system integration and research synthesis

## Delivered prototype

The repository connects acquisition, quality-controlled M15 bars, features,
Gaussian/GMM HMM fitting, causal replay, versioned probability storage,
interpretation, walk-forward evaluation, visualization and a costed trading
experiment. `scripts/run_pipeline.py` runs consecutive stages and propagates
failures. Its default reuses checked-in artifacts; a full rebuild explicitly
requests phases 1 through 9. See README for dependency and reproduction commands.

## Evidence and uncertainty

Phase 7 reports higher mean predictive log-likelihood for GMM-HMM K3-M3 than
the Gaussian baseline across five retrospective 2023 folds. This describes
density fit, not directional accuracy or trading profitability. Fold overlap,
model-selection exposure and reference-profile overlap limit generalization.

Phase 8 exposes filtered probabilities, Shannon entropy, confidence, occupancy
and gap-aware transitions. Ambiguous probability mass should remain visible.
Markov expected durations, Viterbi durations and filtered argmax run lengths
are distinct quantities.

Phase 9 defines turbulence from train-only dispersion and selects threshold
0.25 on validation. On 4,168 test bars, baseline trades 3,988 times and loses
USD 3,916.60; the filtered strategy trades 3,140 times and loses USD 3,270.90.
Both start with USD 10,000 and trade a fixed 10,000 EUR. Close-equity maximum
drawdown is 39.27% and 32.81%, respectively. Cash has zero profit and trading
drawdown. Reduced loss accompanies reduced exposure, not demonstrated profitable
trading value. Cost scenarios at 2x and 3x are retained in the Phase 9 JSON.

## Reproducibility and verification

Phase 9 JSON records input SHA-256 hashes, model identity, chronological split
boundaries, validation scores and cost sensitivity. The test-equity Parquet
preserves every bar, including no-exposure periods. New tests cover next-open
execution, lagged probabilities, gaps, partial bars, costs, continuous drawdown,
future-price independence and invalid input. Existing tests cover model fitting,
inference persistence, calendar handling, interpretation and walk-forward logic.
Final verification runs the entire test suite and Phase 9 on existing artifacts;
it does not refit or independently re-audit all historical experiments.

## Remaining research

- Untouched multi-year holdout and independent timezone verification.
- Historical broker bid/ask records, variable execution costs and latency.
- Exposure-matched controls, predeclared alternative strategies and confidence
  intervals for trading metrics.
- Training-window, retraining-frequency and restart-count sensitivity.
- Live connectivity, monitoring and production operations.

Completion means an integrated research prototype with documented experiments,
not resolution of every empirical question or production readiness.
