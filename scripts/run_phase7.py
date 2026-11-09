"""
scripts.run_phase7
~~~~~~~~~~~~~~~~~~
Phase 7: Out-of-Sample Walk-Forward Validation, Model Architecture Comparison,
Parameter Sensitivity Analysis, and Cross-Regime Robustness Evaluation.

Executes the 30 unique model fits across 5 expanding-window folds:
1. Primary Architecture Comparison (RQ2):
   - Gaussian HMM K3-M1 (cov_reg=1e-4, S=5, [r_oc, log_vol]): 5 folds
   - GMM-HMM K3-M2     (cov_reg=1e-3, S=5, [r_oc, log_vol]): 5 folds
   - GMM-HMM K3-M3     (cov_reg=1e-3, S=5, [r_oc, log_vol]): 5 folds (Anchor Baseline)
2. Covariance Regularization Sensitivity (RQ3):
   - GMM-HMM K3-M3 with cov_reg=1e-4: 5 folds
   - GMM-HMM K3-M3 with cov_reg=1e-2: 5 folds
   - (cov_reg=1e-3 reused from Anchor Baseline)
3. Feature-Representation / Diurnal Sensitivity (RQ3):
   - GMM-HMM K3-M3 with [r_oc, range_zscore] (cov_reg=1e-3, S=5): 5 folds
   - ([r_oc, log_vol] reused from Anchor Baseline)
4. Sequential Evaluation:
   - Policy A (continuous forward propagation via FX calendar) on Val and Test
   - Policy B (independent diagnostic sequence) on Val and Test
5. Statistical Testing:
   - Diebold-Mariano tests with Newey-West HAC (lags 4, 8, 16) and HLN correction
   - Holm-Bonferroni FWER step-down adjustment across pairwise comparisons
6. In-Sample Regime Interpretation:
   - Forward-backward smoothed posterior weighting on Train
   - Hungarian bipartite alignment to canonical reference (GMM_HMM_K3_M3_v1)
   - Volatility ordering monotonicity and duration metrics
7. Publication Figures & JSON Artifact:
   - Figure 19: Out-of-Sample Predictive Likelihood Trajectories
   - Figure 20: Regime Parameter Stability & Alignment
   - Full JSON results to data/reports/phase7_walkforward_results.json
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.gaussian_hmm import GaussianHMM
from src.models.gmm_hmm import GMMHMM
from src.features.candidates import (
    compute_log_returns_oc,
    compute_log_volatility,
    compute_log_range,
)
from src.features.seasonality import DiurnalVolatilityScaler
from src.models.splits import identify_continuous_sequences
from src.inference.calendar import FXCalendarGapHandler
from src.inference.versioning import ModelArtifact
from src.interpretation.smoother import compute_smoothed_posteriors
from src.interpretation.characteristics import calculate_state_characteristics
from src.interpretation.alignment import ReferenceProfileScaler, align_model_states
from src.evaluation.walk_forward import (
    WalkForwardFold,
    generate_walk_forward_folds,
    sequential_forward_filter,
    diebold_mariano_test,
    holm_bonferroni_adjustment,
    fit_model_with_seed_schedule,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase7Runner")


def get_canonical_reference_scaler(
    df_complete: pd.DataFrame,
    ref_artifact_path: str = "models/GMM_HMM_K3_M3_v1.json",
) -> Tuple[Any, Any]:
    """
    Extract canonical reference characteristics and fit ReferenceProfileScaler
    using the audited Phase 6 training partition (first 12,501 bars).
    """
    logger.info("Loading canonical reference artifact from %s...", ref_artifact_path)
    ref_art = ModelArtifact.load_json(ref_artifact_path)
    ref_model = ref_art.to_model_instance()

    # Phase 6 Train slice: 12,501 bars
    df_train_phase6 = df_complete.iloc[:12501].copy()
    X_train_phase6 = df_train_phase6[["r_oc", "log_vol"]].values
    lengths_phase6, _ = identify_continuous_sequences(df_train_phase6, expected_step_minutes=15)

    # Compute smoothed posteriors
    fb = compute_smoothed_posteriors(ref_model, X_train_phase6, lengths_phase6)
    ref_chars = calculate_state_characteristics(ref_model, df_train_phase6, fb.gamma, "GMM_HMM_K3_M3_v1")
    scaler = ReferenceProfileScaler.fit(ref_chars)
    logger.info("Canonical ReferenceProfileScaler fitted successfully.")
    return ref_chars, scaler


def run_phase7_experiments(
    data_path: str = "data/processed/EURUSD_M15_2023.parquet",
    n_restarts: int = 5,
    base_random_state: int = 42,
    output_json_path: str = "data/reports/phase7_walkforward_results.json",
    figures_dir: str = "data/reports/figures",
    max_folds: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete Phase 7 experiment matrix across 5 expanding folds."""
    t_start_total = time.perf_counter()

    logger.info("=" * 80)
    logger.info("STARTING PHASE 7: OUT-OF-SAMPLE WALK-FORWARD & ROBUSTNESS ANALYSIS")
    logger.info("Configuration: S=%d restarts | Base Seed=%d | Folds=%s", n_restarts, base_random_state, "all 5" if max_folds is None else f"first {max_folds}")
    logger.info("=" * 80)

    # 1. Load Data and Engineer Base Features
    df_raw = pd.read_parquet(data_path)
    logger.info("Loaded raw dataset: %d total bars from %s", len(df_raw), data_path)

    df_c = df_raw[df_raw["is_complete"] == True].copy()
    df_c["timestamp"] = pd.to_datetime(df_c["timestamp"], utc=True)
    df_c = df_c.sort_values("timestamp").reset_index(drop=True)

    df_c["r_oc"] = compute_log_returns_oc(df_c)
    df_c["log_vol"] = compute_log_volatility(df_c)
    df_c["log_range"] = compute_log_range(df_c)
    logger.info("Engineered base features: r_oc, log_vol, log_range across %d complete bars", len(df_c))

    # 2. Generate and Verify Walk-Forward Folds
    folds = generate_walk_forward_folds(df_c)
    if max_folds is not None:
        folds = folds[:max_folds]
    gap_handler = FXCalendarGapHandler()

    # 3. Canonical Reference Scaler for Hungarian Label Alignment
    ref_chars, ref_scaler = get_canonical_reference_scaler(df_c)

    # Output structures
    all_fit_provenance: List[Dict[str, Any]] = []
    fold_results: Dict[int, Dict[str, Any]] = {}

    total_fits_executed = 0

    # =========================================================================
    # Loop over 5 Walk-Forward Folds
    # =========================================================================
    for fold in folds:
        fid = fold.fold_id
        logger.info("\n" + "#" * 70)
        logger.info("EXECUTING FOLD %d/5: Train=%d bars | Val=%d bars | Test=%d bars", fid, fold.train_bars, fold.val_bars, fold.test_bars)
        logger.info("Train Window: %s to %s", fold.train_start, fold.train_end)
        logger.info("Val Window:   %s to %s", fold.val_start, fold.val_end)
        logger.info("Test Window:  %s to %s", fold.test_start, fold.test_end)
        logger.info("#" * 70)

        fold_record: Dict[str, Any] = {
            "fold_id": fid,
            "train_start": fold.train_start.isoformat(),
            "train_end": fold.train_end.isoformat(),
            "train_bars": fold.train_bars,
            "val_start": fold.val_start.isoformat(),
            "val_end": fold.val_end.isoformat(),
            "val_bars": fold.val_bars,
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test_end.isoformat(),
            "test_bars": fold.test_bars,
            "models": {},
            "pairwise_comparisons": {},
        }

        # Prepare fold feature matrices (Baseline representation: [r_oc, log_vol])
        base_features = ["r_oc", "log_vol"]
        X_train_base = fold.df_train[base_features].values
        X_val_base = fold.df_val[base_features].values
        X_test_base = fold.df_test[base_features].values

        # Prepare Diurnal-adjusted representation: [r_oc, range_zscore]
        # Fit scaler STRICTLY on fold.df_train
        diurnal_scaler = DiurnalVolatilityScaler(target_col="log_range")
        df_train_diurnal = diurnal_scaler.fit_transform(fold.df_train)
        df_val_diurnal = diurnal_scaler.transform(fold.df_val)
        df_test_diurnal = diurnal_scaler.transform(fold.df_test)

        diurnal_features = ["r_oc", "range_zscore"]
        X_train_diurnal = df_train_diurnal[diurnal_features].values
        X_val_diurnal = df_val_diurnal[diurnal_features].values
        X_test_diurnal = df_test_diurnal[diurnal_features].values

        # ---------------------------------------------------------------------
        # Define the 6 model configurations to evaluate on this fold
        # (3 Primary + 2 Reg sensitivity + 1 Diurnal sensitivity)
        # ---------------------------------------------------------------------
        configs_to_fit = [
            # Primary Architectures (RQ2)
            {
                "key": "Gaussian_HMM_K3_M1",
                "class": GaussianHMM,
                "init": {"n_components": 3, "covariance_type": "full", "n_iter": 35, "tol": 1e-4, "cov_reg": 1e-4},
                "X_tr": X_train_base,
                "df_tr": fold.df_train,
                "df_va": fold.df_val,
                "df_te": fold.df_test,
                "feat_cols": base_features,
                "category": "primary",
            },
            {
                "key": "GMM_HMM_K3_M2",
                "class": GMMHMM,
                "init": {"n_components": 3, "n_mix": 2, "covariance_type": "full", "n_iter": 35, "tol": 1e-4, "cov_reg": 1e-3},
                "X_tr": X_train_base,
                "df_tr": fold.df_train,
                "df_va": fold.df_val,
                "df_te": fold.df_test,
                "feat_cols": base_features,
                "category": "primary",
            },
            {
                "key": "GMM_HMM_K3_M3_cov1e-3",  # ANCHOR BASELINE
                "class": GMMHMM,
                "init": {"n_components": 3, "n_mix": 3, "covariance_type": "full", "n_iter": 35, "tol": 1e-4, "cov_reg": 1e-3},
                "X_tr": X_train_base,
                "df_tr": fold.df_train,
                "df_va": fold.df_val,
                "df_te": fold.df_test,
                "feat_cols": base_features,
                "category": "primary_and_anchor",
            },
            # Covariance Regularization Sensitivity (RQ3)
            {
                "key": "GMM_HMM_K3_M3_cov1e-4",
                "class": GMMHMM,
                "init": {"n_components": 3, "n_mix": 3, "covariance_type": "full", "n_iter": 35, "tol": 1e-4, "cov_reg": 1e-4},
                "X_tr": X_train_base,
                "df_tr": fold.df_train,
                "df_va": fold.df_val,
                "df_te": fold.df_test,
                "feat_cols": base_features,
                "category": "cov_reg_sensitivity",
            },
            {
                "key": "GMM_HMM_K3_M3_cov1e-2",
                "class": GMMHMM,
                "init": {"n_components": 3, "n_mix": 3, "covariance_type": "full", "n_iter": 35, "tol": 1e-4, "cov_reg": 1e-2},
                "X_tr": X_train_base,
                "df_tr": fold.df_train,
                "df_va": fold.df_val,
                "df_te": fold.df_test,
                "feat_cols": base_features,
                "category": "cov_reg_sensitivity",
            },
            # Feature-Representation / Diurnal Sensitivity (RQ3)
            {
                "key": "GMM_HMM_K3_M3_diurnal",
                "class": GMMHMM,
                "init": {"n_components": 3, "n_mix": 3, "covariance_type": "full", "n_iter": 35, "tol": 1e-4, "cov_reg": 1e-3},
                "X_tr": X_train_diurnal,
                "df_tr": df_train_diurnal,
                "df_va": df_val_diurnal,
                "df_te": df_test_diurnal,
                "feat_cols": diurnal_features,
                "category": "diurnal_sensitivity",
            },
        ]

        # Execute the 6 fits for this fold
        fitted_models: Dict[str, Any] = {}
        filter_results_test_policy_a: Dict[str, Any] = {}

        for cfg in configs_to_fit:
            m_key = cfg["key"]
            m_cls = cfg["class"]
            m_init = cfg["init"]
            X_tr = cfg["X_tr"]
            df_tr = cfg["df_tr"]
            df_va = cfg["df_va"]
            df_te = cfg["df_te"]
            f_cols = cfg["feat_cols"]

            total_fits_executed += 1
            logger.info(
                "[Fold %d/5 | Fit %d/30] Training %s (category: %s)...",
                fid, total_fits_executed, m_key, cfg["category"],
            )

            # Fit model with deterministic seed schedule
            fit_res = fit_model_with_seed_schedule(
                model_class=m_cls,
                init_kwargs=m_init,
                X_train=X_tr,
                lengths=fold.train_lengths,
                model_name=m_key,
                fold_id=fid,
                n_restarts=n_restarts,
                base_random_state=base_random_state,
            )

            fitted_models[m_key] = fit_res.model

            # Log fit provenance
            provenance_record = {
                "fit_index": total_fits_executed,
                "fold_id": fid,
                "model_name": m_key,
                "category": cfg["category"],
                "base_random_state": fit_res.base_random_state,
                "n_restarts": fit_res.n_restarts,
                "restart_seed_schedule": fit_res.restart_seed_schedule,
                "parameter_sha256": fit_res.parameter_sha256,
                "best_log_likelihood": fit_res.best_log_likelihood,
                "n_iter_done": fit_res.n_iter_done,
                "converged": fit_res.converged,
                "reached_max_iter": fit_res.reached_max_iter,
                "training_time_seconds": round(fit_res.training_time_seconds, 2),
            }
            all_fit_provenance.append(provenance_record)

            logger.info(
                "  -> Fit %d done in %.2fs | Best LL=%.2f | Converged=%s | SHA256=%s",
                total_fits_executed, fit_res.training_time_seconds, fit_res.best_log_likelihood,
                fit_res.converged, fit_res.parameter_sha256[:8],
            )

            # -----------------------------------------------------------------
            # Sequential Forward Filtering: Train -> Val -> Test
            # -----------------------------------------------------------------
            # 1. Filter Train In-Sample to obtain terminal train belief
            res_train = sequential_forward_filter(
                model=fit_res.model,
                df=df_tr,
                feature_cols=f_cols,
                policy="B",  # Fresh start on Train
                gap_handler=gap_handler,
            )

            # 2. Filter Validation (Policy A: Carry-over from Train)
            res_val_pol_a = sequential_forward_filter(
                model=fit_res.model,
                df=df_va,
                feature_cols=f_cols,
                prev_posterior=res_train.terminal_posterior,
                prev_timestamp=res_train.terminal_timestamp,
                policy="A",
                gap_handler=gap_handler,
            )

            # Diagnostic Benchmark: Validation under Policy B (Fresh reset)
            res_val_pol_b = sequential_forward_filter(
                model=fit_res.model,
                df=df_va,
                feature_cols=f_cols,
                policy="B",
                gap_handler=gap_handler,
            )

            # 3. Filter Test (Policy A: Carry-over from Validation)
            res_test_pol_a = sequential_forward_filter(
                model=fit_res.model,
                df=df_te,
                feature_cols=f_cols,
                prev_posterior=res_val_pol_a.terminal_posterior,
                prev_timestamp=res_val_pol_a.terminal_timestamp,
                policy="A",
                gap_handler=gap_handler,
            )
            filter_results_test_policy_a[m_key] = res_test_pol_a

            # Diagnostic Benchmark: Test under Policy B (Fresh reset)
            res_test_pol_b = sequential_forward_filter(
                model=fit_res.model,
                df=df_te,
                feature_cols=f_cols,
                policy="B",
                gap_handler=gap_handler,
            )

            # -----------------------------------------------------------------
            # Regime Interpretation & Hungarian Alignment on Train
            # -----------------------------------------------------------------
            fb_train = compute_smoothed_posteriors(fit_res.model, X_tr, fold.train_lengths)
            state_chars = calculate_state_characteristics(fit_res.model, df_tr, fb_train.gamma, f"{m_key}_f{fid}")
            align_report = align_model_states(
                candidate_chars=state_chars,
                reference_chars=ref_chars,
                scaler=ref_scaler,
                distance_threshold=2.50,
                ambiguity_margin=0.20,
            )

            # Check volatility monotonicity on aligned states
            aligned_mapping = {cand_k: m.matched_ref_state_id for cand_k, m in align_report.matches.items() if m.status == "MATCHED"}
            # In canonical reference: 0 = LowVol, 2 = MedVol, 1 = HighVol
            # Find candidate states assigned to Low, Med, High
            cand_low = next((k for k, v in aligned_mapping.items() if v == 0), None)
            cand_med = next((k for k, v in aligned_mapping.items() if v == 2), None)
            cand_high = next((k for k, v in aligned_mapping.items() if v == 1), None)

            if cand_low is not None and cand_med is not None and cand_high is not None:
                sigma_low = state_chars.states[cand_low].r_oc_std
                sigma_med = state_chars.states[cand_med].r_oc_std
                sigma_high = state_chars.states[cand_high].r_oc_std
                is_monotonic_vol = bool(sigma_low < sigma_med < sigma_high)
            else:
                is_monotonic_vol = False
                sigma_low, sigma_med, sigma_high = None, None, None

            # Record model results for this fold
            fold_record["models"][m_key] = {
                "category": cfg["category"],
                "features": f_cols,
                "val_total_pred_ll_pol_a": round(res_val_pol_a.total_pred_ll, 2),
                "val_mean_pred_ll_pol_a": round(res_val_pol_a.mean_pred_ll, 5),
                "val_total_pred_ll_pol_b": round(res_val_pol_b.total_pred_ll, 2),
                "val_mean_pred_ll_pol_b": round(res_val_pol_b.mean_pred_ll, 5),
                "test_total_pred_ll_pol_a": round(res_test_pol_a.total_pred_ll, 2),
                "test_mean_pred_ll_pol_a": round(res_test_pol_a.mean_pred_ll, 5),
                "test_total_pred_ll_pol_b": round(res_test_pol_b.total_pred_ll, 2),
                "test_mean_pred_ll_pol_b": round(res_test_pol_b.mean_pred_ll, 5),
                "policy_a_vs_b_delta_test_nats": round(res_test_pol_a.total_pred_ll - res_test_pol_b.total_pred_ll, 3),
                "test_switch_frequency": round(res_test_pol_a.switch_frequency, 4),
                "test_mean_entropy": round(float(np.mean(res_test_pol_a.entropy)), 4),
                "alignment": {
                    "match_rate": align_report.candidate_match_rate,
                    "avg_cost": round(align_report.average_matched_cost, 4),
                    "max_cost": round(align_report.max_matched_cost, 4),
                    "global_gap": round(align_report.global_assignment_gap, 4),
                    "is_ambiguous": align_report.is_globally_ambiguous,
                    "matched_mapping": {str(k): int(v.matched_ref_state_id) for k, v in align_report.matches.items() if v.status == "MATCHED"},
                    "volatility_monotonic": is_monotonic_vol,
                    "sigma_returns": {
                        "aligned_low_vol": sigma_low,
                        "aligned_med_vol": sigma_med,
                        "aligned_high_vol": sigma_high,
                    },
                },
                "transition_matrix": fit_res.model.transmat_.tolist(),
                "initial_distribution": fit_res.model.startprob_.tolist(),
            }

            logger.info(
                "  [Eval] %s: Val LL/bar=%.5f | Test LL/bar=%.5f | SwFreq=%.1f%% | MonotonicVol=%s | MatchCost=%.3f",
                m_key, res_val_pol_a.mean_pred_ll, res_test_pol_a.mean_pred_ll,
                res_test_pol_a.switch_frequency * 100, is_monotonic_vol, align_report.average_matched_cost,
            )

        # ---------------------------------------------------------------------
        # Pairwise Statistical Comparisons (Diebold-Mariano & Holm-Bonferroni)
        # on Test Window under Policy A
        # ---------------------------------------------------------------------
        ll_m1 = filter_results_test_policy_a["Gaussian_HMM_K3_M1"].pred_lls
        ll_m2 = filter_results_test_policy_a["GMM_HMM_K3_M2"].pred_lls
        ll_m3 = filter_results_test_policy_a["GMM_HMM_K3_M3_cov1e-3"].pred_lls

        # 3 Pairwise differentials
        diffs = {
            "M3_vs_M1": ll_m3 - ll_m1,
            "M2_vs_M1": ll_m2 - ll_m1,
            "M3_vs_M2": ll_m3 - ll_m2,
        }

        raw_dm_results = {}
        for comp_name, d_seq in diffs.items():
            dm_res = diebold_mariano_test(d_seq, lags=[4, 8, 16])
            raw_dm_results[comp_name] = dm_res

        # Holm-Bonferroni adjustment for each lag L
        for L in [4, 8, 16]:
            raw_p_dict = {k: raw_dm_results[k].p_value[L] for k in diffs.keys()}
            adj_p_dict = holm_bonferroni_adjustment(raw_p_dict)

            fold_record["pairwise_comparisons"][f"lag_{L}"] = {}
            for comp_name in diffs.keys():
                dm_res = raw_dm_results[comp_name]
                fold_record["pairwise_comparisons"][f"lag_{L}"][comp_name] = {
                    "mean_loss_diff_nats": round(dm_res.mean_loss_diff, 6),
                    "hac_variance": round(dm_res.hac_variance[L], 6),
                    "dm_statistic_hln": round(dm_res.dm_statistic[L], 4),
                    "raw_p_value": dm_res.p_value[L],
                    "holm_adjusted_p_value": adj_p_dict[comp_name],
                    "is_significant_adj_05": bool(adj_p_dict[comp_name] < 0.05),
                }

        fold_results[fid] = fold_record
        logger.info("Fold %d evaluation and statistical comparisons completed successfully.", fid)

    # =========================================================================
    # 4. Generate Diagnostic Figures
    # =========================================================================
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig19_path = fig_dir / "19_walk_forward_oos_predictive_likelihood.png"
    fig20_path = fig_dir / "20_regime_parameter_stability_and_alignment.png"

    logger.info("Generating Figure 19: %s...", fig19_path)
    generate_figure_19(fold_results, str(fig19_path))

    logger.info("Generating Figure 20: %s...", fig20_path)
    generate_figure_20(fold_results, str(fig20_path))

    # =========================================================================
    # 5. Compile Final Summary JSON
    # =========================================================================
    total_time = time.perf_counter() - t_start_total
    summary_output = {
        "metadata": {
            "title": "Phase 7 Out-of-Sample Walk-Forward Validation & Cross-Regime Robustness",
            "execution_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "total_fits_executed": total_fits_executed,
            "total_wall_clock_seconds": round(total_time, 2),
            "total_wall_clock_minutes": round(total_time / 60.0, 2),
            "base_random_state": base_random_state,
            "n_restarts": n_restarts,
        },
        "fit_provenance": all_fit_provenance,
        "fold_results": fold_results,
    }

    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    logger.info("Saved complete Phase 7 JSON artifact to %s", out_p)

    logger.info("=" * 80)
    logger.info("PHASE 7 EXPERIMENTS COMPLETED IN %.2f MINUTES (%d FITS)", total_time / 60.0, total_fits_executed)
    logger.info("=" * 80)

    return summary_output


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def generate_figure_19(fold_results: Dict[int, Dict[str, Any]], output_path: str):
    """
    Generate Figure 19:
    Out-of-Sample Predictive Likelihood Trajectories Across Folds & Sensitivity Sweeps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    folds = sorted(fold_results.keys())

    # Panel A: Primary Architectures (RQ2)
    ax = axes[0, 0]
    m1_vals = [fold_results[f]["models"]["Gaussian_HMM_K3_M1"]["test_mean_pred_ll_pol_a"] for f in folds]
    m2_vals = [fold_results[f]["models"]["GMM_HMM_K3_M2"]["test_mean_pred_ll_pol_a"] for f in folds]
    m3_vals = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-3"]["test_mean_pred_ll_pol_a"] for f in folds]

    ax.plot(folds, m1_vals, "o-", color="#1f77b4", linewidth=2.2, label="Gaussian HMM K3-M1")
    ax.plot(folds, m2_vals, "s-", color="#ff7f0e", linewidth=2.2, label="GMM-HMM K3-M2")
    ax.plot(folds, m3_vals, "^-", color="#2ca02c", linewidth=2.2, label="GMM-HMM K3-M3 (Anchor)")
    ax.set_title("A. Architecture Out-of-Sample Predictive LL (Per Bar)", fontweight="bold")
    ax.set_xlabel("Walk-Forward Fold")
    ax.set_ylabel("Predictive Log-Likelihood (nats/bar)")
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel B: Covariance Regularization Sensitivity (RQ3)
    ax = axes[0, 1]
    reg_1e4 = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-4"]["test_mean_pred_ll_pol_a"] for f in folds]
    reg_1e3 = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-3"]["test_mean_pred_ll_pol_a"] for f in folds]
    reg_1e2 = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-2"]["test_mean_pred_ll_pol_a"] for f in folds]

    ax.plot(folds, reg_1e4, "v--", color="#9467bd", linewidth=2.0, label="cov_reg = 1e-4 (0.01% var)")
    ax.plot(folds, reg_1e3, "^-", color="#2ca02c", linewidth=2.2, label="cov_reg = 1e-3 (0.1% var, Baseline)")
    ax.plot(folds, reg_1e2, "d:", color="#d62728", linewidth=2.0, label="cov_reg = 1e-2 (1.0% var)")
    ax.set_title("B. Covariance Regularization Sensitivity (GMM K3-M3)", fontweight="bold")
    ax.set_xlabel("Walk-Forward Fold")
    ax.set_ylabel("Predictive Log-Likelihood (nats/bar)")
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel C: Feature Representation / Diurnal Sensitivity (RQ3)
    ax = axes[1, 0]
    base_feat = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-3"]["test_mean_pred_ll_pol_a"] for f in folds]
    diurn_feat = [fold_results[f]["models"]["GMM_HMM_K3_M3_diurnal"]["test_mean_pred_ll_pol_a"] for f in folds]

    ax.plot(folds, base_feat, "^-", color="#2ca02c", linewidth=2.2, label="[r_oc, log_vol] (Baseline)")
    ax.plot(folds, diurn_feat, "o-.", color="#8c564b", linewidth=2.2, label="[r_oc, range_zscore] (Diurnal-Scaled)")
    ax.set_title("C. Feature Representation / Diurnal Sensitivity", fontweight="bold")
    ax.set_xlabel("Walk-Forward Fold")
    ax.set_ylabel("Predictive Log-Likelihood (nats/bar)")
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel D: Policy A vs Policy B Boundary Impact
    ax = axes[1, 1]
    m1_delta = [fold_results[f]["models"]["Gaussian_HMM_K3_M1"]["policy_a_vs_b_delta_test_nats"] for f in folds]
    m2_delta = [fold_results[f]["models"]["GMM_HMM_K3_M2"]["policy_a_vs_b_delta_test_nats"] for f in folds]
    m3_delta = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-3"]["policy_a_vs_b_delta_test_nats"] for f in folds]

    width = 0.25
    x_arr = np.array(folds)
    ax.bar(x_arr - width, m1_delta, width=width, color="#1f77b4", label="Gaussian K3-M1")
    ax.bar(x_arr, m2_delta, width=width, color="#ff7f0e", label="GMM K3-M2")
    ax.bar(x_arr + width, m3_delta, width=width, color="#2ca02c", label="GMM K3-M3")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("D. Policy A vs Policy B Delta on Test Set (Nats)", fontweight="bold")
    ax.set_xlabel("Walk-Forward Fold")
    ax.set_ylabel("Delta LL (Policy A - Policy B nats)")
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle("Figure 19: Walk-Forward Out-of-Sample Predictive Likelihood Trajectories Across Folds", fontweight="bold", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_figure_20(fold_results: Dict[int, Dict[str, Any]], output_path: str):
    """
    Generate Figure 20:
    Regime Parameter Stability, Volatility Monotonicity & Diebold-Mariano Diagnostics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    folds = sorted(fold_results.keys())

    # Panel A: Aligned Return Volatility by Regime across Folds (GMM K3-M3)
    ax = axes[0, 0]
    low_sig = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-3"]["alignment"]["sigma_returns"]["aligned_low_vol"] * 10000 for f in folds]
    med_sig = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-3"]["alignment"]["sigma_returns"]["aligned_med_vol"] * 10000 for f in folds]
    high_sig = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-3"]["alignment"]["sigma_returns"]["aligned_high_vol"] * 10000 for f in folds]

    ax.plot(folds, low_sig, "o-", color="#2ca02c", linewidth=2.2, label="Aligned Low Volatility")
    ax.plot(folds, med_sig, "s-", color="#ff7f0e", linewidth=2.2, label="Aligned Medium Volatility")
    ax.plot(folds, high_sig, "^-", color="#d62728", linewidth=2.2, label="Aligned High Volatility")
    ax.set_title("A. In-Sample Return Volatility by Aligned Regime (bps)", fontweight="bold")
    ax.set_xlabel("Walk-Forward Fold")
    ax.set_ylabel("Sigma(r_oc) (basis points)")
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel B: Out-of-Sample Causal Switching Rate across Architectures
    ax = axes[0, 1]
    m1_sw = [fold_results[f]["models"]["Gaussian_HMM_K3_M1"]["test_switch_frequency"] * 100 for f in folds]
    m2_sw = [fold_results[f]["models"]["GMM_HMM_K3_M2"]["test_switch_frequency"] * 100 for f in folds]
    m3_sw = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-3"]["test_switch_frequency"] * 100 for f in folds]

    ax.plot(folds, m1_sw, "o-", color="#1f77b4", linewidth=2.2, label="Gaussian K3-M1")
    ax.plot(folds, m2_sw, "s-", color="#ff7f0e", linewidth=2.2, label="GMM K3-M2")
    ax.plot(folds, m3_sw, "^-", color="#2ca02c", linewidth=2.2, label="GMM K3-M3")
    ax.set_title("B. Out-of-Sample Causal Switching Frequency (%)", fontweight="bold")
    ax.set_xlabel("Walk-Forward Fold")
    ax.set_ylabel("Argmax Switch Frequency (%)")
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel C: Diebold-Mariano HLN Test Statistics (Lag L=8)
    ax = axes[1, 0]
    dm_m3_m1 = [fold_results[f]["pairwise_comparisons"]["lag_8"]["M3_vs_M1"]["dm_statistic_hln"] for f in folds]
    dm_m2_m1 = [fold_results[f]["pairwise_comparisons"]["lag_8"]["M2_vs_M1"]["dm_statistic_hln"] for f in folds]
    dm_m3_m2 = [fold_results[f]["pairwise_comparisons"]["lag_8"]["M3_vs_M2"]["dm_statistic_hln"] for f in folds]

    ax.plot(folds, dm_m3_m1, "^-", color="#2ca02c", linewidth=2.2, label="M3 vs M1 (Gaussian)")
    ax.plot(folds, dm_m2_m1, "s-", color="#ff7f0e", linewidth=2.2, label="M2 vs M1 (Gaussian)")
    ax.plot(folds, dm_m3_m2, "o-.", color="#9467bd", linewidth=2.2, label="M3 vs M2")
    ax.axhline(1.96, color="red", linestyle="--", linewidth=1.2, label="p = 0.05 Critical Ceiling (+1.96)")
    ax.axhline(-1.96, color="red", linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("C. Diebold-Mariano HLN Statistic (L=8 Newey-West Lags)", fontweight="bold")
    ax.set_xlabel("Walk-Forward Fold")
    ax.set_ylabel("DM Statistic (z-score)")
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel D: Hungarian Label Alignment Diagnostic Costs
    ax = axes[1, 1]
    cost_m1 = [fold_results[f]["models"]["Gaussian_HMM_K3_M1"]["alignment"]["avg_cost"] for f in folds]
    cost_m2 = [fold_results[f]["models"]["GMM_HMM_K3_M2"]["alignment"]["avg_cost"] for f in folds]
    cost_m3 = [fold_results[f]["models"]["GMM_HMM_K3_M3_cov1e-3"]["alignment"]["avg_cost"] for f in folds]

    ax.plot(folds, cost_m1, "o-", color="#1f77b4", linewidth=2.2, label="Gaussian K3-M1")
    ax.plot(folds, cost_m2, "s-", color="#ff7f0e", linewidth=2.2, label="GMM K3-M2")
    ax.plot(folds, cost_m3, "^-", color="#2ca02c", linewidth=2.2, label="GMM K3-M3")
    ax.axhline(2.50, color="red", linestyle=":", linewidth=1.5, label="d_max Safety Ceiling (2.50)")
    ax.set_title("D. Hungarian Alignment Mean Matching Cost", fontweight="bold")
    ax.set_xlabel("Walk-Forward Fold")
    ax.set_ylabel("Mean Normalized Profile Cost")
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle("Figure 20: Cross-Regime Robustness, Monotonicity, and Diebold-Mariano Diagnostics", fontweight="bold", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 7 Walk-Forward Experiments")
    parser.add_argument("--data-path", type=str, default="data/processed/EURUSD_M15_2023.parquet")
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/reports/phase7_walkforward_results.json")
    args = parser.parse_args()

    run_phase7_experiments(
        data_path=args.data_path,
        n_restarts=args.n_restarts,
        base_random_state=args.seed,
        output_json_path=args.output,
        max_folds=args.max_folds,
    )
