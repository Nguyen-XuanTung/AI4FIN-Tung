"""
scripts/audit_phase4.py

Scientific audit and benchmark script for Phase 4 GMM-HMM.
Executes rigorous benchmarks, stability checks across seeds,
inference latency measurements, and mathematical verifications.
"""

import time
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy import linalg

from src.models.splits import split_chronological
from src.models.gaussian_hmm import GaussianHMM
from src.models.gmm_hmm import GMMHMM
from src.models.forward_filter import ForwardFilter
from src.models.evaluation import (
    calculate_num_free_params,
    calculate_num_free_params_gmm,
    evaluate_model_diagnostics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase4Audit")


def load_data():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "EURUSD_M15_2023_features.parquet"
    df = pd.read_parquet(data_path)
    splits = split_chronological(
        df,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        timestamp_col="timestamp",
        filter_incomplete=True,
    )
    feature_cols = ["r_oc", "log_vol"]
    return splits["train"], splits["val"], splits["test"], feature_cols, splits["metadata"]


def benchmark_training_and_stability(train_df, val_df, test_df, feature_cols):
    """Benchmark training time and stability across 5 independent seeds."""
    logger.info("=== 1. Benchmarking Training Time & Stability Across Seeds ===")
    seeds = [42, 100, 2023, 777, 999]
    configs = [
        ("Gaussian_HMM_K3_M1", 3, 1),
        ("GMM_HMM_K3_M2", 3, 2),
        ("GMM_HMM_K3_M3", 3, 3),
    ]

    results = {}
    for name, K, M in configs:
        train_times = []
        val_lls = []
        test_lls = []
        sw_freqs = []
        min_soft_occs = []
        min_hard_occs = []
        all_models = []

        for seed in seeds:
            t0 = time.perf_counter()
            if M == 1:
                model = GaussianHMM(
                    n_components=K,
                    covariance_type="full",
                    n_iter=35,
                    tol=1e-4,
                    random_state=seed,
                )
            else:
                model = GMMHMM(
                    n_components=K,
                    n_mix=M,
                    covariance_type="full",
                    n_iter=35,
                    tol=1e-4,
                    random_state=seed,
                )
            # Fit with 1 restart to measure clean single-fit training time
            X_train = train_df[feature_cols].values
            model.fit(X_train)
            t_elapsed = time.perf_counter() - t0
            train_times.append(t_elapsed)

            # Evaluate on Validation
            ff = ForwardFilter(model=model, gap_policy="reset_prior", max_gap_minutes=15)
            X_val = val_df[feature_cols].values
            val_res = ff.filter_batch(X_val, timestamps=val_df["timestamp"])
            val_pred_ll = float(np.sum(val_res["pred_log_likelihoods"]))
            val_lls.append(val_pred_ll)

            # Switching freq on Validation
            pred_states = np.argmax(val_res["filtered_probs"], axis=1)
            sw_freq = float(np.sum(pred_states[1:] != pred_states[:-1]) / (len(pred_states) - 1))
            sw_freqs.append(sw_freq)

            # Occupancies
            soft_occ = np.mean(val_res["filtered_probs"], axis=0)
            hard_occ = [np.mean(pred_states == k) for k in range(K)]
            min_soft_occs.append(float(np.min(soft_occ)))
            min_hard_occs.append(float(np.min(hard_occ)))

            # Test LL
            X_test = test_df[feature_cols].values
            test_res = ff.filter_batch(X_test, timestamps=test_df["timestamp"])
            test_pred_ll = float(np.sum(test_res["pred_log_likelihoods"]))
            test_lls.append(test_pred_ll)

            all_models.append(model)

        results[name] = {
            "K": K,
            "M": M,
            "train_time_mean": float(np.mean(train_times)),
            "train_time_std": float(np.std(train_times)),
            "val_pred_ll_mean": float(np.mean(val_lls)),
            "val_pred_ll_std": float(np.std(val_lls)),
            "test_pred_ll_mean": float(np.mean(test_lls)),
            "test_pred_ll_std": float(np.std(test_lls)),
            "sw_freq_mean": float(np.mean(sw_freqs)),
            "sw_freq_std": float(np.std(sw_freqs)),
            "min_soft_occ_mean": float(np.mean(min_soft_occs)),
            "min_soft_occ_std": float(np.std(min_soft_occs)),
            "min_hard_occ_mean": float(np.mean(min_hard_occs)),
            "min_hard_occ_std": float(np.std(min_hard_occs)),
            "sample_model": all_models[0],  # seed 42
        }
        logger.info(
            f"{name}: TrainTime = {np.mean(train_times):.2f}s ± {np.std(train_times):.2f}s | "
            f"ValPredLL = {np.mean(val_lls):.1f} ± {np.std(val_lls):.1f} | "
            f"TestPredLL = {np.mean(test_lls):.1f} ± {np.std(test_lls):.1f} | "
            f"SwFreq = {np.mean(sw_freqs)*100:.1f}% ± {np.std(sw_freqs)*100:.1f}%"
        )

    return results


def benchmark_inference_latency(results, val_df, feature_cols):
    """Benchmark batch filtering and single-bar step filtering latency."""
    logger.info("=== 2. Benchmarking Inference Latency ===")
    X_val = val_df[feature_cols].values
    timestamps = val_df["timestamp"]
    N_val = len(X_val)

    latency_results = {}
    for name, data in results.items():
        model = data["sample_model"]
        ff = ForwardFilter(model=model, gap_policy="reset_prior", max_gap_minutes=15)

        # 1. Batch filtering latency (measure over 30 repetitions)
        batch_times = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = ff.filter_batch(X_val, timestamps=timestamps)
            batch_times.append(time.perf_counter() - t0)

        # 2. Single-bar sequential step latency (measure 2000 steps)
        single_times = []
        x_sample = X_val[0]
        prev_prob = np.ones(model.n_components) / model.n_components
        for _ in range(2000):
            t0 = time.perf_counter()
            filt, prior, pll = ff.step(x_sample, prev_filtered_prob=prev_prob)
            single_times.append(time.perf_counter() - t0)

        latency_results[name] = {
            "batch_time_ms_mean": float(np.mean(batch_times) * 1000.0),
            "batch_time_ms_std": float(np.std(batch_times) * 1000.0),
            "single_bar_us_mean": float(np.mean(single_times) * 1e6),
            "single_bar_us_std": float(np.std(single_times) * 1e6),
            "single_bar_ms_mean": float(np.mean(single_times) * 1000.0),
        }
        logger.info(
            f"{name} Latency: Batch (N={N_val}) = {np.mean(batch_times)*1000.0:.2f}ms ± {np.std(batch_times)*1000.0:.2f}ms | "
            f"Single-Bar Step = {np.mean(single_times)*1e6:.1f}µs ± {np.std(single_times)*1e6:.1f}µs "
            f"({np.mean(single_times)*1000.0:.4f}ms)"
        )

    return latency_results


def audit_test_initialization(val_df, test_df, feature_cols, sample_gmm_m3):
    """Audit how filter initialization affects Test Predictive Log-Likelihood."""
    logger.info("=== 3. Auditing Test Partition Filter Initialization ===")
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    val_ts = val_df["timestamp"]
    test_ts = test_df["timestamp"]

    ff = ForwardFilter(model=sample_gmm_m3, gap_policy="reset_prior", max_gap_minutes=15)

    # Method A: Separate batch on Test starting with startprob_ (current implementation)
    res_A = ff.filter_batch(X_test, timestamps=test_ts)
    ll_A = float(np.sum(res_A["pred_log_likelihoods"]))
    bar0_prior_A = res_A["prior_probs"][0]

    # Method B: Continuous filtering carrying forward Validation final filtered posterior
    res_val = ff.filter_batch(X_val, timestamps=val_ts)
    final_val_prob = res_val["filtered_probs"][-1]

    # Time delta between val end and test start
    t_val_end = val_ts.iloc[-1]
    t_test_start = test_ts.iloc[0]
    dt_minutes = (t_test_start - t_val_end).total_seconds() / 60.0
    logger.info(f"Time gap between Validation end ({t_val_end}) and Test start ({t_test_start}): {dt_minutes:.1f} minutes")

    # Step method B
    is_gap = dt_minutes > 15
    gap_steps = int(round(dt_minutes / 15.0))
    filt0_B, prior0_B, pll0_B = ff.step(
        x_t=X_test[0],
        prev_filtered_prob=final_val_prob,
        is_gap=is_gap,
        gap_steps=gap_steps,
    )
    # Remaining bars in test
    pred_lls_B = [pll0_B]
    prev_f = filt0_B
    deltas = test_ts.diff()
    for t in range(1, len(X_test)):
        gap_flag = pd.notna(deltas.iloc[t]) and (deltas.iloc[t] > pd.Timedelta(minutes=15))
        g_steps = int(round(deltas.iloc[t].total_seconds() / 60.0 / 15.0)) if gap_flag else 1
        filt, prior, pll = ff.step(X_test[t], prev_filtered_prob=prev_f, is_gap=gap_flag, gap_steps=g_steps)
        pred_lls_B.append(pll)
        prev_f = filt
    ll_B = float(np.sum(pred_lls_B))

    logger.info(f"Test LL Method A (New startprob_): {ll_A:.4f}")
    logger.info(f"Test LL Method B (Continuous from Val final posterior): {ll_B:.4f}")
    logger.info(f"Difference (B - A): {ll_B - ll_A:+.4f} nats across {len(X_test)} bars")
    logger.info(f"Bar 0 Prior Method A: {bar0_prior_A}")
    logger.info(f"Bar 0 Prior Method B: {prior0_B}")

    return {
        "test_ll_reset_startprob": ll_A,
        "test_ll_carry_val_posterior": ll_B,
        "delta_nats": ll_B - ll_A,
        "gap_minutes": dt_minutes,
        "bar0_prior_reset": bar0_prior_A.tolist(),
        "bar0_prior_carry": prior0_B.tolist(),
    }


def audit_mixture_components(results):
    """Extract and audit mixture weights, means, and covariance matrices for K3-M2 and K3-M3."""
    logger.info("=== 4. Auditing Mixture Component Parameters & Positive Definiteness ===")
    component_details = {}

    for name in ["GMM_HMM_K3_M2", "GMM_HMM_K3_M3"]:
        model = results[name]["sample_model"]
        K = model.n_components
        M = model.n_mix
        cov_type = model.covariance_type

        details = {"K": K, "M": M, "states": []}
        for k in range(K):
            state_info = {"state": k, "components": []}
            for m in range(M):
                w = float(model.weights_[k, m])
                mu = model.means_[k, m].tolist()
                cov = model.covars_[k, m]
                eigenvals = np.linalg.eigvals(cov)
                is_pos_def = bool(np.all(eigenvals > 0))
                min_eig = float(np.min(eigenvals))

                comp_info = {
                    "component": m,
                    "weight": w,
                    "mean_body_bps": float(mu[0] * 10000.0),
                    "mean_log_vol": float(mu[1]),
                    "cov_matrix": cov.tolist(),
                    "eigenvals": eigenvals.tolist(),
                    "min_eigenvalue": min_eig,
                    "is_positive_definite": is_pos_def,
                }
                state_info["components"].append(comp_info)
            details["states"].append(state_info)
        component_details[name] = details

    return component_details


def audit_emission_consistency(results, val_df, feature_cols):
    """Verify consistency between model._compute_log_emissions and ForwardFilter."""
    logger.info("=== 5. Auditing Emission Likelihood Consistency ===")
    X_val = val_df[feature_cols].values
    consistency_results = {}

    for name in ["Gaussian_HMM_K3_M1", "GMM_HMM_K3_M2", "GMM_HMM_K3_M3"]:
        model = results[name]["sample_model"]
        ff = ForwardFilter(model=model)

        # Batch emissions
        batch_log_b = model._compute_log_emissions(X_val)

        # Single emissions
        single_log_b = np.empty_like(batch_log_b)
        for t in range(len(X_val)):
            single_log_b[t] = ff.compute_single_log_emission(X_val[t])

        max_abs_diff = float(np.max(np.abs(batch_log_b - single_log_b)))
        consistency_results[name] = {
            "max_abs_diff": max_abs_diff,
            "consistent": bool(max_abs_diff < 1e-10),
        }
        logger.info(f"{name} Emission Consistency: Max |diff| = {max_abs_diff:.2e} (Pass: {max_abs_diff < 1e-10})")

    return consistency_results


def main():
    train_df, val_df, test_df, feature_cols, meta = load_data()

    # 1. Training & stability benchmark
    stability_res = benchmark_training_and_stability(train_df, val_df, test_df, feature_cols)

    # 2. Inference latency benchmark
    latency_res = benchmark_inference_latency(stability_res, val_df, feature_cols)

    # 3. Test initialization audit
    test_init_res = audit_test_initialization(val_df, test_df, feature_cols, stability_res["GMM_HMM_K3_M3"]["sample_model"])

    # 4. Mixture component parameters
    comp_details = audit_mixture_components(stability_res)

    # 5. Emission consistency
    emission_res = audit_emission_consistency(stability_res, val_df, feature_cols)

    # Compile comprehensive audit summary
    audit_output = {
        "stability_and_training": {
            k: {
                "train_time_s": f"{v['train_time_mean']:.2f} ± {v['train_time_std']:.2f}",
                "val_pred_ll": f"{v['val_pred_ll_mean']:.1f} ± {v['val_pred_ll_std']:.1f}",
                "test_pred_ll": f"{v['test_pred_ll_mean']:.1f} ± {v['test_pred_ll_std']:.1f}",
                "switch_freq": f"{v['sw_freq_mean']*100:.1f}% ± {v['sw_freq_std']*100:.1f}%",
                "min_soft_occ": f"{v['min_soft_occ_mean']*100:.1f}% ± {v['min_soft_occ_std']*100:.1f}%",
                "min_hard_occ": f"{v['min_hard_occ_mean']*100:.1f}% ± {v['min_hard_occ_std']*100:.1f}%",
            }
            for k, v in stability_res.items()
        },
        "latency_benchmarks": latency_res,
        "test_initialization": test_init_res,
        "component_details": comp_details,
        "emission_consistency": emission_res,
    }

    out_json = Path(__file__).resolve().parent.parent / "data" / "reports" / "phase4_scientific_audit_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_output, f, indent=2)
    logger.info(f"Saved complete scientific audit results to {out_json}")


if __name__ == "__main__":
    main()
