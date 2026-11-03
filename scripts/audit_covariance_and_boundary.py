"""
scripts/audit_covariance_and_boundary.py

Audits:
1. Covariance conditioning (eigenvalues, condition numbers kappa).
2. Covariance regularization sensitivity sweep (lambda in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]).
3. Exact timestamps and gap at the Validation/Test boundary, confirming identical observations.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.splits import split_chronological, identify_continuous_sequences
from src.models.gmm_hmm import GMMHMM
from src.models.forward_filter import ForwardFilter


def main():
    # 1. Load data
    data_path = project_root / "data" / "processed" / "EURUSD_M15_2023_features.parquet"
    df = pd.read_parquet(data_path)
    splits = split_chronological(df, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, filter_incomplete=True)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    feat_cols = ["r_oc", "log_vol"]
    X_train = train_df[feat_cols].values
    X_val = val_df[feat_cols].values
    X_test = test_df[feat_cols].values
    train_lengths, _ = identify_continuous_sequences(train_df)

    # 2. Covariance conditioning on candidate models (from JSON)
    models_dir = project_root / "models"
    cov_audit_results = {}
    for m_file in ["gmm_hmm_k3_m2.json", "gmm_hmm_k3_m3.json"]:
        p = models_dir / m_file
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded_model = GMMHMM.from_dict(data)
        covars = loaded_model.covars_
        K, M, D, _ = covars.shape
        model_name = m_file.replace(".json", "").upper()

        model_res = {"K": K, "M": M, "states": []}
        all_kappas = []
        for k in range(K):
            st_info = {"state": k, "components": []}
            for m in range(M):
                cov = covars[k, m]
                eigenvals = np.linalg.eigvalsh(cov)  # symmetric eigenvalues
                min_eig = float(np.min(eigenvals))
                max_eig = float(np.max(eigenvals))
                kappa = float(max_eig / max(min_eig, 1e-15))
                all_kappas.append(kappa)
                st_info["components"].append({
                    "component": m,
                    "min_eigenvalue": min_eig,
                    "max_eigenvalue": max_eig,
                    "condition_number": kappa,
                    "log10_condition_number": float(np.log10(kappa)),
                    "is_positive_definite": bool(min_eig > 0),
                })
            model_res["states"].append(st_info)
        model_res["max_kappa"] = float(np.max(all_kappas))
        model_res["median_kappa"] = float(np.median(all_kappas))
        cov_audit_results[model_name] = model_res

    # 3. Covariance Regularization Sensitivity Sweep (for K3-M3 and K3-M2)
    reg_sweep_results = {}
    lambdas = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    for cfg_name, K, M in [("GMM_HMM_K3_M2", 3, 2), ("GMM_HMM_K3_M3", 3, 3)]:
        sweep_data = []
        for reg in lambdas:
            gmm = GMMHMM(
                n_components=K,
                n_mix=M,
                covariance_type="full",
                n_iter=35,
                tol=1e-3,
                cov_reg=reg,
                random_state=42 + K * 11 + M * 7,
            )
            gmm.fit(X_train, lengths=train_lengths, n_restarts=3)
            ff = ForwardFilter(model=gmm, gap_policy="reset_prior", max_gap_minutes=15)
            val_res = ff.filter_batch(X_val, timestamps=val_df["timestamp"])
            val_ll = float(np.sum(val_res["pred_log_likelihoods"]))
            test_res = ff.filter_batch(X_test, timestamps=test_df["timestamp"])
            test_ll = float(np.sum(test_res["pred_log_likelihoods"]))

            # check max kappa
            kappas = []
            for k in range(K):
                for m in range(M):
                    evs = np.linalg.eigvalsh(gmm.covars_[k, m])
                    kappas.append(np.max(evs) / max(np.min(evs), 1e-15))

            sweep_data.append({
                "cov_reg": reg,
                "val_predictive_ll": val_ll,
                "test_predictive_ll": test_ll,
                "max_condition_number": float(np.max(kappas)),
                "median_condition_number": float(np.median(kappas)),
            })
        reg_sweep_results[cfg_name] = sweep_data

    # 4. Validation/Test Boundary Audit
    t_val_last = val_df["timestamp"].iloc[-1]
    t_test_first = test_df["timestamp"].iloc[0]
    dt_seconds = (t_test_first - t_val_last).total_seconds()
    dt_minutes = dt_seconds / 60.0

    # Load candidate K3-M3 model
    with open(models_dir / "gmm_hmm_k3_m3.json", "r", encoding="utf-8") as f:
        m3_dict = json.load(f)
    gmm_m3 = GMMHMM.from_dict(m3_dict)
    ff_m3 = ForwardFilter(gmm_m3, gap_policy="reset_prior", max_gap_minutes=15)

    # Carry forward vs Fresh prior
    val_filter_res = ff_m3.filter_batch(X_val, timestamps=val_df["timestamp"])
    final_val_post = val_filter_res["filtered_probs"][-1]

    # Fresh prior
    test_fresh = ff_m3.filter_batch(X_test, timestamps=test_df["timestamp"])
    ll_fresh = float(np.sum(test_fresh["pred_log_likelihoods"]))

    # Carry forward
    is_gap = dt_minutes > 15
    gap_steps = int(round(dt_minutes / 15.0))
    filt0_c, prior0_c, pll0_c = ff_m3.step(
        x_t=X_test[0],
        prev_filtered_prob=final_val_post,
        is_gap=is_gap,
        gap_steps=gap_steps,
    )
    carry_plls = [pll0_c]
    prev_f = filt0_c
    deltas = test_df["timestamp"].diff()
    for t in range(1, len(X_test)):
        gap_flag = pd.notna(deltas.iloc[t]) and (deltas.iloc[t] > pd.Timedelta(minutes=15))
        g_steps = int(round(deltas.iloc[t].total_seconds() / 60.0 / 15.0)) if gap_flag else 1
        filt, prior, pll = ff_m3.step(X_test[t], prev_filtered_prob=prev_f, is_gap=gap_flag, gap_steps=g_steps)
        carry_plls.append(pll)
        prev_f = filt
    ll_carry = float(np.sum(carry_plls))

    boundary_audit = {
        "val_last_timestamp": str(t_val_last),
        "test_first_timestamp": str(t_test_first),
        "elapsed_seconds": dt_seconds,
        "elapsed_minutes": dt_minutes,
        "is_gap_at_boundary": bool(dt_minutes > 15),
        "test_sample_count": len(X_test),
        "test_ll_fresh_prior": ll_fresh,
        "test_ll_carry_forward": ll_carry,
        "difference_nats": ll_carry - ll_fresh,
        "mean_diff_per_bar": (ll_carry - ll_fresh) / len(X_test),
        "bar0_prior_fresh": test_fresh["prior_probs"][0].tolist(),
        "bar0_prior_carry": prior0_c.tolist(),
        "bar0_filtered_fresh": test_fresh["filtered_probs"][0].tolist(),
        "bar0_filtered_carry": filt0_c.tolist(),
        "bar1_filtered_fresh": test_fresh["filtered_probs"][1].tolist(),
        "bar1_filtered_carry": ff_m3.step(X_test[1], prev_filtered_prob=filt0_c)[0].tolist(),
    }

    output = {
        "covariance_conditioning": cov_audit_results,
        "covariance_regularization_sweep": reg_sweep_results,
        "boundary_audit": boundary_audit,
    }

    out_file = project_root / "data" / "reports" / "phase4_covariance_and_boundary_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved audit output to {out_file}")


if __name__ == "__main__":
    main()
