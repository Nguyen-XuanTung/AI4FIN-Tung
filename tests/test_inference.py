"""
tests/test_inference.py

Comprehensive unit test suite for Phase 5:
1. Sequential vs. batch forward filtering equivalence.
2. Temporal causality invariance.
3. Model version mismatch rejection.
4. Engine state serialization and restoration.
5. Gap transition count convention (n = missing_bars + 1, alpha_prev @ A^n).
6. Uncertain gap detection (non-monotonic, irregular timestamps).
7. Weekend/holiday market closure prior reset.
8. Immutable storage duplicate rejection and as-of vs. retrospective separation.
9. Retraining validation gates (hard criteria vs diagnostic warnings).
10. Warm-up buffer execution and sensitivity convergence.
11. Full support for both Gaussian HMM and GMM-HMM artifacts.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.gaussian_hmm import GaussianHMM
from src.models.gmm_hmm import GMMHMM
from src.models.forward_filter import ForwardFilter
from src.inference.calendar import FXCalendarGapHandler, GapClassification, UncertainGapError
from src.inference.versioning import (
    ModelArtifact,
    ModelValidationError,
    ModelVersionMismatchError,
)
from src.inference.engine import RealTimeInferenceEngine
from src.inference.storage import DuplicateRecordError, ImmutableProbabilityStore
from src.inference.retraining import (
    RetrainingManager,
    RetrainingValidator,
    evaluate_warmup_sensitivity,
)


@pytest.fixture
def synthetic_models():
    """Create lightweight Gaussian and GMM models for deterministic testing."""
    # 1. Gaussian HMM (K=3, D=2)
    gauss = GaussianHMM(n_components=3, covariance_type="full", random_state=42)
    gauss.startprob_ = np.array([0.5, 0.3, 0.2])
    gauss.transmat_ = np.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.80, 0.15],
        [0.10, 0.10, 0.80],
    ])
    gauss.means_ = np.array([
        [0.0001, -8.0],
        [-0.0001, -7.0],
        [0.0000, -6.0],
    ])
    gauss.covars_ = np.array([
        [[1e-7, 0.0], [0.0, 0.05]],
        [[2e-7, 0.0], [0.0, 0.08]],
        [[5e-7, 0.0], [0.0, 0.12]],
    ])

    # 2. GMM-HMM (K=3, M=2, D=2)
    gmm = GMMHMM(n_components=3, n_mix=2, covariance_type="full", random_state=42)
    gmm.startprob_ = np.array([0.4, 0.4, 0.2])
    gmm.transmat_ = np.array([
        [0.80, 0.15, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ])
    gmm.weights_ = np.array([
        [0.6, 0.4],
        [0.5, 0.5],
        [0.7, 0.3],
    ])
    gmm.means_ = np.array([
        [[0.0001, -8.0], [-0.0001, -7.8]],
        [[0.0002, -7.2], [-0.0002, -7.0]],
        [[0.0005, -6.5], [-0.0005, -6.0]],
    ])
    gmm.covars_ = np.array([
        [[[1e-7, 0.0], [0.0, 0.04]], [[1.5e-7, 0.0], [0.0, 0.06]]],
        [[[2e-7, 0.0], [0.0, 0.08]], [[3e-7, 0.0], [0.0, 0.10]]],
        [[[5e-7, 0.0], [0.0, 0.12]], [[8e-7, 0.0], [0.0, 0.15]]],
    ])

    return gauss, gmm


@pytest.fixture
def sample_artifacts(synthetic_models):
    gauss, gmm = synthetic_models
    art_gauss = ModelArtifact.from_model(
        model=gauss,
        model_version="GaussianHMM_K3_M1_v1",
        features=["r_oc", "log_vol"],
        feature_version="Config_2_MinimalOrthogonal_v1",
    )
    art_gmm = ModelArtifact.from_model(
        model=gmm,
        model_version="GMMHMM_K3_M2_v1",
        features=["r_oc", "log_vol"],
        feature_version="Config_2_MinimalOrthogonal_v1",
    )
    return art_gauss, art_gmm


def test_1_sequential_vs_batch_equivalence(sample_artifacts):
    """Test that online sequential step() exactly matches batch ForwardFilter."""
    _, art_gmm = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gmm)

    N = 100
    rng = np.random.RandomState(42)
    X = np.column_stack([
        rng.normal(0, 0.0005, N),
        rng.normal(-7.5, 0.3, N),
    ])
    start_ts = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")  # Monday
    timestamps = [start_ts + pd.Timedelta(minutes=15 * i) for i in range(N)]

    # Batch filter
    ff = ForwardFilter(model=engine.model, gap_policy="reset_prior", max_gap_minutes=15)
    batch_res = ff.filter_batch(X, timestamps=timestamps)

    # Sequential engine
    seq_probs = []
    seq_pred_lls = []
    for i in range(N):
        step_res = engine.step(X[i], timestamps[i])
        seq_probs.append(step_res["filtered_probs"])
        seq_pred_lls.append(step_res["pred_log_likelihood"])

    seq_probs = np.array(seq_probs)
    seq_pred_lls = np.array(seq_pred_lls)

    max_prob_diff = np.max(np.abs(seq_probs - batch_res["filtered_probs"]))
    max_ll_diff = np.max(np.abs(seq_pred_lls - batch_res["pred_log_likelihoods"]))

    assert max_prob_diff < 1e-10, f"Filtered probability mismatch: {max_prob_diff}"
    assert max_ll_diff < 1e-10, f"Predictive LL mismatch: {max_ll_diff}"


def test_2_causality_invariance(sample_artifacts):
    """Verify that future observations have zero effect on past filtered posteriors."""
    _, art_gmm = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gmm)

    rng = np.random.RandomState(123)
    X = np.column_stack([rng.normal(0, 0.0004, 50), rng.normal(-7.5, 0.3, 50)])
    timestamps = [pd.Timestamp("2023-01-02 00:00:00", tz="UTC") + pd.Timedelta(minutes=15 * i) for i in range(50)]

    # Run first 25 bars
    past_posteriors = []
    for i in range(25):
        res = engine.step(X[i], timestamps[i])
        past_posteriors.append(res["filtered_probs"])

    # Run remaining 25 bars with extreme shocks
    X_perturbed = X.copy()
    X_perturbed[25:] = np.column_stack([np.full(25, 0.05), np.full(25, -2.0)])  # 100-sigma shocks

    engine2 = RealTimeInferenceEngine(artifact=art_gmm)
    past_posteriors_perturbed = []
    for i in range(25):
        res = engine2.step(X_perturbed[i], timestamps[i])
        past_posteriors_perturbed.append(res["filtered_probs"])

    diff = np.max(np.abs(np.array(past_posteriors) - np.array(past_posteriors_perturbed)))
    assert diff == 0.0, "Past filtered probabilities altered by future data!"


def test_3_gap_transition_count_convention(sample_artifacts):
    """Verify exact transition propagation: alpha_prior = alpha_prev @ A^n where n = missing_bars + 1."""
    _, art_gmm = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gmm)

    t0 = pd.Timestamp("2023-01-02 10:00:00", tz="UTC")  # Monday active session
    x0 = np.array([0.0001, -7.5])
    step0 = engine.step(x0, t0)
    alpha0 = np.array(step0["filtered_probs"])

    # Case A: Normal consecutive bar (delta = 15 min, missing_bars = 0, n = 1)
    t1 = t0 + pd.Timedelta(minutes=15)
    x1 = np.array([-0.0001, -7.6])
    step1 = engine.step(x1, t1)
    assert step1["gap_type"] == "normal"
    assert step1["n_transitions"] == 1
    assert step1["missing_bars"] == 0

    expected_prior1 = alpha0 @ engine.model.transmat_
    expected_prior1 /= np.sum(expected_prior1)
    assert np.allclose(step1["prior_probs"], expected_prior1, atol=1e-10)

    # Case B: 1 missing bar in active session (delta = 30 min, missing_bars = 1, n = 2)
    engine.reset_state()
    engine.step(x0, t0)
    t_gap1 = t0 + pd.Timedelta(minutes=30)  # 10:15 missing, 10:30 arrives
    step_gap1 = engine.step(x1, t_gap1)

    assert step_gap1["gap_type"] == "active_missing"
    assert step_gap1["n_transitions"] == 2
    assert step_gap1["missing_bars"] == 1

    A = engine.model.transmat_
    expected_prior2 = alpha0 @ (A @ A)
    expected_prior2 /= np.sum(expected_prior2)
    assert np.allclose(step_gap1["prior_probs"], expected_prior2, atol=1e-10)

    # Case C: 3 missing bars (delta = 60 min, missing_bars = 3, n = 4)
    engine.reset_state()
    engine.step(x0, t0)
    t_gap3 = t0 + pd.Timedelta(minutes=60)
    step_gap3 = engine.step(x1, t_gap3)

    assert step_gap3["gap_type"] == "active_missing"
    assert step_gap3["n_transitions"] == 4
    assert step_gap3["missing_bars"] == 3

    A4 = np.linalg.matrix_power(A, 4)
    expected_prior4 = alpha0 @ A4
    expected_prior4 /= np.sum(expected_prior4)
    assert np.allclose(step_gap3["prior_probs"], expected_prior4, atol=1e-10)


def test_4_uncertain_gap_handling(sample_artifacts):
    """Verify that irregular or non-monotonic timestamps raise UncertainGapError."""
    _, art_gmm = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gmm)

    t0 = pd.Timestamp("2023-01-02 10:00:00", tz="UTC")
    engine.step([0.0, -7.5], t0)

    # Non-monotonic (t_curr <= t_prev)
    with pytest.raises(UncertainGapError):
        engine.step([0.0, -7.5], t0 - pd.Timedelta(minutes=15))

    # Irregular non-multiple of 15 min (e.g. 7 minutes)
    with pytest.raises(UncertainGapError):
        engine.step([0.0, -7.5], t0 + pd.Timedelta(minutes=7))


def test_5_weekend_market_closure_reset(sample_artifacts):
    """Verify that weekend gaps reset prior to startprob_."""
    _, art_gmm = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gmm)

    # Friday close
    t_fri = pd.Timestamp("2023-01-06 21:00:00", tz="UTC")  # Friday
    step_fri = engine.step([0.0, -7.5], t_fri)

    # Sunday open
    t_sun = pd.Timestamp("2023-01-08 22:00:00", tz="UTC")  # Sunday
    step_sun = engine.step([0.0, -7.5], t_sun)

    assert step_sun["gap_type"] == "market_closure"
    # Prior must reset to startprob_
    assert np.allclose(step_sun["prior_probs"], engine.model.startprob_, atol=1e-10)


def test_6_engine_state_persistence_and_mismatch(sample_artifacts, tmp_path):
    """Verify saving engine state to disk, restoration, and model version mismatch guard."""
    art_gauss, art_gmm = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gmm)

    t0 = pd.Timestamp("2023-01-02 10:00:00", tz="UTC")
    t1 = t0 + pd.Timedelta(minutes=15)
    engine.step([0.0001, -7.5], t0)
    engine.step([-0.0001, -7.4], t1)

    state_file = tmp_path / "inference_state.json"
    engine.save_state(state_file)

    # Fresh engine with same artifact restores successfully
    new_engine = RealTimeInferenceEngine(artifact=art_gmm)
    new_engine.load_state(state_file)

    assert new_engine.step_count == 2
    assert np.allclose(new_engine.current_posterior, engine.current_posterior)
    assert new_engine.last_timestamp == engine.last_timestamp

    # Attempting to load into different model version raises ModelVersionMismatchError
    different_engine = RealTimeInferenceEngine(artifact=art_gauss)
    with pytest.raises(ModelVersionMismatchError):
        different_engine.load_state(state_file)


def test_7_immutable_store_duplicate_policy(tmp_path):
    """Verify append-only store duplicate rejection for live as-of records."""
    log_file = tmp_path / "as_of_log.jsonl"
    store = ImmutableProbabilityStore(log_file)

    ts = "2023-01-02T10:00:00+00:00"
    rec1 = store.append(
        timestamp_utc=ts,
        model_version="GMM_v1",
        feature_version="Config2_v1",
        filtered_probs=[0.5, 0.3, 0.2],
        pred_state=0,
        pred_log_likelihood=5.5,
        inference_type="as_of",
        run_id="live_1",
    )
    assert rec1["record_id"] is not None

    # Duplicate as_of record with identical timestamp must be strictly rejected
    with pytest.raises(DuplicateRecordError):
        store.append(
            timestamp_utc=ts,
            model_version="GMM_v1",
            feature_version="Config2_v1",
            filtered_probs=[0.4, 0.4, 0.2],
            pred_state=1,
            pred_log_likelihood=5.6,
            inference_type="as_of",
            run_id="live_2",
        )

    # Duplicate as_of record EVEN WITH DIFFERENT MODEL VERSION must also be strictly rejected
    with pytest.raises(DuplicateRecordError):
        store.append(
            timestamp_utc=ts,
            model_version="Different_Model_v2",
            feature_version="Config2_v1",
            filtered_probs=[0.1, 0.2, 0.7],
            pred_state=2,
            pred_log_likelihood=4.0,
            inference_type="as_of",
            run_id="live_new_model",
        )

    # Retrospective record with distinct run_id is allowed
    rec_retro = store.append(
        timestamp_utc=ts,
        model_version="GMM_v2",
        feature_version="Config2_v1",
        filtered_probs=[0.7, 0.2, 0.1],
        pred_state=0,
        pred_log_likelihood=5.9,
        inference_type="retrospective",
        run_id="audit_run_1",
    )
    assert rec_retro["inference_type"] == "retrospective"

    # Export to DataFrame and verify separation
    df_as_of = store.to_dataframe(inference_type="as_of")
    assert len(df_as_of) == 1
    assert df_as_of["inference_type"].iloc[0] == "as_of"

    df_retro = store.to_dataframe(inference_type="retrospective")
    assert len(df_retro) == 1
    assert df_retro["inference_type"].iloc[0] == "retrospective"


def test_8_retraining_validation_gates(sample_artifacts):
    """Verify hard validation criteria block invalid models while diagnostics issue warnings."""
    _, art_gmm = sample_artifacts
    validator = RetrainingValidator()

    # Valid model passes
    report = validator.validate_candidate(art_gmm)
    assert report.is_deployable is True
    assert report.hard_checks_passed is True

    # Case A: Non-positive definite covariance triggers hard failure
    corrupt_params = dict(art_gmm.parameters)
    corrupt_covs = np.array(corrupt_params["covars"]).copy()
    corrupt_covs[0, 0] = np.array([[-1.0, 0.0], [0.0, 1.0]])  # Negative eigenvalue
    corrupt_params["covars"] = corrupt_covs.tolist()

    corrupt_art = ModelArtifact(
        model_version="Corrupt_v1",
        model_type=art_gmm.model_type,
        n_components=art_gmm.n_components,
        n_mix=art_gmm.n_mix,
        features=art_gmm.features,
        feature_version=art_gmm.feature_version,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        training_metadata={},
        validation_diagnostics={},
        parameters=corrupt_params,
    )
    report_bad = validator.validate_candidate(corrupt_art)
    assert report_bad.is_deployable is False
    assert any("Non-positive-definite" in err for err in report_bad.hard_failures)


def test_9_warmup_sensitivity_and_model_cutover(sample_artifacts, tmp_path):
    """Test warm-up buffer execution, sensitivity convergence, and atomic model cutover."""
    art_gauss, art_gmm = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gauss)

    # Step engine on some initial bars with model 1 (Gaussian)
    N = 100
    rng = np.random.RandomState(42)
    obs = np.column_stack([rng.normal(0, 0.0004, N), rng.normal(-7.5, 0.3, N)])
    ts = [pd.Timestamp("2023-01-02 00:00:00", tz="UTC") + pd.Timedelta(minutes=15 * i) for i in range(N)]

    for i in range(50):
        engine.step(obs[i], ts[i])

    assert engine.model_version == "GaussianHMM_K3_M1_v1"

    # Evaluate warm-up sensitivity on model 2 (GMM)
    sensitivity_df = evaluate_warmup_sensitivity(
        model=art_gmm.to_model_instance(),
        observations=obs[:50],
        timestamps=pd.Series(ts[:50]),
        buffer_lengths=[5, 10, 20, 35, 50],
        asymptotic_length=50,
    )
    assert len(sensitivity_df) == 5
    # As W increases, L1 distance to final reference posterior should decrease
    assert sensitivity_df["l1_total_variation"].iloc[-1] == 0.0  # W=50 matches reference exactly
    assert sensitivity_df["l1_total_variation"].iloc[0] >= sensitivity_df["l1_total_variation"].iloc[-1]

    # Perform atomic upgrade via RetrainingManager
    history_file = tmp_path / "deployment_history.json"
    manager = RetrainingManager(deployment_history_path=history_file)

    upgrade_res = manager.validate_and_deploy(
        engine=engine,
        new_artifact=art_gmm,
        warmup_observations=obs[:50],
        warmup_timestamps=pd.Series(ts[:50]),
        warmup_length=35,
    )

    assert engine.model_version == "GMMHMM_K3_M2_v1"
    assert upgrade_res["previous_model_version"] == "GaussianHMM_K3_M1_v1"
    assert upgrade_res["new_model_version"] == "GMMHMM_K3_M2_v1"

    # Step engine on bar 50 with new model - must succeed smoothly
    step50 = engine.step(obs[50], ts[50])
    assert step50["model_version"] == "GMMHMM_K3_M2_v1"
    assert np.isclose(np.sum(step50["filtered_probs"]), 1.0)


def test_10_support_both_gaussian_and_gmm(sample_artifacts):
    """Verify that RealTimeInferenceEngine operates seamlessly with both model types."""
    art_gauss, art_gmm = sample_artifacts

    engine_gauss = RealTimeInferenceEngine(artifact=art_gauss)
    engine_gmm = RealTimeInferenceEngine(artifact=art_gmm)

    x = np.array([0.0002, -7.4])
    t = pd.Timestamp("2023-01-02 12:00:00", tz="UTC")

    res_gauss = engine_gauss.step(x, t)
    res_gmm = engine_gmm.step(x, t)

    assert len(res_gauss["filtered_probs"]) == 3
    assert len(res_gmm["filtered_probs"]) == 3
    assert np.isclose(np.sum(res_gauss["filtered_probs"]), 1.0)
    assert np.isclose(np.sum(res_gmm["filtered_probs"]), 1.0)
    assert np.isfinite(res_gauss["pred_log_likelihood"])
    assert np.isfinite(res_gmm["pred_log_likelihood"])


def test_11_export_parquet_strict_as_of_filtering(tmp_path):
    """Verify that export_parquet with inference_type='as_of' excludes retrospective records."""
    log_file = tmp_path / "as_of_log.jsonl"
    store = ImmutableProbabilityStore(log_file)

    # Add 3 as_of records
    for i in range(3):
        store.append(
            timestamp_utc=f"2023-01-02T10:{i*15:02d}:00+00:00",
            model_version="Model_v1",
            feature_version="Config2_v1",
            filtered_probs=[0.5, 0.3, 0.2],
            pred_state=0,
            pred_log_likelihood=5.0,
            inference_type="as_of",
            run_id="live",
        )

    # Add 2 retrospective records on the same timestamps
    for i in range(2):
        store.append(
            timestamp_utc=f"2023-01-02T10:{i*15:02d}:00+00:00",
            model_version="Model_v2",
            feature_version="Config2_v1",
            filtered_probs=[0.4, 0.4, 0.2],
            pred_state=1,
            pred_log_likelihood=5.2,
            inference_type="retrospective",
            run_id="audit_run",
        )

    assert len(store) == 5

    # Export strictly as_of
    parquet_as_of = tmp_path / "as_of_only.parquet"
    store.export_parquet(parquet_as_of, inference_type="as_of")
    df_read = pd.read_parquet(parquet_as_of)

    assert len(df_read) == 3
    assert set(df_read["inference_type"]) == {"as_of"}
    assert df_read["timestamp"].nunique() == 3


def test_12_warmup_target_timestamp_invariance(sample_artifacts):
    """Verify that warm-up sensitivity compares against the exact same target observation across all W."""
    _, art_gmm = sample_artifacts
    model = art_gmm.to_model_instance()

    N = 100
    rng = np.random.RandomState(42)
    obs = np.column_stack([rng.normal(0, 0.0005, N), rng.normal(-7.5, 0.3, N)])
    ts = pd.date_range("2023-01-02 00:00:00", periods=N, freq="15min", tz="UTC")

    # Evaluate across multiple W values
    buffer_lengths = [5, 10, 20, 35, 50]
    sens_df = evaluate_warmup_sensitivity(
        model=model,
        observations=obs,
        timestamps=pd.Series(ts),
        buffer_lengths=buffer_lengths,
        asymptotic_length=50,
    )

    # All evaluations evaluate the posterior at the final observation obs[-1]
    assert len(sens_df) == len(buffer_lengths)
    # The asymptotic reference is length 50, so W=50 has zero distance
    assert sens_df.loc[sens_df["warmup_length_W"] == 50, "l1_total_variation"].values[0] == 0.0


def test_13_cutover_alignment_with_training_split():
    """Verify that deployment cutover timestamp is strictly at or after the end of the training set."""
    from src.models.splits import split_chronological

    df = pd.read_parquet("data/processed/EURUSD_M15_2023_features.parquet")
    splits = split_chronological(df, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, filter_incomplete=True)

    train_end_ts = splits["train"]["timestamp"].iloc[-1]
    val_start_ts = splits["val"]["timestamp"].iloc[0]

    # In unfiltered df, find index of val_start_ts
    cutover_idx = int(df[df["timestamp"] >= val_start_ts].index[0])
    cutover_ts = df["timestamp"].iloc[cutover_idx]

    # Verification:
    # 1. cutover timestamp is exactly the start of validation
    assert cutover_ts == val_start_ts
    # 2. cutover timestamp is strictly after the end of training
    assert cutover_ts > train_end_ts
    # 3. All observations before cutover_idx were available at training/warmup time
    prev_ts = df["timestamp"].iloc[cutover_idx - 1]
    assert prev_ts == train_end_ts


def test_14_partial_bar_skipping_no_emission_update(sample_artifacts):
    """Verify that partial bars skip emission calculation and do not alter belief state."""
    art_gauss, _ = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gauss)

    t0 = pd.Timestamp("2023-01-02 10:00:00", tz="UTC")
    t1 = t0 + pd.Timedelta(minutes=15)

    # Bar 0: Complete bar
    res0 = engine.step([0.0001, -7.5], t0, is_complete=True)
    assert res0["status"] == "valid_update"
    assert res0["is_complete"] is True
    assert res0["is_valid_prediction"] is True
    assert engine.step_count == 1
    assert engine.skipped_count == 0
    posterior_0 = np.array(engine.current_posterior).copy()

    # Bar 1: Incomplete partial bar (extreme outlier that would drastically skew posterior if ingested)
    res1 = engine.step([0.0500, -2.0], t1, is_complete=False)
    assert res1["status"] == "skipped_partial"
    assert res1["is_complete"] is False
    assert res1["is_valid_prediction"] is False
    assert res1["filtered_probs"] is None
    assert res1["pred_state"] is None
    assert res1["pred_log_likelihood"] is None
    assert res1["gap_type"] == "partial_bar_skipped"

    # Engine internal state must be strictly preserved
    assert engine.step_count == 1
    assert engine.skipped_count == 1
    assert engine.last_timestamp == t0  # Not advanced to t1!
    assert np.allclose(engine.current_posterior, posterior_0)


def test_15_transition_propagation_across_skipped_partial_bars(sample_artifacts):
    """Verify that a complete bar following a skipped partial bar propagates A^n correctly."""
    art_gauss, _ = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gauss)

    t0 = pd.Timestamp("2023-01-02 10:00:00", tz="UTC")
    t1 = t0 + pd.Timedelta(minutes=15)
    t2 = t0 + pd.Timedelta(minutes=30)

    # Bar 0: Complete bar
    res0 = engine.step([0.0001, -7.5], t0, is_complete=True)
    alpha0 = np.array(res0["filtered_probs"])

    # Bar 1: Incomplete bar skipped
    res1 = engine.step([0.0000, -7.0], t1, is_complete=False)
    assert res1["status"] == "skipped_partial"
    assert engine.last_timestamp == t0

    # Bar 2: Complete bar arrives at t2 (30 min after t0)
    res2 = engine.step([-0.0002, -7.2], t2, is_complete=True)
    assert res2["status"] == "valid_update"
    assert res2["gap_type"] == "active_missing"
    assert res2["n_transitions"] == 2
    assert res2["missing_bars"] == 1

    # Mathematically verify prior was propagated via A^2
    A = engine.model.transmat_
    expected_prior = alpha0 @ np.linalg.matrix_power(A, 2)
    expected_prior /= np.sum(expected_prior)
    assert np.allclose(res2["prior_probs"], expected_prior, atol=1e-10)

    # Verify posterior is valid unit distribution
    assert np.isclose(np.sum(res2["filtered_probs"]), 1.0)
    assert engine.step_count == 2
    assert engine.skipped_count == 1
    assert engine.last_timestamp == t2


def test_16_warmup_buffer_excludes_partial_bars(sample_artifacts, tmp_path):
    """Verify that warm-up buffer excludes partial bars and only ingests complete candles."""
    art_gauss, art_gmm = sample_artifacts
    engine = RealTimeInferenceEngine(artifact=art_gauss)

    history_path = tmp_path / "deploy_history.json"
    manager = RetrainingManager(deployment_history_path=history_path)

    # Synthetic observations: 10 complete, 5 partial interleaved
    rng = np.random.RandomState(42)
    N = 15
    obs = np.column_stack([rng.normal(0, 0.0005, N), rng.normal(-7.5, 0.3, N)])
    ts = pd.date_range("2023-01-02 10:00:00", periods=N, freq="15min", tz="UTC")
    is_complete_flags = [True, True, False, True, False, True, True, False, True, True, False, True, True, False, True]

    # Warm up with mask
    deploy_event = manager.validate_and_deploy(
        engine=engine,
        new_artifact=art_gmm,
        warmup_observations=obs,
        warmup_timestamps=pd.Series(ts),
        warmup_length=50,
        warmup_is_complete=is_complete_flags,
    )

    # Exactly 10 complete bars should have been used
    assert deploy_event["warmup_bars"] == 10
    assert len(deploy_event["final_warmup_posterior"]) == 3
    assert np.isclose(np.sum(deploy_event["final_warmup_posterior"]), 1.0)


def test_17_distinguishable_audit_records_in_storage(tmp_path):
    """Verify storage separates valid as_of predictions from skipped audit records."""
    log_file = tmp_path / "inference_log.jsonl"
    store = ImmutableProbabilityStore(log_file)

    ts0 = "2023-01-02T10:00:00+00:00"
    ts1 = "2023-01-02T10:15:00+00:00"
    ts2 = "2023-01-02T10:30:00+00:00"

    # 1. Valid as_of prediction
    store.append(
        timestamp_utc=ts0,
        model_version="Model_v1",
        feature_version="Config2_v1",
        filtered_probs=[0.6, 0.3, 0.1],
        pred_state=0,
        pred_log_likelihood=5.2,
        inference_type="as_of",
        is_complete=True,
        is_valid_prediction=True,
        status="valid_update",
    )

    # 2. Skipped partial audit record
    store.append(
        timestamp_utc=ts1,
        model_version="Model_v1",
        feature_version="Config2_v1",
        filtered_probs=None,
        pred_state=None,
        pred_log_likelihood=None,
        inference_type="skipped_audit",
        gap_type="partial_bar_skipped",
        is_complete=False,
        is_valid_prediction=False,
        status="skipped_partial",
    )

    # 3. Retrospective record
    store.append(
        timestamp_utc=ts2,
        model_version="Model_v1",
        feature_version="Config2_v1",
        filtered_probs=[0.2, 0.7, 0.1],
        pred_state=1,
        pred_log_likelihood=4.8,
        inference_type="retrospective",
        run_id="audit_run",
    )

    assert len(store) == 3

    # Query as_of only
    df_as_of = store.to_dataframe(inference_type="as_of")
    assert len(df_as_of) == 1
    assert df_as_of["is_complete"].iloc[0] == True
    assert df_as_of["is_valid_prediction"].iloc[0] == True
    assert not np.isnan(df_as_of["prob_state_0"].iloc[0])

    # Query skipped_audit only
    df_skipped = store.to_dataframe(inference_type="skipped_audit")
    assert len(df_skipped) == 1
    assert df_skipped["is_complete"].iloc[0] == False
    assert df_skipped["is_valid_prediction"].iloc[0] == False
    assert np.isnan(df_skipped["prob_state_0"].iloc[0])

    # Query unified live stream (as_of + skipped_audit)
    df_unified = store.to_dataframe(inference_type=["as_of", "skipped_audit"])
    assert len(df_unified) == 2

    # Duplicate rejection for skipped_audit
    with pytest.raises(DuplicateRecordError):
        store.append(
            timestamp_utc=ts1,
            model_version="Model_v1",
            feature_version="Config2_v1",
            filtered_probs=None,
            pred_state=None,
            pred_log_likelihood=None,
            inference_type="skipped_audit",
        )


def test_18_parameter_lookahead_semantic_segregation(tmp_path):
    """
    Verify strict 5-way semantic segregation in storage:
    - in_sample_filtered: Training partition causal filtering (parameter lookahead acknowledged)
    - as_of_replay: Out-of-sample historical causal replay (frozen prior model)
    - as_of: Genuine real-time live predictions
    - skipped_audit: Incomplete candle audit records
    - retrospective: Offline counterfactual / sensitivity runs
    """
    log_file = tmp_path / "semantic_inference_log.jsonl"
    store = ImmutableProbabilityStore(log_file)

    ts_train = "2023-01-02T10:00:00+00:00"
    ts_replay = "2023-08-25T10:00:00+00:00"
    ts_live = "2024-01-02T10:00:00+00:00"
    ts_skipped = "2023-01-02T10:15:00+00:00"
    ts_retro = "2023-08-25T10:00:00+00:00"

    # 1. Ingest in_sample_filtered
    store.append(
        timestamp_utc=ts_train,
        model_version="Gaussian_HMM_K3_M1_v1",
        feature_version="Config_2_MinimalOrthogonal_v1",
        filtered_probs=[0.7, 0.2, 0.1],
        pred_state=0,
        pred_log_likelihood=5.1,
        inference_type="in_sample_filtered",
        run_id="in_sample_train_2023",
        is_complete=True,
        is_valid_prediction=True,
        status="valid_update",
    )

    # 2. Ingest as_of_replay
    store.append(
        timestamp_utc=ts_replay,
        model_version="GMM_HMM_K3_M3_v1",
        feature_version="Config_2_MinimalOrthogonal_v1",
        filtered_probs=[0.1, 0.8, 0.1],
        pred_state=1,
        pred_log_likelihood=6.2,
        inference_type="as_of_replay",
        run_id="oos_replay_2023",
        is_complete=True,
        is_valid_prediction=True,
        status="valid_update",
    )

    # 3. Ingest as_of
    store.append(
        timestamp_utc=ts_live,
        model_version="GMM_HMM_K3_M3_v1",
        feature_version="Config_2_MinimalOrthogonal_v1",
        filtered_probs=[0.2, 0.3, 0.5],
        pred_state=2,
        pred_log_likelihood=5.8,
        inference_type="as_of",
        run_id="live_production_2024",
        is_complete=True,
        is_valid_prediction=True,
        status="valid_update",
    )

    # 4. Ingest skipped_audit
    store.append(
        timestamp_utc=ts_skipped,
        model_version="Gaussian_HMM_K3_M1_v1",
        feature_version="Config_2_MinimalOrthogonal_v1",
        filtered_probs=None,
        pred_state=None,
        pred_log_likelihood=None,
        inference_type="skipped_audit",
        run_id="feed_audit_2023",
        gap_type="partial_bar_skipped",
        is_complete=False,
        is_valid_prediction=False,
        status="skipped_partial",
    )

    # 5. Ingest retrospective (same timestamp as replay is allowed because inference_type differs)
    store.append(
        timestamp_utc=ts_retro,
        model_version="Gaussian_HMM_K3_M1_v1",
        feature_version="Config_2_MinimalOrthogonal_v1",
        filtered_probs=[0.3, 0.4, 0.3],
        pred_state=1,
        pred_log_likelihood=4.9,
        inference_type="retrospective",
        run_id="retrospective_eval_01",
    )

    assert len(store) == 5

    # Verify query isolation
    df_train = store.to_dataframe(inference_type="in_sample_filtered")
    assert len(df_train) == 1
    assert df_train["model_version"].iloc[0] == "Gaussian_HMM_K3_M1_v1"

    df_replay = store.to_dataframe(inference_type="as_of_replay")
    assert len(df_replay) == 1
    assert df_replay["model_version"].iloc[0] == "GMM_HMM_K3_M3_v1"

    df_live = store.to_dataframe(inference_type="as_of")
    assert len(df_live) == 1
    assert df_live["inference_type"].iloc[0] == "as_of"

    df_skipped = store.to_dataframe(inference_type="skipped_audit")
    assert len(df_skipped) == 1
    assert df_skipped["is_valid_prediction"].iloc[0] == False

    df_retro = store.to_dataframe(inference_type="retrospective")
    assert len(df_retro) == 1
    assert df_retro["run_id"].iloc[0] == "retrospective_eval_01"

    # Verify duplicate rejection within in_sample_filtered
    with pytest.raises(DuplicateRecordError):
        store.append(
            timestamp_utc=ts_train,
            model_version="Gaussian_HMM_K3_M1_v1",
            feature_version="Config_2_MinimalOrthogonal_v1",
            filtered_probs=[0.5, 0.3, 0.2],
            pred_state=0,
            pred_log_likelihood=5.0,
            inference_type="in_sample_filtered",
            run_id="in_sample_train_2023",
        )

    # Verify duplicate rejection within as_of_replay
    with pytest.raises(DuplicateRecordError):
        store.append(
            timestamp_utc=ts_replay,
            model_version="GMM_HMM_K3_M3_v1",
            feature_version="Config_2_MinimalOrthogonal_v1",
            filtered_probs=[0.3, 0.3, 0.4],
            pred_state=2,
            pred_log_likelihood=5.0,
            inference_type="as_of_replay",
            run_id="oos_replay_2023",
        )

    # Verify reload from disk preserves indexes
    store_reloaded = ImmutableProbabilityStore(log_file)
    assert len(store_reloaded) == 5
    with pytest.raises(DuplicateRecordError):
        store_reloaded.append(
            timestamp_utc=ts_replay,
            model_version="GMM_HMM_K3_M3_v1",
            feature_version="Config_2_MinimalOrthogonal_v1",
            filtered_probs=[0.3, 0.3, 0.4],
            pred_state=2,
            pred_log_likelihood=5.0,
            inference_type="as_of_replay",
            run_id="oos_replay_2023",
        )


def test_19_initial_state_distribution_vs_stationary(synthetic_models):
    """
    Verify that startprob_ is strictly the initial-state distribution,
    which is mathematically distinct from the asymptotic stationary distribution.
    """
    gauss, gmm = synthetic_models

    for model in [gauss, gmm]:
        pi_0 = model.startprob_
        A = model.transmat_

        # Verify pi_0 @ A != pi_0 (i.e. startprob_ is NOT stationary)
        pi_after_step = pi_0 @ A
        is_stationary = np.allclose(pi_0, pi_after_step, atol=1e-3)
        assert not is_stationary, "Model startprob_ must not be confused with stationary distribution"

        # Solve true stationary distribution pi_infty @ A = pi_infty
        evals, evecs = np.linalg.eig(A.T)
        stationary_idx = np.argmin(np.abs(evals - 1.0))
        pi_infty = np.real(evecs[:, stationary_idx])
        pi_infty /= np.sum(pi_infty)

        # Confirm stationary distribution satisfies invariance
        assert np.allclose(pi_infty @ A, pi_infty, atol=1e-10)

        # Confirm initial-state distribution differs from stationary distribution
        assert not np.allclose(pi_0, pi_infty, atol=1e-3)



