"""
src.models.gmm_hmm
~~~~~~~~~~~~~~~~~~
Multivariate Gaussian Mixture Hidden Markov Model (GMM-HMM) implementation
using Baum-Welch (EM) in log-space with multi-sequence gap support,
multiple random restarts, and feature-scaled covariance regularization.

Mathematical Formulation:
-------------------------
For state k in {1..K}, the emission distribution is a mixture of M Gaussians:
    b_k(x_t) = sum_{m=1}^M w_{km} * N(x_t | mu_{km}, Sigma_{km})
In log-space:
    log b_k(x_t) = LogSumExp_m [ log w_{km} + log N(x_t | mu_{km}, Sigma_{km}) ]

Key Guarantees:
- Pure NumPy/SciPy implementation without C compiler dependencies.
- Strict numerical stability via fast vectorized logsumexp.
- Multi-sequence gap awareness: avoids computing transitions across non-contiguous intervals.
- Feature-scaled covariance regularizer floor prevents degenerate singular states.
- Deterministic reproducibility through explicit random state management.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from src.models.gaussian_hmm import (
    _fast_logsumexp_axis0,
    _fast_logsumexp_axis1,
    _fast_logsumexp_1d,
)

logger = logging.getLogger(__name__)


class GMMHMM:
    """
    Gaussian Mixture Hidden Markov Model with full or diagonal covariance.

    Parameters
    ----------
    n_components : int
        Number of hidden states (K).
    n_mix : int
        Number of Gaussian mixture components per state (M).
    covariance_type : str
        'full' for unrestricted symmetric covariance, or 'diag' for diagonal.
    n_iter : int
        Maximum number of EM iterations per restart (default: 100).
    tol : float
        Convergence threshold on log-likelihood increase (default: 1e-4).
    cov_reg : float
        Feature-scaled covariance regularization floor multiplier (default: 1e-3).
    random_state : Optional[int]
        Seed for reproducible initialization.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_mix: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        cov_reg: float = 1e-3,
        random_state: Optional[int] = 42,
    ):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if n_mix < 1:
            raise ValueError(f"n_mix must be >= 1, got {n_mix}")
        if covariance_type not in ("full", "diag"):
            raise ValueError(f"covariance_type must be 'full' or 'diag', got {covariance_type}")

        self.n_components = n_components
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.cov_reg = cov_reg
        self.random_state = random_state

        # Model parameters (populated after fit)
        self.startprob_: Optional[np.ndarray] = None       # (K,)
        self.transmat_: Optional[np.ndarray] = None        # (K, K)
        self.weights_: Optional[np.ndarray] = None         # (K, M)
        self.means_: Optional[np.ndarray] = None           # (K, M, D)
        self.covars_: Optional[np.ndarray] = None          # (K, M, D, D) or (K, M, D)
        self.n_features_: Optional[int] = None

        # Convergence diagnostics
        self.converged_: bool = False
        self.n_iter_done_: int = 0
        self.score_history_: List[float] = []
        self.best_log_likelihood_: float = -np.inf

    def _init_params(self, X: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize parameters using hierarchical K-Means clustering and empirical covariances."""
        K = self.n_components
        M = self.n_mix
        N, D = X.shape
        self.n_features_ = D

        # 1. Initial state distribution: uniform with minor perturbation
        startprob = np.ones(K) / K + rng.uniform(0, 0.01, size=K)
        self.startprob_ = startprob / startprob.sum()

        # 2. Transition matrix: diagonal persistence prior (e.g. 0.7 on diagonal)
        transmat = np.eye(K) * 0.7 + rng.uniform(0.01, 0.05, size=(K, K))
        self.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)

        # 3. Mixture weights: uniform with minor perturbation per state
        weights = np.ones((K, M)) / M + rng.uniform(0, 0.01, size=(K, M))
        self.weights_ = weights / weights.sum(axis=1, keepdims=True)

        # 4. Covariance regularizer floor based on global variance
        global_cov = np.cov(X, rowvar=False) if N > 1 else np.eye(D)
        if D == 1:
            global_cov = np.array([[float(global_cov)]])

        f_var = np.diag(global_cov) if global_cov.ndim == 2 else np.ones(D) * float(global_cov)
        self._cov_reg_diag = np.maximum(f_var, 1e-12) * self.cov_reg
        self._cov_reg_mat = np.diag(self._cov_reg_diag)

        # 5. Hierarchical K-Means initialization for Means
        self.means_ = np.zeros((K, M, D), dtype=np.float64)
        if self.covariance_type == "full":
            self.covars_ = np.zeros((K, M, D, D), dtype=np.float64)
        else:
            self.covars_ = np.zeros((K, M, D), dtype=np.float64)

        if N >= K * M:
            try:
                # First level: K clusters
                kmeans_k = KMeans(n_clusters=K, random_state=rng.randint(0, 100000), n_init=3, max_iter=30)
                state_labels = kmeans_k.fit_predict(X)
                state_centers = kmeans_k.cluster_centers_

                for k in range(K):
                    X_k = X[state_labels == k]
                    if len(X_k) >= M:
                        kmeans_m = KMeans(n_clusters=M, random_state=rng.randint(0, 100000), n_init=2, max_iter=20)
                        kmeans_m.fit(X_k)
                        self.means_[k] = kmeans_m.cluster_centers_.copy()
                    else:
                        # Fallback perturbation around state center
                        for m in range(M):
                            noise = rng.normal(0, 0.05, size=D) * np.sqrt(self._cov_reg_diag)
                            self.means_[k, m] = state_centers[k] + noise
            except Exception:
                indices = rng.choice(N, size=K * M, replace=False)
                self.means_ = X[indices].reshape(K, M, D).copy()
        else:
            self.means_ = rng.normal(0, 1, size=(K, M, D))

        # 6. Initialize covariances
        for k in range(K):
            for m in range(M):
                if self.covariance_type == "full":
                    self.covars_[k, m] = 0.5 * (global_cov + global_cov.T) + self._cov_reg_mat
                else:
                    self.covars_[k, m] = f_var + self._cov_reg_diag

    def _compute_component_log_emissions(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log phi_{km}(x_t) = log N(x_t | mu_{km}, Sigma_{km}).

        Returns
        -------
        log_phi : np.ndarray, shape (N, K, M)
        """
        N, D = X.shape
        K = self.n_components
        M = self.n_mix
        log_phi = np.empty((N, K, M), dtype=np.float64)

        for k in range(K):
            for m in range(M):
                mean = self.means_[k, m]
                if self.covariance_type == "full":
                    cov = 0.5 * (self.covars_[k, m] + self.covars_[k, m].T)
                    try:
                        mvn = multivariate_normal(mean=mean, cov=cov, allow_singular=False)
                        log_phi[:, k, m] = mvn.logpdf(X)
                    except Exception:
                        cov_safe = cov + np.eye(D) * 1e-9
                        mvn = multivariate_normal(mean=mean, cov=cov_safe, allow_singular=True)
                        log_phi[:, k, m] = mvn.logpdf(X)
                else:
                    cov_diag = np.maximum(self.covars_[k, m], 1e-12)
                    mvn = multivariate_normal(mean=mean, cov=np.diag(cov_diag), allow_singular=True)
                    log_phi[:, k, m] = mvn.logpdf(X)

        return np.nan_to_num(log_phi, nan=-1e12, neginf=-1e12)

    def _compute_log_emissions(
        self,
        X: np.ndarray,
        log_phi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute log b_k(x_t) = logsumexp_m [ log w_{km} + log phi_{km}(x_t) ].

        Returns
        -------
        log_b : np.ndarray, shape (N, K)
        """
        if log_phi is None:
            log_phi = self._compute_component_log_emissions(X)

        N, K, M = log_phi.shape
        log_w = np.log(np.maximum(self.weights_, 1e-12))  # (K, M)

        # weighted_log_phi: shape (N, K, M)
        weighted_log_phi = log_phi + log_w[None, :, :]

        # LogSumExp over mixture axis M (axis=2)
        mx = np.max(weighted_log_phi, axis=2, keepdims=True)  # (N, K, 1)
        sum_exp = np.sum(np.exp(weighted_log_phi - mx), axis=2)  # (N, K)
        log_b = np.squeeze(mx, axis=2) + np.log(np.maximum(sum_exp, 1e-300))

        return np.nan_to_num(log_b, nan=-1e12, neginf=-1e12)

    def _em_fit_single(
        self,
        X: np.ndarray,
        lengths: List[int],
        rng: np.random.RandomState,
    ) -> Tuple[float, List[float], bool]:
        """Run Baum-Welch EM algorithm for a single GMM-HMM initialization."""
        self._init_params(X, rng)
        K = self.n_components
        M = self.n_mix
        N, D = X.shape

        is_start = np.zeros(N, dtype=bool)
        is_end = np.zeros(N, dtype=bool)
        seq_id = np.empty(N, dtype=np.int32)
        curr = 0
        for idx, l in enumerate(lengths):
            is_start[curr] = True
            is_end[curr + l - 1] = True
            seq_id[curr : curr + l] = idx
            curr += l

        n_seqs = len(lengths)
        score_history = []
        converged = False

        log_alpha = np.empty((N, K), dtype=np.float64)
        log_beta = np.empty((N, K), dtype=np.float64)

        for it in range(self.n_iter):
            log_phi = self._compute_component_log_emissions(X)
            log_b = self._compute_log_emissions(X, log_phi=log_phi)
            log_startprob = np.log(np.maximum(self.startprob_, 1e-12))
            log_transmat = np.log(np.maximum(self.transmat_, 1e-12))
            log_weights = np.log(np.maximum(self.weights_, 1e-12))

            # 1. Forward Pass
            seq_lls = np.empty(n_seqs, dtype=np.float64)
            for t in range(N):
                if is_start[t]:
                    log_alpha[t] = log_startprob + log_b[t]
                else:
                    m_vec = log_alpha[t - 1, :, None] + log_transmat
                    log_alpha[t] = log_b[t] + _fast_logsumexp_axis0(m_vec)
                if is_end[t]:
                    seq_lls[seq_id[t]] = _fast_logsumexp_1d(log_alpha[t])

            total_log_likelihood = float(np.sum(seq_lls))
            score_history.append(total_log_likelihood)

            # Check convergence
            if it > 0:
                change = total_log_likelihood - score_history[-2]
                if 0 <= change < self.tol:
                    converged = True
                    break

            # 2. Backward Pass
            for t in range(N - 1, -1, -1):
                if is_end[t]:
                    log_beta[t] = 0.0
                else:
                    m_vec = log_transmat + log_b[t + 1, None, :] + log_beta[t + 1, None, :]
                    log_beta[t] = _fast_logsumexp_axis1(m_vec)

            # 3. State Responsibilities gamma_t(k)
            seq_ll_per_t = seq_lls[seq_id, None]
            log_gamma = log_alpha + log_beta - seq_ll_per_t
            gamma = np.exp(np.clip(log_gamma, -700, 0))
            gamma_row_sum = gamma.sum(axis=1, keepdims=True)
            gamma /= np.maximum(gamma_row_sum, 1e-12)  # (N, K)

            # 4. Mixture Component Responsibilities gamma_tkm = P(S_t=k, C_t=m | X)
            # log gamma_tkm = log gamma_tk + log w_{km} + log phi_{km}(x_t) - log b_k(x_t)
            log_gamma_km = (
                log_gamma[:, :, None]
                + log_weights[None, :, :]
                + log_phi
                - log_b[:, :, None]
            )
            gamma_km = np.exp(np.clip(log_gamma_km, -700, 0))  # (N, K, M)
            # Normalize along mixture axis so sum_m gamma_tkm == gamma_tk
            gamma_km_sum_m = gamma_km.sum(axis=2, keepdims=True)
            gamma_km = np.where(gamma_km_sum_m > 1e-12, gamma_km / gamma_km_sum_m * gamma[:, :, None], 0.0)

            # 5. Transition statistics xi[t, i, j]
            non_end_mask = ~is_end
            t_indices = np.where(non_end_mask)[0]
            if len(t_indices) > 0:
                t_plus_1 = t_indices + 1
                seq_ll_non_end = seq_lls[seq_id[t_indices], None, None]
                log_xi = (
                    log_alpha[t_indices, :, None]
                    + log_transmat[None, :, :]
                    + log_b[t_plus_1, None, :]
                    + log_beta[t_plus_1, None, :]
                    - seq_ll_non_end
                )
                xi = np.exp(np.clip(log_xi, -700, 0))
                xi_accum = xi.sum(axis=0)
            else:
                xi_accum = np.zeros((K, K), dtype=np.float64)

            # M-step: Parameter updates
            # 1. Update startprob
            start_gamma = gamma[is_start].sum(axis=0) + 1e-6
            self.startprob_ = start_gamma / start_gamma.sum()

            # 2. Update transmat
            xi_row_sum = xi_accum.sum(axis=1, keepdims=True)
            xi_row_sum = np.where(xi_row_sum == 0, 1.0, xi_row_sum)
            new_transmat = (xi_accum / xi_row_sum) + 1e-6
            self.transmat_ = new_transmat / new_transmat.sum(axis=1, keepdims=True)

            # 3. Update mixture weights w_{km}
            gamma_sum_k = gamma.sum(axis=0)  # (K,)
            gamma_km_sum = gamma_km.sum(axis=0)  # (K, M)
            new_weights = (gamma_km_sum + 1e-4) / (gamma_sum_k[:, None] + M * 1e-4)
            self.weights_ = new_weights / new_weights.sum(axis=1, keepdims=True)

            # 4. Update mixture means mu_{km} and covariances Sigma_{km}
            cov_reg_mat = getattr(self, "_cov_reg_mat", np.eye(D) * self.cov_reg)
            cov_reg_diag = getattr(self, "_cov_reg_diag", np.ones(D) * self.cov_reg)

            for k in range(K):
                for m in range(M):
                    denom_km = max(gamma_km_sum[k, m], 1e-6)
                    self.means_[k, m] = (gamma_km[:, k, m, None] * X).sum(axis=0) / denom_km

                    diff = X - self.means_[k, m]
                    if self.covariance_type == "full":
                        weighted_diff = diff * gamma_km[:, k, m, None]
                        cov_km = (weighted_diff.T @ diff) / denom_km
                        self.covars_[k, m] = 0.5 * (cov_km + cov_km.T) + cov_reg_mat
                    else:
                        var_km = (gamma_km[:, k, m, None] * (diff ** 2)).sum(axis=0) / denom_km
                        self.covars_[k, m] = var_km + cov_reg_diag

        # Recalculate final log-likelihood for updated parameters
        final_ll = self.score(X, lengths)
        if len(score_history) == 0 or abs(final_ll - score_history[-1]) > 1e-9:
            score_history.append(final_ll)
        total_log_likelihood = final_ll

        return total_log_likelihood, score_history, converged

    def fit(
        self,
        X: Union[np.ndarray, List[float]],
        lengths: Optional[List[int]] = None,
        n_restarts: int = 5,
    ) -> "GMMHMM":
        """
        Fit GMM-HMM parameters using Baum-Welch EM with multiple random restarts.

        Selects best restart strictly according to training log-likelihood.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Observation matrix.
        lengths : Optional[List[int]]
            Lengths of continuous sequences in X.
        n_restarts : int
            Number of random initializations.

        Returns
        -------
        self : GMMHMM
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        N, D = X_arr.shape

        if lengths is None:
            lengths = [N]
        elif sum(lengths) != N:
            raise ValueError(f"Sum of lengths ({sum(lengths)}) != rows in X ({N})")

        base_seed = self.random_state if self.random_state is not None else 42
        best_ll = -np.inf
        best_params = None
        best_history = []
        best_converged = False

        for r in range(n_restarts):
            restart_rng = np.random.RandomState(base_seed + r * 107)
            try:
                ll, history, conv = self._em_fit_single(X_arr, lengths, restart_rng)
                if np.isfinite(ll) and ll > best_ll:
                    best_ll = ll
                    best_converged = conv
                    best_history = history
                    best_params = {
                        "startprob": self.startprob_.copy(),
                        "transmat": self.transmat_.copy(),
                        "weights": self.weights_.copy(),
                        "means": self.means_.copy(),
                        "covars": self.covars_.copy(),
                    }
            except Exception as e:
                logger.warning(f"GMM-HMM restart {r+1}/{n_restarts} exception: {e}")

        if best_params is None:
            raise RuntimeError(
                f"GMM-HMM failed to fit across all {n_restarts} random restarts."
            )

        self.startprob_ = best_params["startprob"]
        self.transmat_ = best_params["transmat"]
        self.weights_ = best_params["weights"]
        self.means_ = best_params["means"]
        self.covars_ = best_params["covars"]
        self.best_log_likelihood_ = best_ll
        self.score_history_ = best_history
        self.converged_ = best_converged
        self.n_iter_done_ = len(best_history)

        logger.info(
            f"GMM-HMM (K={self.n_components}, M={self.n_mix}, cov={self.covariance_type}) fitted: "
            f"LL={best_ll:.2f}, Converged={best_converged}, Iterations={self.n_iter_done_}"
        )
        return self

    def score(
        self,
        X: Union[np.ndarray, List[float]],
        lengths: Optional[List[int]] = None,
    ) -> float:
        """
        Compute total log-likelihood of observation sequence(s) under current model parameters.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
        lengths : Optional[List[int]]

        Returns
        -------
        total_ll : float
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        N, D = X_arr.shape

        if lengths is None:
            lengths = [N]

        log_b = self._compute_log_emissions(X_arr)
        log_startprob = np.log(np.maximum(self.startprob_, 1e-12))
        log_transmat = np.log(np.maximum(self.transmat_, 1e-12))

        is_start = np.zeros(N, dtype=bool)
        is_end = np.zeros(N, dtype=bool)
        curr = 0
        for l in lengths:
            is_start[curr] = True
            is_end[curr + l - 1] = True
            curr += l

        log_alpha = np.empty((N, self.n_components), dtype=np.float64)
        total_ll = 0.0

        for t in range(N):
            if is_start[t]:
                log_alpha[t] = log_startprob + log_b[t]
            else:
                m_vec = log_alpha[t - 1, :, None] + log_transmat
                log_alpha[t] = log_b[t] + _fast_logsumexp_axis0(m_vec)
            if is_end[t]:
                total_ll += _fast_logsumexp_1d(log_alpha[t])

        return float(total_ll)

    def to_dict(self) -> Dict:
        """Export model parameters to a JSON-serializable dictionary."""
        return {
            "model_type": "GMMHMM",
            "n_components": int(self.n_components),
            "n_mix": int(self.n_mix),
            "covariance_type": self.covariance_type,
            "n_features": int(self.n_features_) if self.n_features_ is not None else None,
            "cov_reg": float(self.cov_reg),
            "converged": bool(self.converged_),
            "n_iter_done": int(self.n_iter_done_),
            "best_log_likelihood": float(self.best_log_likelihood_),
            "startprob": self.startprob_.tolist() if self.startprob_ is not None else None,
            "transmat": self.transmat_.tolist() if self.transmat_ is not None else None,
            "weights": self.weights_.tolist() if self.weights_ is not None else None,
            "means": self.means_.tolist() if self.means_ is not None else None,
            "covars": self.covars_.tolist() if self.covars_ is not None else None,
            "cov_reg_diag": self._cov_reg_diag.tolist() if hasattr(self, "_cov_reg_diag") else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GMMHMM":
        """Instantiate a GMMHMM from an exported parameter dictionary."""
        model = cls(
            n_components=data["n_components"],
            n_mix=data["n_mix"],
            covariance_type=data["covariance_type"],
            cov_reg=data["cov_reg"],
        )
        model.n_features_ = data.get("n_features")
        model.converged_ = data.get("converged", False)
        model.n_iter_done_ = data.get("n_iter_done", 0)
        model.best_log_likelihood_ = data.get("best_log_likelihood", -np.inf)
        if data.get("startprob") is not None:
            model.startprob_ = np.array(data["startprob"], dtype=np.float64)
        if data.get("transmat") is not None:
            model.transmat_ = np.array(data["transmat"], dtype=np.float64)
        if data.get("weights") is not None:
            model.weights_ = np.array(data["weights"], dtype=np.float64)
        if data.get("means") is not None:
            model.means_ = np.array(data["means"], dtype=np.float64)
        if data.get("covars") is not None:
            model.covars_ = np.array(data["covars"], dtype=np.float64)
        if data.get("cov_reg_diag") is not None:
            model._cov_reg_diag = np.array(data["cov_reg_diag"], dtype=np.float64)
            model._cov_reg_mat = np.diag(model._cov_reg_diag)
        return model
