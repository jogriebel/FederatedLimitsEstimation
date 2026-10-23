"""
Theoretical performance limit estimation via Crow & Watts.

Computes kappa_max — the theoretical upper bound on Cohen's Kappa achievable
by any binary classifier — using a k-nearest-neighbour estimator of the
Kullback-Leibler divergence between class-conditional distributions, combined
with greedy feature selection and Fisher's discriminant as a fallback.

Key design decisions, grounded in Crow & Watts (arXiv:2403.01571):

  - CDR is averaged over multiple stochastic binning realisations to reduce
    variance. The paper (Section 3.3) states: "the CDI can be estimated
    multiple times and then averaged. This reduces the error and provides a
    lower bound on the error for one computation."

  - kappa_max is computed as 1 - 2^(-mean(CDR)), NOT mean(1 - 2^(-CDR_i)).
    Because kappa = 1 - 2^(-CDR) is concave, Jensen's inequality means
    averaging kappas instead of CDRs introduces a systematic downward bias.
    Computing kappa from the mean CDR is both correct and avoids this.

  - Greedy selection bins every feature into 32 percentile bins; the final
    CDR uses n_bins (N_BINS = 16).

  - Dither is uniform(0, 1) — non-negative, non-overlapping bin values.
  - Greedy stopping threshold is 1e-6 (tight, matching working script).
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

from general_config import MAX_FEATURES, N_BINS, N_ITERATIONS


# =============================================================================
# Actual Kappa
# =============================================================================

def compute_actual_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's Kappa from hard predictions."""
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    n  = tn + fp + fn + tp
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


# =============================================================================
# KL Divergence and CDR 
# =============================================================================

# Guard applied to each one-sided kNN KL estimate before the resistor average.
#
# Why it exists: the Wang/Perez-Cruz kNN KL estimator is not guaranteed to be
# non-negative. For two nearly identical class-conditional samples it can
# return zero or a small negative value, and the resistor average
#     CDR = (KL01 * KL10) / (KL01 + KL10)
# would then be negative or divide by (near-)zero. Flooring each direction
# keeps CDR positive and finite.
#
# This is a sign/zero guard, NOT log-of-zero protection: the logarithm is
# already protected by eps = 1e-12 inside _kl_divergence_knn. KL_FLOOR is
# therefore many orders of magnitude above machine precision and it is visible
# in the reported numbers: with one direction floored and the other large the
# resistor average saturates just below KL_FLOOR, so the smallest CDR this
# estimator can report is ~0.001 bits and the smallest kappa_max is
# 1 - 2**(-0.001) ~= 0.000693 rather than 0.
#
# Because the guard cannot be removed without reintroducing the division by
# zero, every estimate that touches it is flagged instead (`cdr_floored`,
# `cdr_floored_frac` in the compute_theoretical_kappa result), so that floored
# cells are identifiable downstream rather than being read as genuine small
# discriminability.
KL_FLOOR = 0.001


def _kl_divergence_knn(X_src: np.ndarray, X_tgt: np.ndarray, d: int) -> float:
    """KL divergence via kNN estimator """
    n_src, n_tgt = len(X_src), len(X_tgt)
    if n_src < 2 or n_tgt < 1:
        return KL_FLOOR

    dist_within  = NearestNeighbors(n_neighbors=2).fit(X_src).kneighbors(X_src)[0][:, 1]
    dist_between = NearestNeighbors(n_neighbors=1).fit(X_tgt).kneighbors(X_src)[0][:, 0]

    eps        = 1e-12
    log_ratios = np.log2(np.maximum(dist_between, eps) / np.maximum(dist_within, eps))
    return (d / n_src) * log_ratios.sum() + np.log2(n_tgt / (n_src - 1))


def _crow_watts_binning(data: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Crow & Watts percentile binning with uniform dither in [0, 1).

    Dither in [0, 1) keeps all bin values non-negative and non-overlapping,
    preserving the kNN distance geometry CDR depends on.
    """
    n, p   = data.shape
    binned = np.zeros_like(data)

    for j in range(p):
        col   = data[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            binned[:, j] = col
            continue
        edges = np.unique(np.percentile(col[valid], np.linspace(0, 100, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([col[valid].min() - 1, col[valid].max() + 1])
        idx               = np.clip(np.digitize(col, edges) - 1, 0, len(edges) - 2)
        binned[:, j]      = idx + np.random.uniform(0, 1, n)
        binned[~valid, j] = np.nan

    return binned


def _compute_cdr(X: np.ndarray, y: np.ndarray) -> tuple:
    """Symmetric Class Discriminability Rate (Resistor Average Distance estimate).

    Returns (cdr, floored). `floored` is True when either one-sided KL estimate
    was raised to KL_FLOOR, i.e. when the returned CDR is a guard artefact
    rather than a measurement. The CDR value itself is unchanged by the
    addition of the flag.
    """
    c0, c1 = X[y == 0], X[y == 1]
    if len(c0) < 2 or len(c1) < 2:
        return KL_FLOOR, True
    d      = X.shape[1]
    raw_01 = _kl_divergence_knn(c0, c1, d)
    raw_10 = _kl_divergence_knn(c1, c0, d)
    kl_01  = max(raw_01, KL_FLOOR)
    kl_10  = max(raw_10, KL_FLOOR)
    floored = bool(raw_01 <= KL_FLOOR or raw_10 <= KL_FLOOR)
    return (kl_01 * kl_10) / (kl_01 + kl_10), floored


def _cdr_with_binning(X: np.ndarray, y: np.ndarray, n_bins: int, n_iter: int) -> float:
    """CDR averaged over n_iter stochastic binning realisations."""
    if len(np.unique(y)) < 2:
        return KL_FLOOR
    cdrs = []
    for _ in range(n_iter):
        binned = _crow_watts_binning(X, n_bins)
        mask   = ~np.isnan(binned).any(axis=1)
        if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
            continue
        cdrs.append(_compute_cdr(binned[mask], y[mask])[0])
    return np.mean(cdrs) if cdrs else KL_FLOOR


# =============================================================================
# Feature Selection
# =============================================================================

def _auto_n_bins(n_samples: int) -> int:
    """
    Data-adaptive bin count: max(2, min(sqrt(n), n//4)).
    Matches crow_watts_binning(data, n_bins=None) in the working kaggle script.
    """
    return max(2, min(int(np.sqrt(n_samples)), n_samples // 4))


def _greedy_feature_selection(
    X: np.ndarray, y: np.ndarray, max_features: int, n_iter: int
) -> list:
    """
    Greedy forward feature selection maximising CDR.

    Uses auto-computed n_bins during selection (mirrors n_bins=None in the
    working script). Falls back to Fisher's discriminant to guarantee
    max_features are returned when greedy stops early.

    When n_features <= max_features all features are returned directly.
    """
    n_samples, n_features = X.shape

    if n_features <= max_features:
        return list(range(n_features))

    n_bins_sel   = 32 #_auto_n_bins(n_samples)
    # print("=====================",n_bins_sel,"======================================")
    search_limit = n_features

    single_cdrs = sorted(
        [(i, _cdr_with_binning(X[:, [i]], y, n_bins_sel, n_iter))
         for i in range(search_limit)],
        key=lambda x: x[1], reverse=True
    )
    selected    = [single_cdrs[0][0]]
    current_cdr = single_cdrs[0][1]
    remaining   = [i for i, _ in single_cdrs[1:n_features]]

    while remaining and len(selected) < max_features:
        best_idx, best_cdr = None, current_cdr
        for feat in remaining:
            cdr = _cdr_with_binning(X[:, selected + [feat]], y, n_bins_sel, n_iter)
            if cdr > best_cdr:
                best_cdr, best_idx = cdr, feat
        if best_idx is None or best_cdr <= current_cdr + 1e-6:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_cdr = best_cdr

    # Fisher's discriminant fallback
    remaining = list(set(range(n_features)) - set(selected))
    X_rem     = X[:, remaining] if isinstance(X, np.ndarray) else X.iloc[:, remaining].values

    fisher_scores = []
    for i in range(len(remaining)):
        vals = X_rem[:, i]
        v0, v1 = vals[y == 0], vals[y == 1]
        between = (np.mean(v0) - np.mean(v1)) ** 2
        within  = ((np.var(v0, ddof=1) if len(v0) > 1 else 0) +
                   (np.var(v1, ddof=1) if len(v1) > 1 else 0))
        fisher_scores.append(between / within if within > 1e-10 else 0)

    order = np.argsort(-np.array(fisher_scores))
    for i in range(min(max_features - len(selected), len(order))):
        feat_idx = remaining[order[i]]
        selected.append(feat_idx)
        print(f"  Fisher fallback: added feature {feat_idx} "
              f"(score={fisher_scores[order[i]]:.4f})")

    return selected


# =============================================================================
# Public API
# =============================================================================

def compute_theoretical_kappa(
    X_data,
    y_data,
    max_features=MAX_FEATURES,
    n_bins=N_BINS,
    n_iterations: int = N_ITERATIONS,
) -> dict:
    """
    Compute the theoretical upper bound on Cohen's Kappa (kappa_max).

    Following Crow & Watts (2024):
      1. Greedy feature selection with 32 percentile bins, CDR averaged over
         n_iterations realisations.
      2. Final CDR averaged over n_iterations binning passes at n_bins.
      3. kappa_max = 1 - 2^(-mean_CDR).

    Computing kappa from the mean CDR (rather than averaging kappas) avoids
    the systematic downward bias introduced by Jensen's inequality, since
    kappa = 1 - 2^(-CDR) is a concave function.

    Parameters
    ----------
    X_data : array-like of shape (n_samples, n_features)
        Pre-scaled feature matrix.
    y_data : array-like of shape (n_samples,)
        Binary labels {0, 1}.
    max_features : int
        Maximum features to include in CDR computation.
    n_bins : int
        Bins for the final CDR computation (default 16).
    n_iterations : int
        Stochastic binning realisations (default N_ITERATIONS).

    Returns
    -------
    dict with keys:
        theoretical_kappa : float  — kappa_max = 1 - 2^(-mean_CDR)
        kappa_std         : float  — std of per-iteration kappas (diagnostic)
        CDR               : float  — mean CDR across iterations
        CDR_std           : float  — std of CDR (estimate reliability)
        selected_features : list
        kappas            : list   — per-iteration kappa values (diagnostic)
    """
    X = np.asarray(X_data)
    y = np.asarray(y_data).ravel()

    selected = _greedy_feature_selection(X, y, max_features, n_iterations)
    X_sel    = X[:, selected]

    # Accumulate CDR across iterations, then convert mean CDR to kappa once.
    # This avoids the Jensen downward bias from averaging concave kappa values.
    cdrs = []
    floored_flags = []
    for _ in range(n_iterations):
        binned  = _crow_watts_binning(X_sel, n_bins)
        mask    = ~np.isnan(binned).any(axis=1)
        X_clean = binned[mask]
        y_clean = y[mask]
        if len(np.unique(y_clean)) < 2 or len(X_clean) < 4:
            continue
        cdr_i, floored_i = _compute_cdr(X_clean, y_clean)
        cdrs.append(cdr_i)
        floored_flags.append(floored_i)

    if not cdrs:
        return {"theoretical_kappa": 0.0, "kappa_std": 0.0,
                "CDR": 0.0, "CDR_std": 0.0,
                "cdr_floored": True, "cdr_floored_frac": 1.0,
                "selected_features": selected, "kappas": []}

    mean_cdr = float(np.mean(cdrs))
    kappa    = 1 - 2 ** (-mean_cdr) if mean_cdr > 0 else 0.0

    # Per-iteration kappas are kept as a diagnostic only
    kappas = [1 - 2 ** (-c) if c > 0 else 0.0 for c in cdrs]

    return {
        "theoretical_kappa": kappa,
        "kappa_std":         float(np.std(kappas)),
        "CDR":               mean_cdr,
        "CDR_std":           float(np.std(cdrs)),
        # Guard diagnostics: cdr_floored is True when any binning iteration hit
        # KL_FLOOR, so the reported CDR/kappa is at least partly a guard
        # artefact; cdr_floored_frac is the fraction of iterations affected.
        "cdr_floored":       bool(any(floored_flags)),
        "cdr_floored_frac":  float(np.mean(floored_flags)),
        "selected_features": selected,
        "kappas":            kappas,
    }