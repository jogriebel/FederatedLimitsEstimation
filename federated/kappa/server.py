"""
Federated Kappa Server — Method D.

Clients send additive sufficient statistics (histograms, Σx, XᵀX).
The server reconstructs a synthetic joint dataset via a Student-t copula and
computes kappa_max using compute_theoretical_kappa (Crow & Watts pipeline).

Usage (Docker / Flower):
    python federated/kappa/server.py --num_clients 5 --n_seeds 20
"""

import argparse
import csv
import json
import logging

import flwr as fl
import numpy as np
from collections import defaultdict
from flwr.common import Parameters
from scipy.linalg import LinAlgError, cholesky
from scipy.stats import t as student_t
from tqdm import tqdm

from general_config import FEDERATED_KAPPA_FILE, MAX_FEATURES, N_BINS, N_ITERATIONS, RESULTS_DIR
from kappa_limits import compute_theoretical_kappa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Merge Sufficient Statistics  — do not modify
# =============================================================================

def merge_payloads(payloads: list) -> dict:
    """Reconstruct exact global statistics from additive client payloads."""
    ref    = payloads[0]
    n_bins = ref["num_bins"]
    n_feat = max(p["num_features"] for p in payloads)
    merged = {
        "n_feat": n_feat, "n_bins": n_bins,
        "bin_lo": ref["bin_lo"], "bin_hi": ref["bin_hi"], "classes": {}
    }

    for label in ["0", "1"]:
        n    = 0
        hists = defaultdict(lambda: np.zeros(n_bins, dtype=np.float64))
        sums  = np.zeros(n_feat, dtype=np.float64)
        xtx   = np.zeros((n_feat, n_feat), dtype=np.float64)

        for p in payloads:
            cls = p.get("classes", {}).get(label)
            if not cls:
                continue
            n    += cls["n"]
            for j, c in cls["histograms"].items():
                hists[int(j)] += np.array(c, dtype=np.float64)
            sums += np.array(cls["feat_sums"], dtype=np.float64)
            xtx  += np.array(cls["xtx"],       dtype=np.float64)

        mean = sums / max(n, 1)
        cov  = xtx / max(n, 1) - np.outer(mean, mean)
        std  = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        corr = np.clip(cov / np.outer(std, std), -1, 1)
        np.fill_diagonal(corr, 1.0)

        merged["classes"][label] = {
            "n": n, "histograms": dict(hists),
            "mean": mean, "std": std, "corr": corr, "cov": cov,
        }

    return merged


# =============================================================================
# t-Copula Sampling  — do not modify
# =============================================================================

def _draw_class(cls: dict, n: int, nf: int, edges: np.ndarray,
                bw: float, df: int, rng: np.random.RandomState) -> np.ndarray:
    corr = cls["corr"].copy()
    ev, evec = np.linalg.eigh(corr)
    corr = evec @ np.diag(np.maximum(ev, 1e-6)) @ evec.T
    np.fill_diagonal(corr, 1.0)
    try:
        L = cholesky(corr, lower=True)
    except LinAlgError:
        L = np.eye(nf)

    Z = rng.standard_normal((n, nf))
    V = rng.chisquare(df, size=n)[:, None]
    U = student_t.cdf(Z / np.sqrt(V / df) @ L.T, df=df)

    X = np.zeros_like(U)
    for j in range(nf):
        cnts = cls["histograms"][j]
        tot  = cnts.sum()
        if tot == 0:
            continue
        cdf = np.cumsum(cnts / tot)
        cdf[-1] = 1.0
        idx = np.clip(np.searchsorted(cdf, U[:, j], side="right"), 0, len(cdf) - 1)
        X[:, j] = edges[idx] + rng.uniform(0, 1, n) * bw
    return X


def sample_t_copula(merged: dict, n0: int, n1: int, df: int = 4, seed: int = 42):
    """Generate synthetic data from a Student-t copula fit to merged statistics."""
    rng   = np.random.RandomState(seed)
    nf    = merged["n_feat"]
    edges = np.linspace(merged["bin_lo"], merged["bin_hi"], merged["n_bins"] + 1)
    bw    = edges[1] - edges[0]

    X0 = _draw_class(merged["classes"]["0"], n0, nf, edges, bw, df, rng)
    X1 = _draw_class(merged["classes"]["1"], n1, nf, edges, bw, df, rng)
    return np.vstack([X0, X1]), np.concatenate([np.zeros(n0), np.ones(n1)])


def generate_synthetic(merged: dict, df: int, seed: int, oversample: int,
                       min_per_class: int = 100):
    """Oversample synthetic data for numerically stable CDR estimation."""
    n0   = merged["classes"]["0"]["n"]
    n1   = merged["classes"]["1"]["n"]
    total = n0 + n1
    if total == 0:
        return None, None, 0, 0
    bal         = n1 / total
    synth_total = max(total * oversample, min_per_class * 2)
    nc0 = max(int(synth_total * (1 - bal)), min_per_class)
    nc1 = max(int(synth_total * bal),       min_per_class)
    X, y = sample_t_copula(merged, nc0, nc1, df=df, seed=seed)
    return X, y, nc0, nc1


# =============================================================================
# Method D: t-Copula → Crow & Watts
# =============================================================================

def run_method_d(X: np.ndarray, y: np.ndarray, max_features: int,
                 n_iterations: int, n_bins: int) -> dict:
    logger.info("Method D: t-Copula → Crow & Watts")
    cw    = compute_theoretical_kappa(X, y, max_features, n_bins, n_iterations)
    kappa = cw.get("theoretical_kappa", 0.0)
    cdr   = cw.get("CDR", 0.0)
    feats = cw.get("selected_features", [])
    logger.info(f"  kappa={kappa:.6f}, CDR={cdr:.6f}, features={feats}")
    return {"valid": True, "kappa_max": kappa, "cdr": cdr,
            "selected_features": feats, "full_results": cw}


def _d_ensemble(merged: dict, df: int, max_features: int, n_iterations: int,
                n_bins: int, n_seeds: int, oversample: int,
                min_per_class: int = 100) -> dict:
    """Run Method D across multiple copula seeds; return median result."""
    runs = []
    for s in tqdm(range(n_seeds)):
        X, y, nc0, nc1 = generate_synthetic(
            merged, df=df, seed=42 + s * 17,
            oversample=oversample, min_per_class=min_per_class)
        if X is None:
            continue
        if s == 0:
            logger.info(f"Ensemble: {n_seeds} seeds, {nc0}+{nc1} per seed, "
                        f"{X.shape[1]} features")
        rd = run_method_d(X, y, max_features, n_iterations, n_bins)
        if rd["valid"]:
            runs.append(rd)

    if not runs:
        return {"valid": False}

    kappas  = [r["kappa_max"] for r in runs]
    cdrs    = [r["cdr"]      for r in runs]
    med_idx = int(np.argsort(kappas)[len(kappas) // 2])
    result  = dict(runs[med_idx])
    result["kappa_max"]       = float(np.median(kappas))
    result["cdr"]             = float(np.mean(cdrs))
    result["ensemble_kappas"] = kappas
    result["ensemble_cdrs"]   = cdrs
    logger.info(f"  Ensemble κ: {[f'{k:.4f}' for k in kappas]} → median={result['kappa_max']:.6f}")
    return result


# =============================================================================
# Flower Strategy
# =============================================================================

class KappaStrategy(fl.server.strategy.FedAvg):
    def __init__(self, df, max_features, n_iterations, n_bins,
                 n_seeds, oversample, **kw):
        super().__init__(**kw)
        self.df           = df
        self.max_features = max_features
        self.n_iterations = n_iterations
        self.n_bins       = n_bins
        self.n_seeds      = n_seeds
        self.oversample   = oversample
        self.results      = []

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        logger.info(f"\n{'='*60}")
        logger.info(f"Round {server_round}: {len(results)} client(s)")
        logger.info(f"{'='*60}")

        payloads = [json.loads(fr.parameters.tensors[0].decode())
                    for _, fr in results if fr.parameters.tensors]
        if not payloads:
            return Parameters(tensor_type="", tensors=[]), {}

        merged = merge_payloads(payloads)
        rd = _d_ensemble(
            merged, df=self.df, max_features=self.max_features,
            n_iterations=self.n_iterations, n_bins=self.n_bins,
            n_seeds=self.n_seeds, oversample=self.oversample)

        self.results.append({"round": server_round, "method_d": rd})

        metrics = {}
        if rd.get("valid"):
            metrics["kappa_d"] = rd["kappa_max"]
            metrics["cdr_d"]   = rd["cdr"]
        return None, metrics


# =============================================================================
# Entry Point
# =============================================================================

def start_server(server_address, num_rounds, df, num_clients,
                 max_features, n_iterations, n_bins, n_seeds, oversample):

    strategy = KappaStrategy(
        df=df, max_features=max_features,
        n_iterations=n_iterations, n_bins=n_bins,
        n_seeds=n_seeds, oversample=oversample,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        min_evaluate_clients=0)

    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy)
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    finally:
        RESULTS_DIR.mkdir(exist_ok=True)

        def _row(rnd, res):
            eks = res.get("ensemble_kappas", [])
            return {
                "round":              rnd,
                "valid":              res.get("valid", False),
                "kappa_max":          res.get("kappa_max", ""),
                "cdr":                res.get("cdr", ""),
                "selected_features":  ";".join(str(f) for f in res.get("selected_features", [])),
                "ensemble_kappas":    ";".join(f"{k:.6f}" for k in eks),
                "ensemble_kappa_std": float(np.std(eks)) if len(eks) > 1 else "",
            }

        rows = [_row(r["round"], r["method_d"]) for r in strategy.results]
        if rows:
            csv_path = RESULTS_DIR / "federated_kappa_results.csv"
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Results saved → {csv_path}")

        # Write a single-value JSON consumed by all FL model clients.
        last = strategy.results[-1]["method_d"] if strategy.results else {}
        if last.get("valid"):
            with open(FEDERATED_KAPPA_FILE, "w") as fh:
                json.dump({"federated_kappa": last["kappa_max"],
                           "cdr":             last.get("cdr", 0.0)}, fh, indent=2)
            logger.info(f"Federated kappa saved → {FEDERATED_KAPPA_FILE} "
                        f"(κ={last['kappa_max']:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server_address", default="0.0.0.0:8095")
    p.add_argument("--num_rounds",    type=int, default=1)
    p.add_argument("--df",            type=int, default=4)
    p.add_argument("--num_clients",   type=int, default=5)
    p.add_argument("--max_features",  type=int, default=MAX_FEATURES)
    p.add_argument("--n_iterations",  type=int, default=N_ITERATIONS)
    p.add_argument("--n_bins",        type=int, default=N_BINS)
    p.add_argument("--n_seeds",       type=int, default=20)
    p.add_argument("--oversample",    type=int, default=1)
    args = p.parse_args()
    start_server(**vars(args))
