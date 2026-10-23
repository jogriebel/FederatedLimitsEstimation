"""
Simulation with known performance limits (S2 Appendix, S3 Fig).

Two Gaussian classes with a common covariance, P = N(0, S) for the negative and
Q = N(mu, S) for the positive entries, S_ij = 0.5^|i-j|, d = 8, and a mean shift
of Mahalanobis length Delta. Then, exactly,
    R(P, Q)   = Delta^2 / (4 ln 2) bits
    kappa_max = 1 - 2^(-R(P, Q))
and the Bayes-optimal kappa is the best kappa of a threshold on the
log-likelihood ratio, which is N(0, 1) for negatives and N(Delta, 1) for positives.

Each dataset is split across five clients with the patient and transfusion
counts of the 20% cohort (Table 1), and z-scored per client. The centralized
estimate applies compute_theoretical_kappa to the pooled client data, the
federated estimate applies Algorithm 1 (compute_statistics -> merge_payloads ->
_d_ensemble) to the client data. Both use the production code unchanged, with
the settings of the clinical analysis. Estimates flagged as floored by the
production code (cdr_floored) are treated as undefined.

Usage (from the repository root):
    python simulation_validation/simulate_known_limits_new.py              # full run
    python simulation_validation/simulate_known_limits_new.py --reps 2     # quick check
Output: simulation_validation/results/{simulation.csv, summary.csv, S3_Fig.*}
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from general_config import MAX_FEATURES, N_BINS, N_ITERATIONS, RANDOM_STATE
from kappa_limits import compute_theoretical_kappa
from federated.kappa.client import compute_statistics
from federated.kappa.server import _d_ensemble, merge_payloads

logging.getLogger("federated.kappa.server").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)

D = 8
CLIENTS = [(7302, 1461), (17810, 3562), (7345, 1469), (20035, 4007), (1415, 283)]
# (n, transfused) of eICU-CRD, MIMIC-IV, UMCdb, HiRID, UKA in the 20% cohort (Table 1)
PREVALENCE = sum(n1 for _, n1 in CLIENTS) / sum(n for n, _ in CLIENTS)   # 10,782 / 53,907
DELTAS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
DF, N_SEEDS, OVERSAMPLE = 4, 20, 1                  # as in the clinical analysis

SIGMA = 0.5 ** np.abs(np.subtract.outer(np.arange(D), np.arange(D)))
DIRECTION = (-1.0) ** np.arange(D)
DIRECTION /= np.sqrt(DIRECTION @ np.linalg.solve(SIGMA, DIRECTION))   # Mahalanobis length 1


def true_limits(delta):
    """R(P, Q) in bits, kappa_max and the Bayes-optimal kappa at prevalence 0.2."""
    rad = delta ** 2 / (4 * np.log(2))
    t = np.linspace(-10, 10 + delta, 20001)                # thresholds on the LLR score
    tpr, fpr = norm.sf(t - delta), norm.sf(t)
    q = PREVALENCE * tpr + (1 - PREVALENCE) * fpr           # predicted positive rate
    po = PREVALENCE * tpr + (1 - PREVALENCE) * (1 - fpr)
    pe = PREVALENCE * q + (1 - PREVALENCE) * (1 - q)
    return rad, 1 - 2 ** (-rad), float(np.max((po - pe) / (1 - pe)))


def simulate_clients(delta, rng):
    clients = []
    for n, n1 in CLIENTS:
        X = rng.multivariate_normal(np.zeros(D), SIGMA, size=n)
        X[:n1] += delta * DIRECTION
        y = np.r_[np.ones(n1), np.zeros(n - n1)]
        clients.append(((X - X.mean(0)) / X.std(0), y))      # z-scored per client
    return clients


def run(task):
    i, rep = task
    delta = DELTAS[i]
    np.random.seed([RANDOM_STATE, i, rep])                   # binning dither uses the global RNG
    clients = simulate_clients(delta, np.random.default_rng([RANDOM_STATE, i, rep]))

    cen = compute_theoretical_kappa(np.vstack([X for X, _ in clients]),
                                    np.concatenate([y for _, y in clients]),
                                    MAX_FEATURES, N_BINS, N_ITERATIONS)
    merged = merge_payloads([json.loads(compute_statistics(X, y)) for X, y in clients])
    fed = _d_ensemble(merged, DF, MAX_FEATURES, N_ITERATIONS, N_BINS, N_SEEDS, OVERSAMPLE)

    rad, kmax, kbayes = true_limits(delta)
    return dict(delta=delta, rep=rep, rad_true=rad, kmax_true=kmax, kappa_bayes=kbayes,
                cen_kappa=cen["theoretical_kappa"], cen_cdr=cen["CDR"],
                cen_floored=cen["cdr_floored"],
                fed_kappa=fed.get("kappa_max", np.nan), fed_cdr=fed.get("cdr", np.nan),
                fed_floored=(not fed["valid"]) or fed["full_results"]["cdr_floored"])


def summarize(df):
    df = df.copy()
    df.loc[df.cen_floored, "cen_kappa"] = np.nan
    df.loc[df.fed_floored, "fed_kappa"] = np.nan
    df["cw_bias"] = df.cen_kappa - df.kmax_true                  # Crow & Watts estimator bias
    df["fed_penalty"] = df.fed_kappa - df.cen_kappa              # federation penalty
    g = df.groupby("delta")
    out = g[["rad_true", "kmax_true", "kappa_bayes"]].first()
    for c in ["cen_kappa", "fed_kappa", "cw_bias", "fed_penalty"]:
        out[c] = g[c].mean()
        out[c + "_se"] = g[c].std() / np.sqrt(g[c].count())
    out["n_reps"] = g.size()
    out["n_cen_floored"] = g.cen_floored.sum()
    out["n_fed_floored"] = g.fed_floored.sum()
    return out.reset_index()


def plot(s, out):
    plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42})
    cen, fed = dict(color="#0072B2", marker="o"), dict(color="#D55E00", marker="s")
    x = s.kmax_true
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    a.plot([0, 1], [0, 1], "k--", lw=0.8, label=r"True $\kappa_{max}$")
    a.plot(x, s.kappa_bayes, color="0.5", lw=1.2, label=r"Bayes-optimal $\kappa$")
    a.errorbar(x, s.cen_kappa, s.cen_kappa_se, ls="none", ms=4, capsize=2, label="Centralized", **cen)
    a.errorbar(x, s.fed_kappa, s.fed_kappa_se, ls="none", ms=4, capsize=2, label="Federated", **fed)
    a.set(xlabel=r"True $\kappa_{max}$", ylabel=r"$\kappa$", xlim=(0, 1), ylim=(0, 1))

    b.axhline(0, color="k", lw=0.8)
    b.errorbar(x, s.cw_bias, s.cw_bias_se, ms=4, capsize=2, lw=1,
               label="(i) Crow & Watts bias\n(centralized $-$ true)", **cen)
    b.errorbar(x, s.fed_penalty, s.fed_penalty_se, ms=4, capsize=2, lw=1,
               label="(ii) Federation penalty\n(federated $-$ centralized)", **fed)
    b.set(xlabel=r"True $\kappa_{max}$", ylabel=r"Difference in $\kappa$", xlim=(0, 1))

    for ax, letter in ((a, "A"), (b, "B")):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        ax.text(-0.15, 1.02, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"S3_Fig.{ext}", dpi=300)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    Image.open(buf).convert("RGB").save(out / "S3_Fig.tif", compression="tiff_lzw", dpi=(300, 300))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    out = ROOT / "simulation_validation" / "results"
    out.mkdir(parents=True, exist_ok=True)

    tasks = [(i, r) for i in range(len(DELTAS)) for r in range(args.reps)]
    with ProcessPoolExecutor(args.jobs) as ex:
        df = pd.DataFrame(ex.map(run, tasks))
    df.to_csv(out / "simulation.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "summary.csv", index=False)
    print(s.round(3).to_string(index=False))
    plot(s, out)
