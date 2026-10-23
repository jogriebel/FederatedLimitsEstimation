"""
kappa_max: direct (real data) vs federated (t-copula).

1. Point estimates (always). The released estimators (compute_theoretical_kappa;
   client.compute_statistics -> merge_payloads -> _d_ensemble) run ONCE per
   scope on the full data, including greedy feature selection. Estimates and
   selected features are written to kappa_point.csv (reused if present, so a
   second run costs nothing).
2. Half-sampling (on by default, --n_rep 0 for point estimates only). In each
   replicate, n_c // 2 patients are drawn WITHOUT replacement within every
   (client, class) stratum. The SAME half-sample feeds the direct and the
   federated estimator, and the pooled scope is the union of the client
   half-samples. Feature selection is NOT repeated: only the features selected
   in step 1 are passed (m features with max_features = m), so the released
   estimators run unchanged and bypass the greedy search.
   Replicates run in chunks until --n_rep or the --hours budget is reached and
   are appended to kappa_halfsamples.csv after every chunk; a rerun resumes.

Why not the bootstrap. A bootstrap resample duplicates ~37% of patients; the
duplicates fall into the same bin cell and become each other's nearest
neighbour, so the kNN divergence estimate of the direct kappa_max is inflated
and its spread collapses (bootstrap failure for nearest-neighbour statistics,
Abadie & Imbens 2008). Half-samples contain no duplicates. For a root-n
estimator, a without-replacement sample of m = n/2 has variance
(1/m - 1/n) sigma^2 = sigma^2/n, i.e. the SD of the replicates estimates the SE
of the full-sample estimate without rescaling (delete-d jackknife with d = n/2,
Shao & Wu 1989; subsampling, Politis, Romano & Wolf 1999).

Interval: CI = estimate +/- z * SE. The CIs are conditional on the selected
feature sets and cover patient-level sampling variability at the observed site
composition and prevalence, not selection variability or estimator bias with
respect to the true kappa_max. Binning dither is redrawn per replicate at
half-sample size, which makes the SE slightly conservative. rep_mean - estimate
reflects the n/2-vs-n bias of the estimators (diagnostic only).

Flower is not required: federated/kappa/{client,server}.py import it at module
level, but the functions used here never call it, so a stand-in satisfies the
import (as in simulation_validation/simulate_known_limits.py).

Usage (from anywhere inside the repository):
    python kappa-boxplot.py                       # point estimates + half-sample CIs and plots
    python kappa-boxplot.py --n_rep 0             # point estimates only
    options: [--n_rep 1000] [--hours 6.5] [--n_iters 5] [--n_seeds 20] [--n_jobs -1] [--out DIR]
Outputs go to --out (default: halfsample_kappa/ next to this script).
"""

import argparse
import json
import logging
import os
import sys
import time
import types
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from matplotlib import font_manager
from matplotlib.patches import Patch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "general_config").is_dir())
T0 = time.time()


def _setup():
    """Repository root on sys.path and as working directory (DATA_DIR is relative),
    progress bars off, and a Flower stand-in. Idempotent; also run in every worker."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    os.environ.setdefault("TQDM_DISABLE", "1")
    if "flwr" in sys.modules:
        return
    stub = type("FlowerStandIn", (), {"__init__": lambda self, *a, **k: None})

    def attr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return stub

    names = ("flwr", "flwr.common", "flwr.client", "flwr.server", "flwr.server.strategy")
    mods = {n: types.ModuleType(n) for n in names}
    for m in mods.values():
        m.__getattr__ = attr
    mods["flwr"].common, mods["flwr"].client, mods["flwr"].server = (
        mods["flwr.common"], mods["flwr.client"], mods["flwr.server"])
    mods["flwr.server"].strategy = mods["flwr.server.strategy"]
    sys.modules.update(mods)


_setup()
from general_config import CLIENT_NAMES, MAX_FEATURES, N_BINS, N_ITERATIONS
from mydataloading import prepare_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DF, OVERSAMPLE = 4, 1                         # server.py defaults
ALPHA = 0.05
Z = NormalDist().inv_cdf(1 - ALPHA / 2)
MIN_REP = 200
MASTER_SEED = 2509
SCOPES = [str(c) for c in CLIENT_NAMES] + ["ALL"]
COLORS = {"direct": "#3A72B2", "federated": "#843414"}
LABELS = {"direct": "Direct (real data)", "federated": "Federated (t-copula)"}

# PLOS: Arial 8-12 pt, width <= 2250 px at 300 dpi (7.5 in)
FONT = next((f for f in ("Arial", "Liberation Sans")
             if any(e.name == f for e in font_manager.fontManager.ttflist)), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "custom",
                     "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.cal": FONT})
FIG_W, FIG_H, DPI = 7.5, 3.6, 300


# data 
@contextmanager
def skip_readonly_writes():
    """prepare_data writes side files (missingness report, split) into FLdata/. If
    that folder is read-only, skip those writes instead of failing; the data read
    are unchanged."""
    originals = {(cls, m): getattr(cls, m) for cls, m in
                 ((pd.DataFrame, "to_csv"), (pd.Series, "to_csv"), (pd.DataFrame, "to_parquet"))}

    def guarded(fn):
        def wrapper(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except PermissionError as e:
                log.warning(f"Read-only, not written: {e.filename}")
        return wrapper

    for (cls, m), fn in originals.items():
        setattr(cls, m, guarded(fn))
    try:
        yield
    finally:
        for (cls, m), fn in originals.items():
            setattr(cls, m, fn)


def load_client(cid):
    """All patients of one client, stacked as in federated/kappa/client.py."""
    with skip_readonly_writes():
        X_tr, y_tr, X_te, y_te, X_va, y_va, names = prepare_data(cid)
    X = np.nan_to_num(np.vstack([X_tr.values, X_te.values, X_va.values]).astype(np.float64))
    y = np.concatenate([y_tr, y_te, y_va]).astype(int)
    return X, y, list(names)


def half(X, y, rng):
    """Half-sample: n_c // 2 patients per class, without replacement."""
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), np.sum(y == c) // 2, replace=False)
                          for c in (0, 1)])
    return X[idx], y[idx]


#estimation

def one_pass(data, rep, n_iters, n_seeds, feats=None, only=None):
    """Direct and federated kappa_max for every scope (or only `only`).
    rep = -1: full data with feature selection; rep >= 0: half-sample with the
    fixed feature sets feats[scope] = (direct, federated), no selection."""
    _setup()
    from kappa_limits import compute_theoretical_kappa
    from federated.kappa.client import compute_statistics
    from federated.kappa.server import merge_payloads, _d_ensemble
    logging.getLogger(_d_ensemble.__module__).setLevel(logging.WARNING)

    rng = np.random.default_rng([MASTER_SEED, rep + 1])
    clients = {str(c): (X, y) if rep < 0 else half(X, y, rng) for c, (X, y) in data.items()}
    rows = []
    for k, scope in enumerate(SCOPES):
        if only not in (None, scope):
            continue
        cs = [clients[scope]] if scope != "ALL" else list(clients.values())
        fd, ff = feats[scope] if feats else (slice(None), slice(None))
        md, mf = (len(fd), len(ff)) if feats else (MAX_FEATURES, MAX_FEATURES)
        np.random.seed([MASTER_SEED, rep + 1, k])        # binning dither uses the global RNG
        rc = compute_theoretical_kappa(np.vstack([X[:, fd] for X, _ in cs]),
                                       np.concatenate([y for _, y in cs]),
                                       md, N_BINS, n_iters)
        merged = merge_payloads([json.loads(compute_statistics(X[:, ff], y)) for X, y in cs])
        rd = _d_ensemble(merged, df=DF, max_features=mf, n_iterations=n_iters,
                         n_bins=N_BINS, n_seeds=n_seeds, oversample=OVERSAMPLE)
        rows.append(dict(rep=rep, scope=scope, direct=rc["theoretical_kappa"],
                         federated=rd.get("kappa_max", np.nan),
                         direct_sel=list(rc["selected_features"]),
                         fed_sel=list(rd.get("selected_features", []))))
    return rows


def run_halfsamples(data, feats, a, rfile):
    """Replicates in chunks, appended to rfile; stops at n_rep or when the next
    chunk would exceed the time budget. Resumes from an existing rfile."""
    done = pd.read_csv(rfile, usecols=["rep"]).rep if rfile.exists() else pd.Series(dtype=int)
    b = int(done.max()) + 1 if len(done) else 0
    if b:
        log.info(f"Resuming after {done.nunique()} replicates in {rfile}")
    chunk = 2 * effective_n_jobs(a.n_jobs)
    deadline, t_chunk = T0 + 3600 * a.hours, 0.0
    with Parallel(n_jobs=a.n_jobs) as par:
        while b < a.n_rep and time.time() + 1.1 * t_chunk < deadline:
            t, idx = time.time(), range(b, min(b + chunk, a.n_rep))
            rows = [r for rs in par(delayed(one_pass)(data, i, a.n_iters, a.n_seeds, feats)
                                    for i in idx) for r in rs]
            n_bad = sum(len(r["direct_sel"]) != len(feats[r["scope"]][0]) or
                        len(r["fed_sel"]) != len(feats[r["scope"]][1]) for r in rows)
            if n_bad:
                log.warning(f"{n_bad} estimates did not keep all fixed features (reselection?)")
            pd.DataFrame(rows).drop(columns=["direct_sel", "fed_sel"]).to_csv(
                rfile, mode="a", header=not rfile.exists(), index=False)
            t_chunk, b = time.time() - t, idx[-1] + 1
            log.info(f"{b}/{a.n_rep} replicates, {t_chunk / len(idx):.1f} s/replicate (wall), "
                     f"{(deadline - time.time()) / 3600:.2f} h budget left")
    if b < a.n_rep:
        log.warning(f"Time budget reached after {b} replicates (rerun to continue)")
    return pd.read_csv(rfile, dtype={"scope": str})


#summary and plots

def add_gaps(df):
    bits = lambda k: -np.log2(1 - k)  # CDR from kappa = 1 - 2^-CDR
    return df.assign(gap=df.federated - df.direct,
                     gap_bits=bits(df.federated) - bits(df.direct))


def summarize(point, reps):
    """Normal CI centred on the point estimate, SE = SD of the half-sample replicates."""
    point, reps = add_gaps(point).set_index("scope"), add_gaps(reps)
    rows = []
    for scope, g in reps.groupby("scope", sort=False):
        for m in ("direct", "federated", "gap", "gap_bits"):
            ok = np.isfinite(g[m].to_numpy())
            v, even = g[m].to_numpy()[ok], g["rep"].to_numpy()[ok] % 2 == 0
            if len(v) < 20:
                log.warning(f"{scope}/{m}: only {len(v)} valid replicates, skipped")
                continue
            est, se = point.loc[scope, m], v.std(ddof=1)
            rows.append(dict(
                scope=scope, metric=m, estimate=est, se=se,
                ci_lo=est - Z * se, ci_hi=est + Z * se,
                rep_mean=v.mean(),                                 # diagnostic
                se_split_reldiff=abs(v[even].std(ddof=1) - v[~even].std(ddof=1)) / se,
                n_valid=len(v)))
    return pd.DataFrame(rows)


def save(fig, out, stem):
    """PDF, PNG and PLOS TIFF (LZW, RGB without alpha, 300 dpi, fixed 7.5 in width)."""
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=DPI)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    img = Image.open(buf).convert("RGB")
    img.save(out / f"{stem}.tif", compression="tiff_lzw", dpi=(DPI, DPI))
    if not (789 <= img.width <= 2250 and img.height <= 2625):
        log.warning(f"{stem}.tif is {img.width}x{img.height} px, outside PLOS limits")
    plt.close(fig)
    log.info(f"Plot saved � {out / stem}.pdf/.png/.tif ({img.width}x{img.height} px)")


def _layout(n_clients):
    return plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                        gridspec_kw={"width_ratios": [n_clients, 1]}, sharey=True)


def _finish(fig, axes, scopes):
    axes[0].set_xticks([i * 3 + 0.5 for i in range(len(scopes))])
    axes[0].set_xticklabels([f"Client {s}" for s in scopes], fontsize=8)
    axes[0].set_ylabel(r"$\hat{\kappa}_{\max}$", fontsize=11)
    axes[0].set_title("Per-client comparison", fontsize=11)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Pooled", "Fed.\njoint"], fontsize=8)
    axes[1].set_title("Global comparison", fontsize=11)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    handles = [Patch(facecolor=COLORS[s], label=LABELS[s]) for s in ("direct", "federated")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.08, 1, 1])


def _panels(scopes):
    """(axis index, scope, x position) for every box/point, as in the original layout."""
    return [(0, s, 3 * i) for i, s in enumerate(scopes)] + [(1, "ALL", 0)]


def make_boxplots(reps, out):
    """
    Left panel : per-client  (Direct vs Federated side by side)
    Right panel: global      (Pooled vs Joint Federated)
    """
    scopes = [s for s in SCOPES if s != "ALL"]
    fig, axes = _layout(len(scopes))
    for a, scope, x in _panels(scopes):
        for j, sc in enumerate(("direct", "federated")):
            v = reps.loc[reps.scope == scope, sc].dropna().values
            bp = axes[a].boxplot([v], positions=[x + j], widths=0.7,
                                 patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(COLORS[sc])
            bp["medians"][0].set_color("black"); bp["medians"][0].set_linewidth(1.5)
    _finish(fig, axes, scopes)
    save(fig, out, "kappa_boxplot_comparison")


def make_ci_plot(summary, out):
    """Point estimates with 95% CIs, same layout and colours as the boxplots."""
    s = summary.set_index(["scope", "metric"])
    scopes = [sc for sc in SCOPES if sc != "ALL"]
    fig, axes = _layout(len(scopes))
    for a, scope, x in _panels(scopes):
        for j, sc in enumerate(("direct", "federated")):
            if (scope, sc) in s.index:
                r = s.loc[(scope, sc)]
                axes[a].errorbar(x + j, r["estimate"], yerr=Z * r["se"], fmt="o", ms=6,
                                 color=COLORS[sc], capsize=4, elinewidth=1.5)
    axes[0].set_xlim(-0.85, 3 * len(scopes) - 1.15)
    axes[1].set_xlim(-0.85, 1.85)
    _finish(fig, axes, scopes)
    save(fig, out, "kappa_ci")


# entry point 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_rep", type=int, default=1000,
                   help="max. half-sample replicates; 0 = point estimates only")
    p.add_argument("--hours", type=float, default=6.5, help="wall-clock budget for the whole run")
    p.add_argument("--n_iters", type=int, default=N_ITERATIONS, help="Crow-Watts binning iterations")
    p.add_argument("--n_seeds", type=int, default=20, help="copula draws S (as in manuscript)")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--out", type=Path, default=HERE / "halfsample_kappa")
    a = p.parse_args()
    out = a.out if a.out.is_absolute() else HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    if FONT != "Arial":
        log.warning(f"Arial not installed, using {FONT}")

    try:
        loaded = {cid: load_client(cid) for cid in CLIENT_NAMES}
    except PermissionError as e:
        sys.exit(f"No write permission for {e.filename}. prepare_data must create its split "
                 f"there once; make FLdata/ writable (e.g. sudo chown -R $USER FLdata).")
    data = {cid: v[:2] for cid, v in loaded.items()}
    names = next(iter(loaded.values()))[2]

    # Feature selection: exactly once per scope, on the full data.
    pfile = out / "kappa_point.csv"
    if pfile.exists():
        log.info(f"Point estimates and features from {pfile} (delete to recompute)")
        point = pd.read_csv(pfile, dtype={"scope": str})
    else:
        log.info("Point estimates with feature selection on full data &")
        rows = Parallel(n_jobs=a.n_jobs)(delayed(one_pass)(data, -1, a.n_iters, a.n_seeds, only=s)
                                         for s in SCOPES)
        point = pd.DataFrame([r for rs in rows for r in rs])
        for col, sel in (("direct_features", "direct_sel"), ("federated_features", "fed_sel")):
            point[col] = [";".join(names[int(j)] for j in f) for f in point.pop(sel)]
        point.to_csv(pfile, index=False)

    if a.n_rep == 0:
        cols = ["scope", "direct", "federated", "gap", "gap_bits"]
        print(add_gaps(point)[cols].round(4).to_string(index=False))
        log.info(f"Selected features in {pfile}. No resampling (use --n_rep N for CIs).")
        return

    feats = {r.scope: ([names.index(f) for f in r.direct_features.split(";")],
                       [names.index(f) for f in r.federated_features.split(";")])
             for r in point.itertuples()}

    log.info(f"Half-sampling: up to {a.n_rep} replicates within {a.hours} h "
             f"(fixed features, no reselection) &")
    reps = run_halfsamples(data, feats, a, out / "kappa_halfsamples.csv")
    if reps.rep.nunique() < MIN_REP:
        log.warning(f"{reps.rep.nunique()} < {MIN_REP} replicates: SE and CIs will be unstable")

    summary = summarize(point, reps)
    summary.to_csv(out / "kappa_ci_summary.csv", index=False)
    make_boxplots(reps, out)
    make_ci_plot(summary, out)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()