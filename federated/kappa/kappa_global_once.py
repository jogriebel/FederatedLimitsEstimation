"""
kappa_max for the global scope only (pooled direct vs joint federated), computed
once on the full data with greedy feature selection. No half-sampling, no
per-client scopes.

The two estimators can run as separate processes, so the wall time is the
longer of the two instead of their sum, and each result is written to disk as
soon as it exists:

    nohup python kappa_global_once.py --which direct    > global_direct.log 2>&1 &
    nohup python kappa_global_once.py --which federated > global_federated.log 2>&1 &
    tail -f global_*.log

Progress bars of the estimators are left on (kappa-boxplot.py switches them off),
and every log line carries the elapsed time.

Reproducibility: the global RNG is seeded as in kappa-boxplot.py for scope "ALL"
(np.random.seed([2509, 0, k_ALL])). The direct estimate is therefore identical to
that script's ALL row. The federated estimate is identical only if _d_ensemble
seeds its own copula draws; otherwise it differs by Monte Carlo noise, because in
kappa-boxplot.py the direct estimate runs first and advances the global RNG.
--which both runs them in sequence in one process and reproduces both exactly.

Safe next to a running kappa-boxplot.py: writes only to --out (default:
global_once/ next to this script) and writes nothing into FLdata/ (all splits are
stacked, so the split files are irrelevant here).

Options: [--which both|direct|federated] [--n_iters 5] [--n_seeds 20] [--out DIR]
"""

import argparse
import json
import logging
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "general_config").is_dir())
T0 = time.time()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                  # DATA_DIR is relative


def _flower_standin():
    """federated/kappa/{client,server}.py import flwr at module level; the
    functions used here never call it."""
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


_flower_standin()
from general_config import CLIENT_NAMES, MAX_FEATURES, N_BINS, N_ITERATIONS
from mydataloading import prepare_data

# federated/kappa/client.py parses sys.argv at import time: hide this script's
# arguments from it (as in simulation_validation/simulate_known_limits.py).
_argv, sys.argv = sys.argv, sys.argv[:1]
try:
    from kappa_limits import compute_theoretical_kappa
    from federated.kappa.client import compute_statistics
    from federated.kappa.server import merge_payloads, _d_ensemble
finally:
    sys.argv = _argv


class _Elapsed(logging.Formatter):
    def format(self, record):
        h, r = divmod(int(time.time() - T0), 3600)
        return f"[{h:02d}:{r // 60:02d}:{r % 60:02d}] {record.getMessage()}"


_h = logging.StreamHandler()
_h.setFormatter(_Elapsed())
logging.basicConfig(level=logging.INFO, handlers=[_h])
log = logging.getLogger(__name__)

DF, OVERSAMPLE = 4, 1                           # server.py defaults
MASTER_SEED = 2509
SCOPES = [str(c) for c in CLIENT_NAMES] + ["ALL"]
K_ALL = SCOPES.index("ALL")                     # seed index of "ALL" in kappa-boxplot.py


@contextmanager
def no_writes():
    """Turn prepare_data's side-file writes into no-ops."""
    targets = ((pd.DataFrame, "to_csv"), (pd.Series, "to_csv"), (pd.DataFrame, "to_parquet"))
    originals = {(cls, m): getattr(cls, m) for cls, m in targets}
    for cls, m in targets:
        setattr(cls, m, lambda self, *a, **k: None)
    try:
        yield
    finally:
        for (cls, m), fn in originals.items():
            setattr(cls, m, fn)


def load_client(cid):
    """All patients of one client, stacked as in federated/kappa/client.py."""
    with no_writes():
        X_tr, y_tr, X_te, y_te, X_va, y_va, names = prepare_data(cid)
    X = np.nan_to_num(np.vstack([X_tr.values, X_te.values, X_va.values]).astype(np.float64))
    y = np.concatenate([y_tr, y_te, y_va]).astype(int)
    return X, y, list(names)


def bits(k):
    return -np.log2(1 - k)                      # CDR from kappa = 1 - 2^-CDR


def save(out, which, kappa, sel, names, a, seconds):
    row = dict(scope="ALL", estimator=which, kappa_max=kappa, cdr_bits=bits(kappa),
               features=";".join(names[int(j)] for j in sel),
               n_iters=a.n_iters, n_seeds=a.n_seeds if which == "federated" else np.nan,
               seconds=round(seconds))
    f = out / f"kappa_ALL_{which}.csv"
    pd.DataFrame([row]).to_csv(f, index=False)
    log.info(f"{which}: kappa_max = {kappa:.4f} ({bits(kappa):.3f} bits), "
             f"{seconds / 3600:.2f} h -> {f}")
    log.info(f"{which} features: {row['features'].replace(';', ', ')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=("both", "direct", "federated"), default="both")
    p.add_argument("--n_iters", type=int, default=N_ITERATIONS, help="Crow-Watts binning iterations")
    p.add_argument("--n_seeds", type=int, default=20, help="copula draws S (as in manuscript)")
    p.add_argument("--out", type=Path, default=HERE / "global_once")
    a = p.parse_args()
    out = a.out if a.out.is_absolute() else HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger(_d_ensemble.__module__).setLevel(logging.INFO)

    loaded = [load_client(cid) for cid in CLIENT_NAMES]
    names = loaded[0][2]
    log.info(f"Loaded {len(loaded)} clients: {sum(len(y) for _, y, _ in loaded)} patients, "
             f"{len(names)} candidate features, MAX_FEATURES={MAX_FEATURES}, N_BINS={N_BINS}, "
             f"n_iters={a.n_iters}, n_seeds={a.n_seeds}")

    np.random.seed([MASTER_SEED, 0, K_ALL])

    if a.which in ("both", "direct"):
        t = time.time()
        log.info("Pooled direct estimate started")
        rc = compute_theoretical_kappa(np.vstack([X for X, _, _ in loaded]),
                                       np.concatenate([y for _, y, _ in loaded]),
                                       MAX_FEATURES, N_BINS, a.n_iters)
        save(out, "direct", rc["theoretical_kappa"], rc["selected_features"], names, a,
             time.time() - t)

    if a.which in ("both", "federated"):
        t = time.time()
        log.info("Joint federated estimate started")
        merged = merge_payloads([json.loads(compute_statistics(X, y)) for X, y, _ in loaded])
        rd = _d_ensemble(merged, df=DF, max_features=MAX_FEATURES, n_iterations=a.n_iters,
                         n_bins=N_BINS, n_seeds=a.n_seeds, oversample=OVERSAMPLE)
        save(out, "federated", rd.get("kappa_max", np.nan), rd.get("selected_features", []),
             names, a, time.time() - t)

    d, f = out / "kappa_ALL_direct.csv", out / "kappa_ALL_federated.csv"
    if d.exists() and f.exists():
        kd, kf = pd.read_csv(d).kappa_max[0], pd.read_csv(f).kappa_max[0]
        log.info(f"Gap (federated - direct): {kf - kd:+.4f} kappa, "
                 f"{bits(kf) - bits(kd):+.3f} bits")


if __name__ == "__main__":
    main()