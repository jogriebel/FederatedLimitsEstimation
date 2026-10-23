"""
kappa_max: direct (real data) vs federated (t-copula), parallel version of
kappa-boxplot.py with the same estimators, estimates and half-sampling design.

Why it is faster. In kappa-boxplot.py one federated estimate is _d_ensemble, i.e.
n_seeds complete Crow & Watts runs (greedy selection included) one after another
on one core; for the global scope that is about 20 x 30 min. Here every direct
estimate and every copula draw is its own task, all tasks run in a process pool
on all available cores, and the draws are aggregated afterwards exactly as in
federated/kappa/server._d_ensemble: kappa_max = median of the valid draws,
CDR = mean, features of the draw at argsort(kappas)[len // 2]. Wall time is
then about one draw on the largest scope instead of n_seeds of them.

Stages (all results go to --out, default kappa_parallel/<DATA_BALANCE>/ next to
this script, i.e. natural/ or 20/)
  1. Point estimates on the full data with greedy feature selection
     -> kappa_point.csv (estimates, per-draw spread, selected features).
  2. Half-sampling: n_c // 2 patients per (client, class) without replacement,
     the same half-sample for both estimators, features fixed to those selected
     in stage 1 (no reselection). Runs until --n_rep replicates or the --hours
     budget -> kappa_halfsamples.csv. SE = SD of the replicates (delete-d
     jackknife with d = n/2), CI = estimate +/- z * SE; the reasoning and the
     caveats are those in the docstring of kappa-boxplot.py.
     The selection runs on X[:, cols], so the estimators return positions in
     that subset; run_task maps them back, and tasks.csv always holds original
     column indices. Results written before this fix hold the positions 0..k-1
     instead; repair them once with fix_tasks_features.py.
  3. Report: kappa_halfsamples.csv, kappa_ci_summary.csv,
     kappa_boxplot_comparison.{tif,pdf,png}, kappa_ci.{tif,pdf,png}. Written at
     the end of every run, and on demand with --plot_only.

Figures follow the PLOS Digital Health figure requirements: TIFF with LZW
compression, RGB 8 bit, no alpha channel, single layer, 300 dpi, width
789-2250 px (full page width 7.5 in), height <= 2625 px, < 10 MB, Arial 8-12 pt,
a 2 pt white border and no surrounding white space, no title or figure number
in the file (they go into the caption), lettered panels (A, B). Colours and
style match simulate_known_limits.py: Okabe-Ito blue #0072B2 for the direct
(centralized) and vermillion #D55E00 for the federated estimate, circles vs
squares, light grid, no top/right spines. Clients are labelled with their
database names (from mydataloading.get_client_name; --client_labels overrides).
The .tif files are the submission files (rename to Fig<n>.tif); .pdf and .png
are for drafts.

--plot_only computes nothing: it rebuilds the report from tasks.csv and
kappa_point.csv in --out (a few seconds) and uses every replicate that is
complete so far. It can run while a computing run is still going.

Every finished task is appended to tasks.csv at once. An interrupted run loses
only the tasks in flight; rerunning with the same --out resumes, and a larger
--n_rep or --hours extends the replicates. config.json guards against resuming
with other settings or other data.

Reproducibility. Half-samples are drawn exactly as in kappa-boxplot.py and the
direct estimates use its global seed ([2509, rep + 1, k]), so direct values are
identical to that script's. Copula draws use _d_ensemble's seeds (42 + 17 s).
The binning dither of each draw gets its own seed ([2509, rep + 1, k, 1000 + s])
instead of continuing the global RNG stream of the previous draw, so federated
values differ from a sequential _d_ensemble run only by dither noise and do not
depend on --n_jobs. --selftest checks one draw and one direct estimate against
_d_ensemble / compute_theoretical_kappa on the smallest client, and one direct
estimate on a fixed feature subset (stage 2) for the logged feature indices.

Usage (from anywhere inside the repository):
    python kappa_parallel.py --selftest            # ~minutes, run once first
    python kappa_parallel.py                       # point estimates + CIs + plots
    python kappa_parallel.py --n_rep 0             # point estimates only
    python kappa_parallel.py --scopes ALL          # global scope only
    python kappa_parallel.py --plot_only           # report from existing results only
    options: [--n_rep 1000] [--hours 6.5] [--n_iters 5] [--n_seeds 20]
             [--rep_seeds S] [--n_jobs N] [--scopes ...] [--out DIR] [--plot_only]
             [--client_labels 1=MIMIC-IV 2=HiRID ...]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one thread per worker; cores go to workers
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import contextlib
import csv
import hashlib
import io
import json
import logging
import multiprocessing as mp
import re
import sys
import time
import types
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from io import BytesIO
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

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
from general_config import CLIENT_NAMES, DATA_BALANCE, MAX_FEATURES, N_BINS, N_ITERATIONS
from mydataloading import prepare_data

try:
    from mydataloading import get_client_name
except ImportError:
    get_client_name = None

_argv, sys.argv = sys.argv, sys.argv[:1]       # client.py parses sys.argv at import
try:
    from kappa_limits import compute_theoretical_kappa
    from federated.kappa.client import compute_statistics
    from federated.kappa.server import _d_ensemble, generate_synthetic, merge_payloads, run_method_d
finally:
    sys.argv = _argv


class _Elapsed(logging.Formatter):
    def format(self, record):
        h, r = divmod(int(time.time() - T0), 3600)
        return f"[{h:02d}:{r // 60:02d}:{r % 60:02d}] {record.getMessage()}"


_handler = logging.StreamHandler()
_handler.setFormatter(_Elapsed())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)  # server.py configured logging first
log = logging.getLogger("kappa_parallel")
for _name in ("federated.kappa.server", "federated.kappa.client", "kappa_limits", "fontTools",
              "matplotlib"):
    logging.getLogger(_name).setLevel(logging.WARNING)

DF, OVERSAMPLE, MIN_PER_CLASS = 4, 1, 100      # server.py defaults
ALPHA = 0.05
Z = NormalDist().inv_cdf(1 - ALPHA / 2)
MIN_REP = 200
MASTER_SEED = 2509
SCOPES = [str(c) for c in CLIENT_NAMES] + ["ALL"]
EST = ("direct", "federated")
COLORS = {"direct": "#0072B2", "federated": "#D55E00"}     # COL_CEN / COL_FED in simulate_known_limits.py
MARKERS = {"direct": "o", "federated": "s"}
LABELS = {"direct": "Direct (real data)", "federated": "Federated (t-copula)"}
DB_LABELS = {"mimic": "MIMIC-IV", "eicu": "eICU-CRD", "hirid": "HiRID", "umc": "UMCdb",
             "aumc": "UMCdb", "amsterdam": "UMCdb", "uka": "UKA", "augsburg": "UKA"}
CLIENT_LABELS = {}                                           # scope -> label, set in main()
FIELDS = ["rep", "scope", "est", "seed", "kappa", "cdr", "valid", "n_sel", "features", "seconds"]



def _find_font():
    """Arial if available. Fonts installed after matplotlib built its cache are
    registered from the font directories (system, user, conda environment)."""
    have = lambda: {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" not in have():
        dirs = [Path(sys.prefix) / "fonts", Path(sys.prefix) / "share" / "fonts",
                Path.home() / ".fonts", Path.home() / ".local" / "share" / "fonts"]
        found = font_manager.findSystemFonts() + font_manager.findSystemFonts(
            fontpaths=[str(d) for d in dirs if d.is_dir()])
        for path in found:
            if Path(path).name.lower().startswith("arial"):
                try:
                    font_manager.fontManager.addfont(path)
                except Exception:
                    pass
    return next((f for f in ("Arial", "Liberation Sans") if f in have()), "DejaVu Sans")


FONT = _find_font()
FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_PANEL = 8, 10, 10, 9, 12   # PLOS: 8-12 pt
plt.rcParams.update({
    "font.family": FONT, "font.size": FS_LABEL,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold", "mathtext.cal": FONT,
    "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL, "legend.fontsize": FS_LEGEND,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
FIG_W, FIG_H, DPI = 7.5, 3.4, 300        # 7.5 in = 2250 px, the PLOS maximum width
PAD_IN = 2 / 72                           # 2 pt white border


# data

@contextlib.contextmanager
def _writes(allowed):
    """prepare_data writes side files into FLdata/. Main process: writes go through,
    read-only failures are skipped (as in kappa-boxplot.py). Workers: no writes at
    all, the split exists by then."""
    targets = ((pd.DataFrame, "to_csv"), (pd.Series, "to_csv"), (pd.DataFrame, "to_parquet"))
    originals = {(cls, m): getattr(cls, m) for cls, m in targets}

    def guarded(fn):
        def wrapper(self, *a, **k):
            if not allowed:
                return None
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


def load_all(writes):
    """All patients of every client, stacked as in federated/kappa/client.py."""
    data = {}
    with _writes(writes):
        for cid in CLIENT_NAMES:
            X_tr, y_tr, X_te, y_te, X_va, y_va, names = prepare_data(cid)
            X = np.nan_to_num(np.vstack([X_tr.values, X_te.values, X_va.values]).astype(np.float64))
            y = np.concatenate([y_tr, y_te, y_va]).astype(int)
            data[str(cid)] = (X, y)
    return data, list(names)


def client_labels(overrides):
    """Database names for the client ids, from mydataloading.get_client_name."""
    labels = {}
    for c in CLIENT_NAMES:
        try:
            raw = str(get_client_name(c)) if get_client_name else f"Client {c}"
        except Exception:
            raw = f"Client {c}"
        key = re.sub(r"[^a-z]", "", raw.lower())
        labels[str(c)] = next((v for k, v in DB_LABELS.items() if key.startswith(k)), raw)
    for item in overrides or []:
        c, _, name = item.partition("=")
        if c not in labels or not name:
            sys.exit(f"--client_labels expects id=name with id in {list(labels)}, got {item!r}")
        labels[c] = name
    labels["ALL"] = "All clients"
    return labels


def fingerprint(data):
    h = hashlib.sha1()
    for c, (X, y) in data.items():
        h.update(c.encode())
        h.update(np.ascontiguousarray(X).tobytes())
        h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()[:16]


_DATA = None


def _init_worker(expected):
    """Each worker loads the data itself (no copy of patient data on disk) and
    checks that it is bit-identical to the main process's."""
    global _DATA
    for h in logging.getLogger().handlers:
        h.setLevel(logging.WARNING)
    with contextlib.redirect_stdout(io.StringIO()):
        _DATA, _ = load_all(writes=False)
    if fingerprint(_DATA) != expected:
        raise RuntimeError("Worker loaded different data than the main process "
                           "(non-deterministic split or row order in prepare_data?)")


def half(X, y, rng):
    """Half-sample: n_c // 2 patients per class, without replacement."""
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), np.sum(y == c) // 2, replace=False)
                          for c in (0, 1)])
    return X[idx], y[idx]


def scope_clients(data, scope, rep):
    """Client arrays of a scope; rep >= 0 gives that replicate's half-samples, drawn
    for all clients in order with one RNG, exactly as in kappa-boxplot.py."""
    if rep >= 0:
        rng = np.random.default_rng([MASTER_SEED, rep + 1])
        data = {c: half(X, y, rng) for c, (X, y) in data.items()}
    return list(data.values()) if scope == "ALL" else [data[scope]]


# tasks

def make_task(est, scope, rep, seed, cols, n_iters):
    k = SCOPES.index(scope)
    gseed = [MASTER_SEED, rep + 1, k] if est == "direct" else [MASTER_SEED, rep + 1, k, 1000 + seed]
    return dict(est=est, scope=scope, rep=rep, seed=seed, cols=cols, n_iters=n_iters,
                max_feat=MAX_FEATURES if cols is None else len(cols), gseed=gseed)


def run_task(t, data=None):
    """One direct estimate or one copula draw of a federated estimate.
    Returns a result row, or a row with 'error' set. Selected features are
    returned as original column indices, also when the task runs on a fixed
    feature subset (t['cols'])."""
    data = _DATA if data is None else data
    t0 = time.time()
    res = {k: t[k] for k in ("rep", "scope", "est", "seed")}
    try:
        cs = scope_clients(data, t["scope"], t["rep"])
        cols = slice(None) if t["cols"] is None else list(t["cols"])
        np.random.seed(t["gseed"])                 # binning dither uses the global RNG
        if t["est"] == "direct":
            r = compute_theoretical_kappa(np.vstack([X[:, cols] for X, _ in cs]),
                                          np.concatenate([y for _, y in cs]),
                                          t["max_feat"], N_BINS, t["n_iters"])
            kappa, cdr, sel = r["theoretical_kappa"], r.get("CDR", np.nan), r.get("selected_features", [])
            valid = bool(np.isfinite(kappa))
        else:
            merged = merge_payloads([json.loads(compute_statistics(X[:, cols], y)) for X, y in cs])
            Xs, ys, _, _ = generate_synthetic(merged, df=DF, seed=42 + 17 * t["seed"],
                                              oversample=OVERSAMPLE, min_per_class=MIN_PER_CLASS)
            if Xs is None:
                raise ValueError("empty scope")
            r = run_method_d(Xs, ys, t["max_feat"], t["n_iters"], N_BINS)
            kappa, cdr, sel, valid = r["kappa_max"], r["cdr"], r["selected_features"], bool(r["valid"])
        sel = [int(j) for j in sel]
        if t["cols"] is not None:                  # sel indexes X[:, cols]: map back
            sel = [int(cols[j]) for j in sel]
        res.update(kappa=float(kappa), cdr=float(cdr), valid=valid, n_sel=len(sel),
                   features=";".join(str(j) for j in sel))
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"[:500]
    res["seconds"] = round(time.time() - t0, 1)
    return res


class TaskLog:
    """Append-only CSV of finished tasks; the set of done keys allows resuming."""

    def __init__(self, path):
        self.path = path
        self.done = set()
        if path.exists() and path.stat().st_size:
            df = self.read()
            self.done = set(zip(df.rep.astype(int), df.scope, df.est, df.seed.astype(int)))
        self.fh = open(path, "a", newline="")
        self.w = csv.DictWriter(self.fh, FIELDS, extrasaction="ignore")
        if self.fh.tell() == 0:
            self.w.writeheader()
        elif not path.read_bytes().endswith(b"\n"):   # half-written line from a killed run
            self.fh.write("\n")

    def add(self, r):
        self.w.writerow(r)
        self.fh.flush()
        self.done.add((r["rep"], r["scope"], r["est"], r["seed"]))

    def read(self):
        return read_tasks(self.path)


def read_tasks(path):
    """tasks.csv; a half-written last line (file read while a run appends) is dropped."""
    df = pd.read_csv(path, dtype={"scope": str, "features": str}, on_bad_lines="skip")
    df = df.dropna(subset=["seconds"]).fillna({"features": ""})
    return df.astype({"rep": int, "seed": int, "n_sel": int, "valid": bool})


def run_pool(ex, tasks, tlog, n_inflight, deadline=None, progress=None, every=60):
    """Run tasks with at most n_inflight queued. With a deadline, no new task is
    submitted once the longest task seen so far would overrun it. Returns True if
    every task was submitted."""
    pending, it, longest, last = {}, iter(tasks), 0.0, time.time()
    exhausted = stopped = False
    while True:
        while not (exhausted or stopped) and len(pending) < n_inflight:
            if deadline and time.time() + longest > deadline:
                stopped = True
                break
            t = next(it, None)
            if t is None:
                exhausted = True
                break
            pending[ex.submit(run_task, t)] = t
        if not pending:
            return exhausted
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for f in done:
            t = pending.pop(f)
            r = f.result()
            tag = f"{t['est']} scope={t['scope']} rep={t['rep']} seed={t['seed']}"
            if "error" in r:
                log.warning(f"FAILED {tag}: {r['error']} (retried on the next run)")
                continue
            tlog.add(r)
            longest = max(longest, r["seconds"])
            if progress is None:
                log.info(f"done {tag}: kappa={r['kappa']:.4f} ({r['seconds'] / 60:.1f} min)")
        if progress and time.time() - last > every:
            progress()
            last = time.time()


def d_aggregate(f):
    """_d_ensemble's aggregation over the draws of one estimate (rows sorted by seed)."""
    runs = f[f.valid.astype(bool)]
    if runs.empty:
        return np.nan, np.nan, "", []
    kappas = runs.kappa.tolist()
    med = int(np.argsort(kappas)[len(kappas) // 2])
    return float(np.median(kappas)), float(np.mean(runs.cdr)), runs.features.iloc[med], kappas


def build_point(tdf, scopes, names, n_seeds):
    rows = []
    for s in scopes:
        g = tdf[(tdf.rep == -1) & (tdf.scope == s)]
        d = g[g.est == "direct"]
        f = g[g.est == "federated"].sort_values("seed")
        if len(d) != 1 or len(f) != n_seeds:
            sys.exit(f"Point estimate for scope {s} incomplete ({len(d)} direct, {len(f)}/{n_seeds} "
                     f"draws), see FAILED lines above. Rerun to retry the missing tasks.")
        fk, fc, ff, ks = d_aggregate(f)
        di = d.features.iloc[0]
        to_names = lambda idx: ";".join(names[int(j)] for j in idx.split(";") if j != "")
        rows.append(dict(scope=s, direct=d.kappa.iloc[0], federated=fk,
                         direct_cdr=d.cdr.iloc[0], federated_cdr=fc,
                         federated_draw_sd=np.std(ks, ddof=1) if len(ks) > 1 else np.nan,
                         n_valid_draws=len(ks),
                         direct_features=to_names(di), federated_features=to_names(ff),
                         direct_idx=di, federated_idx=ff,
                         federated_draws=";".join(f"{k:.6f}" for k in ks)))
    return pd.DataFrame(rows)


def write_point(point, path):
    """kappa_point.csv with the rows of this run's scopes; rows of other scopes
    already in the file (from a run with other --scopes) are kept."""
    if path.exists():
        old = pd.read_csv(path, dtype={"scope": str, "direct_idx": str, "federated_idx": str})
        point = pd.concat([old[~old.scope.isin(point.scope)], point], ignore_index=True)
        point = point.sort_values("scope", key=lambda s: s.map(SCOPES.index)).reset_index(drop=True)
    point.to_csv(path, index=False)


def idx_list(s):
    return [int(j) for j in str(s).split(";") if j not in ("", "nan")]


def fixed_features(point):
    """Feature sets of stage 1, used unchanged in every half-sample replicate."""
    return {r.scope: (idx_list(r.direct_idx), idx_list(r.federated_idx)) for r in point.itertuples()}


def build_reps(tdf, scopes, rep_seeds, feats):
    """One row per (replicate, scope) with all tasks done; only replicates that are
    complete for every scope are kept. Estimates whose selected features differ
    from the fixed set of stage 1 (as a set) are counted and reported."""
    h = tdf[(tdf.rep >= 0) & tdf.scope.isin(scopes)]
    rows, n_resel, n_tot = [], 0, 0
    for (rep, s), g in h.groupby(["rep", "scope"], sort=True):
        d = g[g.est == "direct"]
        f = g[g.est == "federated"].sort_values("seed")
        if len(d) != 1 or len(f) != rep_seeds:
            continue
        fd, ff = set(feats[s][0]), set(feats[s][1])
        n_resel += int(set(idx_list(d.features.iloc[0])) != fd) \
            + int(sum(set(idx_list(x)) != ff for x in f.features))
        n_tot += 1 + len(f)
        rows.append(dict(rep=int(rep), scope=s,
                         direct=d.kappa.iloc[0] if d.valid.iloc[0] else np.nan,
                         federated=d_aggregate(f)[0]))
    if n_resel:
        log.warning(f"{n_resel}/{n_tot} replicate estimates did not use exactly the fixed "
                    f"features of stage 1 (reselection, or a tasks.csv written before the "
                    f"feature-index fix: run fix_tasks_features.py once)")
    reps = pd.DataFrame(rows, columns=["rep", "scope", "direct", "federated"])
    full = reps.groupby("rep").scope.nunique()
    return reps[reps.rep.isin(full.index[full == len(scopes)])].reset_index(drop=True)


# summary and plots (as in kappa-boxplot.py)

def add_gaps(df):
    bits = lambda k: -np.log2(1 - k)            # CDR from kappa = 1 - 2^-CDR
    return df.assign(gap=df.federated - df.direct, gap_bits=bits(df.federated) - bits(df.direct))


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
            rows.append(dict(scope=scope, metric=m, estimate=est, se=se,
                             ci_lo=est - Z * se, ci_hi=est + Z * se,
                             rep_mean=v.mean(),                         # diagnostic
                             se_split_reldiff=abs(v[even].std(ddof=1) - v[~even].std(ddof=1)) / se,
                             n_valid=len(v)))
    return pd.DataFrame(rows)


def save(fig, out, stem):
    """PLOS TIFF (LZW, RGB 8 bit, no alpha, one layer, 300 dpi, cropped to the
    content plus a 2 pt white border), and PDF/PNG with the same crop."""
    kw = dict(bbox_inches="tight", pad_inches=PAD_IN, facecolor="white")
    for _ in range(4):                    # cropping can add labels outside the axes: refit width
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, **kw)
        img = Image.open(buf).convert("RGB")
        if img.width <= 2250:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 2245 / img.width, h)
    fig.savefig(out / f"{stem}.pdf", **kw)
    fig.savefig(out / f"{stem}.png", dpi=DPI, **kw)
    plt.close(fig)
    tif = out / f"{stem}.tif"
    img.save(tif, compression="tiff_lzw", dpi=(DPI, DPI))
    mb = tif.stat().st_size / 2 ** 20
    problems = [p for p, bad in (
        (f"width {img.width} px not in 789-2250", not 789 <= img.width <= 2250),
        (f"height {img.height} px > 2625", img.height > 2625),
        (f"{mb:.1f} MB >= 10 MB", mb >= 10),
        (f"font {FONT} instead of Arial", FONT != "Arial")) if bad]
    log.info(f"Plot saved -> {tif} (+ .pdf/.png), {img.width}x{img.height} px, {DPI} dpi, "
             f"{mb:.2f} MB" + (f"  NOT PLOS-COMPLIANT: {'; '.join(problems)}" if problems else
                               ", PLOS requirements met"))


def figure(scopes, draw, handles, out, stem):
    """(A) each client: direct vs federated on that client's data; (B) all
    clients: pooled direct vs joint federated. A panel is left out if its scopes
    were not run; panel letters only when there are two panels."""
    clients = [s for s in SCOPES if s in scopes and s != "ALL"]
    panels = (["clients"] if clients else []) + (["ALL"] if "ALL" in scopes else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_W if clients else 3.0, FIG_H),
                             sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": [len(clients) if p == "clients" else 1.2
                                                           for p in panels]})
    fig.subplots_adjust(left=0.08 if clients else 0.22, right=0.995, top=0.9,
                        bottom=0.17 if clients else 0.24,
                        wspace=0.08)
    for k, (ax, p) in enumerate(zip(axes[0], panels)):
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if p == "clients":
            for i, s in enumerate(clients):
                for j, e in enumerate(EST):
                    draw(ax, s, e, 3 * i + j)
            ax.set_xticks([3 * i + 0.5 for i in range(len(clients))])
            ax.set_xticklabels([CLIENT_LABELS.get(s, s) for s in clients])
            ax.set_xlim(-0.9, 3 * len(clients) - 1.1)
            ax.set_title("Individual clients")
        else:
            for j, e in enumerate(EST):
                draw(ax, "ALL", e, j)
            ax.set_xticks([0.5])
            ax.set_xticklabels(["Pooled / joint"])
            ax.set_xlim(-0.9, 1.9)
            ax.set_title(CLIENT_LABELS.get("ALL", "All clients"))
        ax.tick_params(axis="x", length=0)
        if len(panels) > 1:
            ax.text(-0.02 if k else -0.09, 1.02, "AB"[k], transform=ax.transAxes,
                    fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="right")
    axes[0][0].set_ylabel(r"$\hat{\kappa}_{\max}$")
    fig.legend(handles=handles, loc="lower center", ncol=2 if clients else 1, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, out, stem)


MEDIAN_COLOR = "white"


def make_boxplots(reps, scopes, out, tag=""):
    """Distribution of the half-sample estimates (box: IQR, white line: median,
    whiskers: 1.5 IQR, outliers not drawn). Box, outline, whiskers and caps are
    all in the estimator's colour, so the boxes match the colours of the CI plot."""
    def draw(ax, s, e, x):
        v = reps.loc[reps.scope == s, e].dropna().to_numpy()
        if len(v):
            c = COLORS[e]
            ax.boxplot([v], positions=[x], widths=0.7, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=c, edgecolor=c, linewidth=1.0),
                       whiskerprops=dict(color=c, linewidth=1.2),
                       capprops=dict(color=c, linewidth=1.2),
                       medianprops=dict(color=MEDIAN_COLOR, linewidth=1.5))
    handles = [Patch(facecolor=COLORS[e], edgecolor=COLORS[e], label=LABELS[e]) for e in EST]
    figure(scopes, draw, handles, out, f"kappa_boxplot_comparison{tag}")


def make_ci_plot(summary, scopes, out, tag=""):
    """Point estimates with 95% CIs, markers as in simulate_known_limits.py."""
    s = summary.set_index(["scope", "metric"])

    def draw(ax, sc, e, x):
        if (sc, e) in s.index:
            r = s.loc[(sc, e)]
            ax.errorbar(x, r["estimate"], yerr=Z * r["se"], fmt=MARKERS[e], ms=5,
                        color=COLORS[e], capsize=3, lw=1.5)
    handles = [Line2D([], [], color=COLORS[e], marker=MARKERS[e], ms=5, lw=1.5,
                      label=f"{LABELS[e]}, 95% CI") for e in EST]
    figure(scopes, draw, handles, out, f"kappa_ci{tag}")


# report

def report(out, point, scopes, rep_seeds, tdf):
    """Half-sample table, CI summary and both figures from the finished tasks.
    Nothing is estimated here. A subset of scopes gets its own file names
    (e.g. kappa_ci_ALL.pdf), so the full report is never overwritten."""
    tag = "" if scopes == SCOPES else "_" + "-".join(scopes)
    reps = build_reps(tdf, scopes, rep_seeds, fixed_features(point))
    reps.to_csv(out / f"kappa_halfsamples{tag}.csv", index=False)
    n = reps.rep.nunique()
    log.info(f"Report: {n} complete half-sample replicates -> {out / f'kappa_halfsamples{tag}.csv'}")
    if n < MIN_REP:
        log.warning(f"{n} < {MIN_REP} complete replicates: SE and CIs will be unstable")
    if n >= 2:
        make_boxplots(reps, scopes, out, tag)
    else:
        log.warning("Fewer than 2 complete replicates: no boxplots.")
    if n < 20:
        log.warning("Fewer than 20 complete replicates: no CIs (run or continue the half-sampling).")
        return
    summary = summarize(point, reps)
    summary.insert(1, "label", summary.scope.map(CLIENT_LABELS))
    summary.to_csv(out / f"kappa_ci_summary{tag}.csv", index=False)
    log.info(f"CI summary -> {out / f'kappa_ci_summary{tag}.csv'}")
    make_ci_plot(summary, scopes, out, tag)
    print(summary.round(4).to_string(index=False))


def plot_only(out, scopes_arg):
    """Report from the results already in `out`, without computing anything."""
    tfile, pfile, cfile = out / "tasks.csv", out / "kappa_point.csv", out / "config.json"
    if not (tfile.exists() and cfile.exists()):
        sys.exit(f"No results in {out} (tasks.csv/config.json missing). Run without --plot_only first.")
    cfg = json.loads(cfile.read_text())
    tdf = read_tasks(tfile)
    if not pfile.exists():
        done = (tdf[tdf.rep == -1].groupby(["scope", "est"]).size()
                .unstack(fill_value=0).reindex(columns=list(EST), fill_value=0))
        print(done.to_string())
        sys.exit(f"Point estimates not complete yet (each scope needs 1 direct and "
                 f"{cfg['n_seeds']} federated tasks); nothing to plot.")
    point = pd.read_csv(pfile, dtype={"scope": str, "direct_idx": str, "federated_idx": str})
    scopes = [s for s in SCOPES if s in set(point.scope) and (not scopes_arg or s in scopes_arg)]
    point = point[point.scope.isin(scopes)].reset_index(drop=True)
    log.info(f"Plot only: results in {out} (n_seeds={cfg['n_seeds']}, rep_seeds={cfg['rep_seeds']}), "
             f"scopes {scopes}")
    print(add_gaps(point)[["scope", "direct", "federated", "gap", "gap_bits",
                           "federated_draw_sd"]].round(4).to_string(index=False))
    report(out, point, scopes, cfg["rep_seeds"], tdf)


# self-test

def selftest(ex, data, n_iters):
    """On the smallest client, via the worker pool: one copula draw and one direct
    estimate against _d_ensemble(n_seeds=1) / compute_theoretical_kappa in the main
    process with the same global seed; then one direct estimate on a fixed,
    reordered feature subset (as in stage 2), whose logged features must be the
    reference's selection mapped back to original column indices."""
    c = min(data, key=lambda k: len(data[k][1]))
    X, y = data[c]
    log.info(f"Self-test on client {c} ({len(y)} patients, {X.shape[1]} features, "
             f"greedy selection of {MAX_FEATURES})")
    ok, direct_sel = True, []
    for est in EST:
        t = make_task(est, c, -1, 0 if est == "federated" else -1, None, n_iters)
        t["gseed"] = [MASTER_SEED, 99]
        got = ex.submit(run_task, t).result()
        np.random.seed(t["gseed"])
        if est == "federated":
            ref = _d_ensemble(merge_payloads([json.loads(compute_statistics(X, y))]), df=DF,
                              max_features=MAX_FEATURES, n_iterations=n_iters, n_bins=N_BINS,
                              n_seeds=1, oversample=OVERSAMPLE, min_per_class=MIN_PER_CLASS)
            rk, rf = ref["kappa_max"], ref["selected_features"]
        else:
            ref = compute_theoretical_kappa(X.copy(), y, MAX_FEATURES, N_BINS, n_iters)
            rk, rf = ref["theoretical_kappa"], ref["selected_features"]
            direct_sel = [int(j) for j in rf]
        rf = ";".join(str(int(j)) for j in rf)
        same = "error" not in got and np.isclose(got["kappa"], rk, rtol=0, atol=1e-12) \
            and got["features"] == rf
        ok &= bool(same)
        log.info(f"  {est:9s}: reference {rk:.6f} [{rf}]  task "
                 f"{got.get('kappa', np.nan):.6f} [{got.get('features', got.get('error'))}]  "
                 f"-> {'OK' if same else 'MISMATCH'}")

    if len(direct_sel) >= 2:
        cols = direct_sel[::-1]                     # reordered, so positions != indices
        t = make_task("direct", c, -1, -1, cols, n_iters)
        t["gseed"] = [MASTER_SEED, 98]
        got = ex.submit(run_task, t).result()
        np.random.seed(t["gseed"])
        ref = compute_theoretical_kappa(X[:, cols].copy(), y, len(cols), N_BINS, n_iters)
        rk = ref["theoretical_kappa"]
        rf = ";".join(str(cols[int(j)]) for j in ref["selected_features"])
        same = "error" not in got and np.isclose(got["kappa"], rk, rtol=0, atol=1e-12) \
            and got["features"] == rf
        ok &= bool(same)
        log.info(f"  fixed set: reference {rk:.6f} [{rf}]  task "
                 f"{got.get('kappa', np.nan):.6f} [{got.get('features', got.get('error'))}]  "
                 f"-> {'OK' if same else 'MISMATCH'}")
    else:
        log.warning("  fixed set: skipped (direct selection returned fewer than 2 features)")
    log.info("Self-test passed." if ok else "Self-test FAILED, do not use the results.")
    return ok


# entry point

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_rep", type=int, default=1000,
                   help="max. half-sample replicates; 0 = point estimates only")
    p.add_argument("--hours", type=float, default=6.5, help="wall-clock budget for the whole run")
    p.add_argument("--n_iters", type=int, default=N_ITERATIONS, help="Crow-Watts binning iterations")
    p.add_argument("--n_seeds", type=int, default=20, help="copula draws S for the point estimates")
    p.add_argument("--rep_seeds", type=int, default=None,
                   help="copula draws per half-sample replicate (default: --n_seeds)")
    p.add_argument("--n_jobs", type=int, default=-1, help="worker processes (-1: all available cores)")
    p.add_argument("--scopes", nargs="+", default=None, help=f"subset of {SCOPES}")
    p.add_argument("--out", type=Path, default=HERE / "kappa_parallel" / str(DATA_BALANCE))
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--plot_only", action="store_true",
                   help="only rebuild CSVs and figures from the results in --out, compute nothing")
    p.add_argument("--client_labels", nargs="+", default=None, metavar="ID=NAME",
                   help="plot labels for client ids, e.g. 5=UKA (default: database names "
                        "from mydataloading.get_client_name)")
    a = p.parse_args()
    a.rep_seeds = a.rep_seeds or a.n_seeds
    if a.scopes and set(a.scopes) - set(SCOPES):
        sys.exit(f"Unknown scope(s) {sorted(set(a.scopes) - set(SCOPES))}; choose from {SCOPES}")
    scopes = [s for s in SCOPES if not a.scopes or s in a.scopes]
    out = a.out if a.out.is_absolute() else HERE / a.out
    n_jobs = a.n_jobs if a.n_jobs > 0 else len(os.sched_getaffinity(0))
    CLIENT_LABELS.update(client_labels(a.client_labels))
    log.info("Client labels: " + ", ".join(f"{c}={CLIENT_LABELS[c]}" for c in SCOPES[:-1]))
    if FONT != "Arial":
        log.warning(f"Arial not found, figures use {FONT}; PLOS accepts only Arial, Times or "
                    f"Symbol. Install Arial (e.g. conda install -c conda-forge mscorefonts, or "
                    f"copy arial*.ttf to ~/.fonts) and rerun with --plot_only.")
    if a.plot_only:
        plot_only(out, a.scopes)
        return

    try:
        data, names = load_all(writes=True)
    except PermissionError as e:
        sys.exit(f"No write permission for {e.filename}. prepare_data must create its split "
                 f"there once; make FLdata/ writable (e.g. sudo chown -R $USER FLdata).")
    fp = fingerprint(data)
    sizes = {s: sum(len(y) for _, y in scope_clients(data, s, -1)) for s in scopes}
    by_size = sorted(scopes, key=sizes.get, reverse=True)       # longest tasks first
    log.info(f"{len(data)} clients, {sizes.get('ALL', sum(sizes.values()))} patients, "
             f"{len(names)} features; DATA_BALANCE={DATA_BALANCE}, MAX_FEATURES={MAX_FEATURES}, "
             f"N_BINS={N_BINS}, n_iters={a.n_iters}, n_seeds={a.n_seeds}, "
             f"rep_seeds={a.rep_seeds}; {n_jobs} worker processes")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(n_jobs, mp_context=ctx, initializer=_init_worker,
                             initargs=(fp,)) as ex:
        if a.selftest:
            sys.exit(0 if selftest(ex, data, a.n_iters) else 1)

        out.mkdir(parents=True, exist_ok=True)
        cfg = dict(n_iters=a.n_iters, n_seeds=a.n_seeds, rep_seeds=a.rep_seeds,
                   max_features=MAX_FEATURES, n_bins=N_BINS, df=DF, oversample=OVERSAMPLE,
                   master_seed=MASTER_SEED, data=fp)
        cfile = out / "config.json"
        if cfile.exists() and json.loads(cfile.read_text()) != cfg:
            sys.exit(f"{out} holds results for other settings or data:\n{cfile.read_text()}\n"
                     f"Use another --out.")
        cfile.write_text(json.dumps(cfg, indent=1))
        tlog = TaskLog(out / "tasks.csv")

        # Stage 1: point estimates, feature selection on the full data
        todo = [make_task(e, s, -1, sd, None, a.n_iters) for s in by_size
                for e, sd in [("direct", -1)] + [("federated", i) for i in range(a.n_seeds)]
                if (-1, s, e, sd) not in tlog.done]
        if todo:
            log.info(f"Point estimates: {len(todo)} tasks (one per direct estimate and copula draw)")
            run_pool(ex, todo, tlog, 2 * n_jobs)
        point = build_point(tlog.read(), scopes, names, a.n_seeds)
        write_point(point, out / "kappa_point.csv")
        print(add_gaps(point)[["scope", "direct", "federated", "gap", "gap_bits",
                               "federated_draw_sd"]].round(4).to_string(index=False))
        if a.n_rep == 0:
            log.info(f"Point estimates and selected features in {out / 'kappa_point.csv'}. "
                     f"Boxplots and CIs need half-samples: run without --n_rep 0.")
            return

        # Stage 2: half-sampling with the fixed feature sets
        feats = fixed_features(point)
        per_rep = len(scopes) * (1 + a.rep_seeds)
        deadline = T0 + 3600 * a.hours

        def tasks():
            for rep in range(a.n_rep):
                for s in by_size:
                    for e, sd, cols in [("direct", -1, feats[s][0])] + \
                            [("federated", i, feats[s][1]) for i in range(a.rep_seeds)]:
                        if (rep, s, e, sd) not in tlog.done:
                            yield make_task(e, s, rep, sd, cols, a.n_iters)

        def rep_counts():
            return Counter(k[0] for k in tlog.done if k[0] >= 0 and k[1] in scopes)

        t_stage, n0 = time.time(), sum(rep_counts().values())

        def progress():
            cnt = rep_counts()
            full = sum(v == per_rep for v in cnt.values())
            rate = (sum(cnt.values()) - n0) / max(time.time() - t_stage, 1) / per_rep
            log.info(f"{full}/{a.n_rep} replicates complete, {rate * 3600:.0f} replicates/h, "
                     f"{(deadline - time.time()) / 3600:.2f} h budget left")

        log.info(f"Half-sampling: up to {a.n_rep} replicates x {per_rep} tasks within the "
                 f"{a.hours} h budget (fixed features, no reselection)")
        if not run_pool(ex, tasks(), tlog, 2 * n_jobs, deadline, progress):
            log.warning("Time budget reached; rerun with the same --out to add replicates.")
        progress()

    report(out, point, scopes, a.rep_seeds, tlog.read())


if __name__ == "__main__":
    main()