"""Summarise feature missingness across databases as a PLOS Digital Health compliant heatmap.

Reads all <db>_features_missingness[_<variant>].csv files in --indir and writes one
TIFF per variant (e.g. Fig_missingness_natural.tif, Fig_missingness_20.tif).
Panel A: time-binned features (database x bin). Panel B: static/derived features.
"""
import argparse
import glob
import io
import os
import re

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from general_config import DATA_DIR  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DB_LABELS = {"eicu": "eICU-CRD", "mimic": "MIMIC-IV", "hirid": "HiRID", "umc": "UMCdb", "uka": "UKA"}
FEATURE_LABELS = {"gender_concept_id": "Sex",
                  "Glasgow coma score eye opening": "GCS eye",
                  "Glasgow coma score motor": "GCS motor",
                  "Glasgow coma score verbal": "GCS verbal"}
FNAME = re.compile(r"^(?P<db>[^_]+)_features_missingness_?(?P<variant>.*)\.csv$", re.I)
BIN = r"^(.*)_(\d+)$"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                     "font.size": 8, "axes.linewidth": 0})


def label(f):
    f = FEATURE_LABELS.get(f, f)
    f = re.sub(r"\s*\[.*?\]", "", f).replace(" in Blood", "")
    return f[0].upper() + f[1:]


def load(indir):
    dfs = []
    for path in sorted(glob.glob(os.path.join(indir, "*_features_missingness*.csv"))):
        m = FNAME.match(os.path.basename(path))
        if not m:
            print(f"Skipping {path}: unexpected file name")
            continue
        df = pd.read_csv(path)
        df["db"] = m["db"].lower()
        df["variant"] = m["variant"] or "all"
        df["n"] = (df.n_missing / df.frac_missing.where(df.frac_missing > 0)).round().median()
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    parts = df.feature.str.extract(BIN)
    df["base"] = parts[0].fillna(df.feature)
    df["bin"] = pd.to_numeric(parts[1])
    df["pct"] = 100 * df.frac_missing
    return df


def plot(d, dbs, out, dpi):
    missing = [db for db in dbs if db not in set(d.db)]
    if missing:
        print(f"Warning: no file for {', '.join(missing)} in {out}")
    tv, st = d[d.bin.notna()], d[d.bin.isna()]
    bins = sorted(tv.bin.unique().astype(int))
    nb, gap = len(bins), 0.4
    x0 = {db: i * (nb + gap) for i, db in enumerate(dbs)}
    panels = [(tv, tv.groupby("base").pct.mean().sort_values().index.tolist(), bins),
              (st, st.groupby("base").pct.mean().sort_values().index.tolist(), [None])]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, min(0.2 * sum(len(p[1]) for p in panels) + 1.5, 8.75)),
                             gridspec_kw={"height_ratios": [len(p[1]) for p in panels]}, layout="constrained")
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("0.85")
    for ax, (sub, rows, cols) in zip(axes, panels):
        w = nb / len(cols)
        for db in dbs:
            s = sub[sub.db == db]
            if cols == [None]:
                m = s.set_index("base").pct.reindex(rows).to_numpy()[:, None]
            else:
                m = s.pivot_table(index="base", columns="bin", values="pct").reindex(index=rows, columns=cols).to_numpy()
            edges = x0[db] + w * np.arange(len(cols) + 1)
            mesh = ax.pcolormesh(edges, np.arange(len(rows) + 1), np.ma.masked_invalid(m),
                                 cmap=cmap, vmin=0, vmax=100, edgecolor="white", linewidth=0.3)
            for (i, j), v in np.ndenumerate(m):
                if np.isfinite(v):
                    ax.text(edges[j] + w / 2, i + 0.5, f"{v:.0f}", ha="center", va="center",
                            color="white" if v > 60 else "black")
        ax.set_xlim(0, x0[dbs[-1]] + nb)
        ax.set_ylim(len(rows), 0)
        ax.set_yticks(np.arange(len(rows)) + 0.5, [label(r) for r in rows])
        ax.tick_params(length=0)
        ax.set_xticks([])
    axes[0].set_xticks([x0[db] + b + 0.5 for db in dbs for b in range(nb)], [str(b) for _ in dbs for b in bins])
    axes[0].set_xlabel("Time bin")
    for db in dbs:
        n = d.loc[d.db == db, "n"]
        n = f"n = {n.iloc[0]:,.0f}" if len(n) else "no data"
        axes[0].text(x0[db] + nb / 2, -0.4, f"{DB_LABELS.get(db, db)}\n({n})", ha="center", va="bottom")
    for ax, p in zip(axes, "AB"):
        ax.text(-0.01, 1, p, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
    fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.03, aspect=50, label="Missing values (%)")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=2 / 72, facecolor="white")
    plt.close(fig)
    img = Image.open(buf).convert("RGB")
    img.save(out, compression="tiff_lzw", dpi=(dpi, dpi))
    img.save(os.path.splitext(out)[0] + ".png", dpi=(dpi, dpi))
    print(f"{out}: {img.size[0]}x{img.size[1]} px at {dpi} dpi")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(DATA_DIR))
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    df = load(a.indir)
    dbs = [k for k in DB_LABELS if k in set(df.db)] + sorted(set(df.db) - set(DB_LABELS))
    for variant, d in df.groupby("variant"):
        plot(d, dbs, os.path.join(a.outdir, f"Fig_missingness_{variant}.tif"), a.dpi)