"""Selected kappa_max features per client and on all data, LL vs FL.
Cell = selection position (1 = first selected, darker = earlier); n = number of clients selecting the feature."""
import re
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from general_config import DATA_BALANCE, ROOT  # noqa: E402

CSV = ROOT / "federated" / "kappa" / "kappa_parallel" / DATA_BALANCE / "kappa_point.csv"
SITES = {"1": "eICU-CRD", "2": "MIMIC-IV", "3": "UMCdb", "4": "HiRID", "5": "UKA", "ALL": "All"}
PANELS = {"LL": ("direct_features", "#3A72B2"), "FL": ("federated_features", "#843414")}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
                     "font.size": 8})


def short(name):
    """'Hemoglobin [Mass/volume] in Blood_1' -> 'Hemoglobin_1'."""
    return re.sub(r" \[[^\]]*\] in [^_]+", "", name)


df = pd.read_csv(CSV, dtype={"scope": str})
sel = pd.DataFrame([(p, SITES[row["scope"]], short(f), r)
                    for p, (col, _) in PANELS.items() for _, row in df.iterrows()
                    for r, f in enumerate(row[col].split(";"), 1)],
                   columns=["paradigm", "site", "feature", "pos"])
n_pos = sel["pos"].max()
clients = sel[sel["site"] != "All"]
# Rows: selected by most clients first (LL + FL), ties by mean position
rows = (clients.groupby("feature")["pos"].agg(["size", "mean"])
        .reindex(sel["feature"].unique()).fillna({"size": 0, "mean": n_pos + 1})
        .sort_values(["size", "mean"], ascending=[False, True]).index)

fig, axes = plt.subplots(1, 2, figsize=(5.8, 0.16 * len(rows) + 0.9), sharey=True,
                         gridspec_kw={"wspace": 0.3})
for ax, (p, (_, color)), label in zip(axes, PANELS.items(), "AB"):
    pos = (sel[sel["paradigm"] == p].pivot(index="feature", columns="site", values="pos")
           .reindex(index=rows, columns=SITES.values()))
    im = ax.imshow((n_pos + 1 - pos).fillna(0), cmap=LinearSegmentedColormap.from_list("", ["white", color]),
              vmin=0, vmax=n_pos, aspect="auto")
    for (i, j), v in np.ndenumerate(pos.values):
        if v == v:
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v <= 3 else "black")
    n = pos.drop(columns="All").notna().sum(axis=1)
    for i, v in enumerate(n):
        ax.text(len(SITES) - 0.1, i, v if v else "", ha="left", va="center")
    ax.text(len(SITES) - 0.1, -0.8, "n", ha="left", va="bottom", style="italic")
    ax.set_xticks(range(len(SITES)), SITES.values(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(len(SITES) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="0.85", lw=0.5)
    ax.axvline(len(SITES) - 1.5, color="black", lw=0.8)
    ax.tick_params(which="both", length=0)
    ax.set_title(p, fontsize=9, pad=10)
    ax.text(-0.02, 1.0, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, weight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.1, aspect=30)
    cb.set_ticks(range(n_pos, 0, -1), labels=range(1, n_pos + 1))
    cb.set_label("Selection position")
    cb.ax.tick_params(length=2)
    cb.outline.set_linewidth(0.5)

stem = f"FigFeatures_{DATA_BALANCE}"
fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
buf = BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.03)
Image.open(buf).convert("RGB").save(f"{stem}.tif", compression="tiff_lzw", dpi=(300, 300))