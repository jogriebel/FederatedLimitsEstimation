"""Fig 1: kappa_max per site vs. federated kappa_max.
Fig 2: achieved test kappa (LL vs FL, 95% CI) per model, with kappa_max references.
Fig 3: FL test AUPRC without vs. with local boost (95% CI) per model.
Fig 4: test AUROC at 20% vs. natural positive ratio (95% CI), LL and FL in separate columns.
       AUROC is prevalence-invariant, so the two ratios can be compared directly.
Fig 5: kappa efficiency (achieved kappa / applicable kappa_max, %) per model, LL vs FL.
PLOS Digital Health: TIFF, LZW, RGB without alpha, 300 dpi, width <= 7.5 in, Arial 8-12 pt.
Figs 1-3 are suffixed with DATA_BALANCE from general_config (e.g. Fig1_20.tif);
Fig 4 always uses both ratios and is saved as Fig4_balance.tif."""
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

DATA_BALANCE = "natural"

# ---------------------------------------------------------------- inputs
DATA = Path("../FLresults")
LL_CSV = "local_metrics_{}_{}.csv"              # model, balance
FL_CSV = "FL_metrics_{}_client_{}_{}_test.csv"  # model, client, balance; one file per model and client; last row = final round
BALANCES = {"20": "20% positives", "natural": "Natural ratio"}
KMAX_FL_ALL = {
    "20": (0.8572152441688894, 0.8517, 0.8627),
    "natural": (0.8882081212354624, 0.8844, 0.8919),  # TODO
}  # global federated kappa_max (est, ci_low, ci_high) per balance
KMAX_FL = KMAX_FL_ALL[DATA_BALANCE]

SITES = {"eicu": "eICU-CRD", "hirid": "HiRID", "mimic": "MIMIC-IV", "uka": "UKA", "umc": "UMCdb"}
CLIENT_ID = {"eicu": 1, "mimic": 2, "umc": 3, "hirid": 4, "uka": 5}
MODELS = {"lr": "LR", "lstm": "LSTM", "NN": "MLP", "xgb": "XGBoost"}  # TODO: check file suffixes
REF_MODEL = "xgb"  # kappa_max is model-independent; taken from this model's LL file
COLORS = {"FL": dict(zip(SITES, ["#fee391", "#fec44f", "#fe9929", "#d95f0e", "#993404"])),  # one shade per site
          "LL": dict(zip(SITES, ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]))}
GLOBAL_LINE = "#8b0000"
KMAX = r"$\kappa_{\mathrm{max}}$"
BOOST = dict(edgecolor="white", hatch="////", lw=0)
NATURAL = dict(edgecolor="0.25", hatch="....", lw=0)  # Fig 4: natural-ratio bars
KAPPA_COL = {"LL": "test_actual_kappa", "FL": "actual_kappa"}
AUROC_COL = {"LL": "test_AUC", "FL": "AUC"}  # AUROC is stored as "AUC" in the metrics CSVs
AUROC_FLOOR = 0.5  # Fig 4: bars start at chance level

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.cal": "sans",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "hatch.linewidth": 0.6,
})


def load(template, model, balance=DATA_BALANCE):
    f = DATA / template.format(model, balance)
    return pd.read_csv(f).set_index("client_name").reindex(list(SITES)) if f.exists() else None


def load_fl(model, balance=DATA_BALANCE):
    """Last row of each per-client FL test CSV, indexed by site; None if no file exists."""
    rows = {}
    for s, c in CLIENT_ID.items():
        f = DATA / FL_CSV.format(model.upper(), c, balance)
        if f.exists():
            rows[s] = pd.read_csv(f).tail(1)
    return pd.concat(rows).droplevel(1).reindex(list(SITES)) if rows else None


def yerr(df, col):
    """Asymmetric error bars from <col>_ci_low/_ci_high; None if the CI columns are absent."""
    if f"{col}_ci_low" not in df:
        return None
    return np.vstack([df[col] - df[f"{col}_ci_low"], df[f"{col}_ci_high"] - df[col]])


def save_fig(fig, stem, suffix=DATA_BALANCE):
    """Save <stem>_<suffix>.tif (PLOS spec) and .png preview."""
    name = f"{stem}_{suffix}"
    fig.savefig(f"{name}.png")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    Image.open(buf).convert("RGB").save(f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    plt.close(fig)


def panel_style(ax, letter):
    ax.text(-0.1, 1.04, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)


ll = {m: load(LL_CSV, m) for m in MODELS}
fl = {m: load_fl(m) for m in MODELS}
ref = ll[REF_MODEL]
x = np.arange(len(SITES))
w = 0.38
err_kw = dict(capsize=2, error_kw=dict(lw=0.7))
xlim = (-0.5, len(SITES) - 0.5)

# ---------------------------------------------------------------- Fig 1
k = ref["theoretical_kappa"]
est = np.r_[k, KMAX_FL[0]]
lo = np.r_[ref.get("theoretical_kappa_ci_low", k * np.nan), KMAX_FL[1]]
hi = np.r_[ref.get("theoretical_kappa_ci_high", k * np.nan), KMAX_FL[2]]

fig, ax = plt.subplots(figsize=(5.2, 2.8), layout="constrained")
fc = [*COLORS["LL"].values(), GLOBAL_LINE]
bars = ax.bar(np.arange(6), est, 0.7, color=fc,
              yerr=[est - lo, hi - est], **err_kw)
labels = ax.bar_label(bars, [f"{v:.3f}" if np.isfinite(v) else "" for v in est],
                      label_type="center")
for t, c in zip(labels, fc):
    t.set_color("k" if np.mean(to_rgb(c)) > 0.6 else "white")
ax.set(xticks=np.arange(6), xticklabels=[*SITES.values(), "FL\n(global)"],
       ylabel=KMAX, ylim=(0, 1))
save_fig(fig, "Fig1")

# ---------------------------------------------------------------- Fig 2
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0), sharey=True, layout="constrained")
for ax, (m, name), letter in zip(axes.flat, MODELS.items(), "ABCD"):
    for i, (p, d) in enumerate([("FL", fl[m]), ("LL", ll[m])]):
        if d is not None:
            ax.bar(x + (i - 0.5) * w, d[KAPPA_COL[p]], w, color=[*COLORS[p].values()],
                   yerr=yerr(d, KAPPA_COL[p]), **err_kw)
    ax.hlines(k, x - 0.45, x + 0.45, colors="k", linestyles=":", lw=1)
    ax.axhline(KMAX_FL[0], color=GLOBAL_LINE, ls="--", lw=1)
    ax.set(title=name, xticks=x, xticklabels=SITES.values(), xlim=xlim, ylim=(0, 1.05))
    panel_style(ax, letter)
for ax in axes[:, 0]:
    ax.set_ylabel(r"Test $\kappa$")

fig.legend(handles=[Patch(color=COLORS["FL"]["mimic"], label="FL"),
                    Patch(color=COLORS["LL"]["mimic"], label="LL"),
                    Line2D([], [], color="k", ls=":", label=f"Local {KMAX}"),
                    Line2D([], [], color=GLOBAL_LINE, ls="--", label=f"Global FL {KMAX}")],
           loc="outside upper center", ncol=4, frameon=False)
save_fig(fig, "Fig2")

# ---------------------------------------------------------------- Fig 3
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0), sharey=True, layout="constrained")
for ax, (m, name), letter in zip(axes.flat, MODELS.items(), "ABCD"):
    d = fl[m]
    if d is not None:
        ax.bar(x - 0.5 * w, d["AUPRC"], w, color=[*COLORS["FL"].values()], yerr=yerr(d, "AUPRC"), **err_kw)
        ax.bar(x + 0.5 * w, d["boost_AUPRC"], w, facecolor=[*COLORS["FL"].values()], **BOOST,
               yerr=yerr(d, "boost_AUPRC"), **err_kw)
    ax.set(title=name, xticks=x, xticklabels=SITES.values(), xlim=xlim, ylim=(0, 1.05))
    panel_style(ax, letter)
for ax in axes[:, 0]:
    ax.set_ylabel("Test AUPRC")

fig.legend(handles=[Patch(color=COLORS["FL"]["mimic"], label="FL"),
                    Patch(facecolor=COLORS["FL"]["mimic"], **BOOST, label="FL + local boost")],
           loc="outside upper center", ncol=2, frameon=False)
save_fig(fig, "Fig3")

# ---------------------------------------------------------------- Fig 4
# Rows = models, columns = LL | FL. Per site: solid bar = 20% positives, dotted bar = natural ratio.
# Bars start at AUROC = 0.5 (chance), so bar length is the gain over a random classifier.
res = {b: {"LL": {m: load(LL_CSV, m, b) for m in MODELS},
           "FL": {m: load_fl(m, b) for m in MODELS}} for b in BALANCES}

fig, axes = plt.subplots(len(MODELS), 2, figsize=(7.5, 8.0), sharex=True, sharey=True,
                         layout="constrained")
letters = iter("ABCDEFGH")
for r, (m, name) in enumerate(MODELS.items()):
    for c, p in enumerate(["LL", "FL"]):
        ax = axes[r, c]
        shades = [*COLORS[p].values()]
        col = AUROC_COL[p]
        for i, b in enumerate(BALANCES):
            d = res[b][p][m]
            if d is None:
                continue
            style = dict(color=shades) if i == 0 else dict(facecolor=shades, **NATURAL)
            ax.bar(x + (i - 0.5) * w, d[col] - AUROC_FLOOR, w, bottom=AUROC_FLOOR,
                   yerr=yerr(d, col), **style, **err_kw)
        ax.set(xticks=x, xticklabels=SITES.values(), xlim=xlim, ylim=(AUROC_FLOOR, 1.0))
        if r == 0:
            ax.set_title({"LL": "Local learning (LL)", "FL": "Federated learning (FL)"}[p])
        if c == 0:
            ax.set_ylabel(f"{name}\nTest AUROC")
        panel_style(ax, next(letters))

fig.legend(handles=[Patch(color="0.6", label=BALANCES["20"]),
                    Patch(facecolor="0.6", **NATURAL, label=BALANCES["natural"])],
           loc="outside upper center", ncol=2, frameon=False)
save_fig(fig, "Fig4", suffix=DATA_BALANCE)

# ---------------------------------------------------------------- Fig 5
# Efficiency = achieved kappa / applicable kappa_max: local kappa_max for LL, global FL kappa_max for FL.
# CIs are the kappa CIs scaled by the point estimate of kappa_max.
kmax_ref = {"LL": k.to_numpy(), "FL": KMAX_FL[0]}
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0), sharey=True, layout="constrained")
for ax, (m, name), letter in zip(axes.flat, MODELS.items(), "ABCD"):
    for i, (p, d) in enumerate([("FL", fl[m]), ("LL", ll[m])]):
        if d is not None:
            s, e = 100 / kmax_ref[p], yerr(d, KAPPA_COL[p])
            ax.bar(x + (i - 0.5) * w, d[KAPPA_COL[p]] * s, w, color=[*COLORS[p].values()],
                   yerr=None if e is None else e * s, **err_kw)
    ax.axhline(100, color="k", ls="--", lw=1)
    ax.set(title=name, xticks=x, xticklabels=SITES.values(), xlim=xlim, ylim=(0, None))
    panel_style(ax, letter)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\kappa$ efficiency (%)")

fig.legend(handles=[Patch(color=COLORS["FL"]["mimic"], label="FL"),
                    Patch(color=COLORS["LL"]["mimic"], label="LL"),
                    Line2D([], [], color="k", ls="--", label=f"100% of {KMAX}")],
           loc="outside upper center", ncol=3, frameon=False)
save_fig(fig, "Fig5")