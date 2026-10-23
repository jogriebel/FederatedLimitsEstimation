"""
One-off repair of the 'features' column in tasks.csv written by kappa_parallel.py
before the feature-index fix.

In stage 2 (half-samples, rep >= 0) the selection ran on X[:, cols] with cols the
fixed feature set of stage 1, and the logged indices were positions in that
subset (always 0..k-1) instead of original column indices. The kappa and CDR
values are correct; only the logged features are wrong. The fixed sets are in
kappa_point.csv (direct_idx / federated_idx per scope), so the original indices
are recovered exactly as cols[position]. Stage-1 rows (rep == -1) were correct
and are not touched.

Each half-sample row is classified before anything is changed:
  local      all indices < k and not a subset of the fixed set -> mapped back
  global     subset of the fixed set, some index >= k           -> already correct, kept
  ambiguous  both (only possible if the fixed set lies within 0..k-1)
             -> mapped back, assuming the row predates the fix (counted and reported)
  invalid    neither                                           -> kept, reported
So rows written by the fixed kappa_parallel.py are left alone even in a file
that mixes old and new rows.

Safety: refuses to run while tasks.csv was modified in the last 5 minutes (a
computing run may be appending; stop it first) unless --force; writes a backup
tasks_before_feature_fix.csv (never overwritten) and a marker
features_repaired.json, and refuses a second repair of the same directory
unless --force. The file is replaced atomically. All other columns are kept
byte-for-byte as text.

Usage:
    python fix_tasks_features.py kappa_parallel/natural kappa_parallel/20 --dry_run
    python fix_tasks_features.py kappa_parallel/natural kappa_parallel/20
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

MARKER = "features_repaired.json"
BACKUP = "tasks_before_feature_fix.csv"
QUIET_SECONDS = 300


def idx_list(s):
    return [int(j) for j in str(s).split(";") if j not in ("", "nan")]


def classify(pos, cols):
    k, fixed = len(cols), set(cols)
    local = all(0 <= j < k for j in pos)
    glob = set(pos) <= fixed
    if not pos:
        return "empty"
    if local and glob:
        return "ambiguous"
    return "local" if local else "global" if glob else "invalid"


def repair(out, force, dry_run):
    tf, pf, mf = out / "tasks.csv", out / "kappa_point.csv", out / MARKER
    print(f"\n== {out}")
    if not (tf.exists() and pf.exists()):
        print(f"   skipped: {tf.name} or {pf.name} missing")
        return False
    if mf.exists() and not force:
        print(f"   skipped: already repaired ({mf.name}: {mf.read_text().strip()}); "
              f"--force to repair again")
        return True
    age = time.time() - tf.stat().st_mtime
    if age < QUIET_SECONDS and not force and not dry_run:
        print(f"   refused: {tf.name} was modified {age:.0f} s ago, a computing run may still "
              f"be appending. Stop it (or wait), then rerun; --force to override.")
        return False

    point = pd.read_csv(pf, dtype=str, keep_default_na=False)
    cols = {(r["scope"], e): idx_list(r[f"{e}_idx"])
            for _, r in point.iterrows() for e in ("direct", "federated")}

    df = pd.read_csv(tf, dtype=str, keep_default_na=False, on_bad_lines="skip")
    n_read = len(df)
    df = df[df["seconds"].fillna("") != ""].copy()          # half-written last line
    if len(df) < n_read:
        print(f"   dropped {n_read - len(df)} incomplete line(s) (as read_tasks does)")

    half = df["rep"].astype(int) >= 0
    counts, unknown, new = Counter(), Counter(), df["features"].copy()
    for i in df.index[half]:
        key = (df.at[i, "scope"], df.at[i, "est"])
        if key not in cols:
            unknown[key] += 1
            continue
        c, pos = cols[key], idx_list(df.at[i, "features"])
        kind = classify(pos, c)
        counts[kind] += 1
        if kind in ("local", "ambiguous"):
            new.at[i] = ";".join(str(c[j]) for j in pos)

    print(f"   {int(half.sum())} half-sample rows: " +
          ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if counts["ambiguous"]:
        print(f"   note: {counts['ambiguous']} ambiguous rows (fixed set within 0..k-1) were "
              f"mapped back on the assumption that they predate the fix")
    if counts["invalid"]:
        print(f"   WARNING: {counts['invalid']} rows fit neither the old nor the new format; "
              f"left unchanged, check them")
    if unknown:
        print(f"   WARNING: no fixed features in {pf.name} for {dict(unknown)}; left unchanged")

    changed = int((new != df["features"]).sum())
    df["features"] = new

    # check: every half-sample row should now use exactly its fixed set (as a set)
    ok = Counter()
    for i in df.index[half]:
        key = (df.at[i, "scope"], df.at[i, "est"])
        if key in cols:
            ok[set(idx_list(df.at[i, "features"])) == set(cols[key])] += 1
    print(f"   after repair: {ok[True]} rows use exactly the fixed set, "
          f"{ok[False]} differ (reselection or invalid)")

    if dry_run:
        print(f"   dry run: {changed} rows would change, nothing written")
        return True
    bk = out / BACKUP
    if not bk.exists():
        bk.write_bytes(tf.read_bytes())
        print(f"   backup -> {bk}")
    tmp = tf.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(tf)
    mf.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), rows_changed=changed,
                                  **{k: v for k, v in counts.items()}), indent=1))
    print(f"   {changed} rows changed -> {tf}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", nargs="+", type=Path,
                   help="result directories of kappa_parallel.py (with tasks.csv, kappa_point.csv)")
    p.add_argument("--dry_run", action="store_true", help="report only, write nothing")
    p.add_argument("--force", action="store_true",
                   help="repair even if the marker exists or tasks.csv was just modified")
    a = p.parse_args()
    ok = all([repair(o.resolve(), a.force, a.dry_run) for o in a.out])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()