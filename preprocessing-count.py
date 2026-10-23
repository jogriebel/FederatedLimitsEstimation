"""Patient-flow counts for the preprocessing diagram.

Replays process_database() step by step with the unmodified pipeline functions
and records remaining / excluded patients per class after each step.
"""
import contextlib
import io

import pandas as pd
import numpy as np

import config_data as cfg
import preprocessing as pp  # <- change to the pipeline's module name

NUM = ["pos", "neg", "total", "excl_pos", "excl_neg", "excl_total"]


def counts(fm):
    v = fm["blood_transfused"].value_counts()
    return int(v.get(1, 0)), int(v.get(0, 0))


def n_ids(df):
    return df["person_id"].nunique() if len(df) else 0


def patient_flow(db, ratio):
    steps = []
    add = lambda name, pos, neg: steps.append((name, pos, neg))

    with contextlib.redirect_stdout(io.StringIO()):  # silence pipeline prints
        df = pp.load_data(db)
        df = pp.handle_outliers(df, cfg.VITAL_SIGNS)  # sets values to NaN only, no exclusions
        df = pp.compute_time_to_transfusion(df)

        n_pos = df.loc[df["first_tx_time"].notna(), "person_id"].nunique()
        n_neg = df["person_id"].nunique() - n_pos
        pre_icu_only = df.loc[(df["blood_transfused"] == 1) & df["first_tx_time"].isna(),
                              "person_id"].nunique()
        add("Loaded (pos = transfused after ICU admission)", n_pos, n_neg)

        df_tx = pp.filter_observation_window(df)
        add(f"Pos: >=1 obs in [-{cfg.OBSERVATION_START}h, -{cfg.OBSERVATION_END}h] window",
            n_ids(df_tx), n_neg)

        df_tx = pp.filter_min_observations(df_tx)
        n_pos = n_ids(df_tx)
        add(f"Pos: {cfg.MIN_OBSERVATION_LENGTH}-{cfg.OBSERVATION_LENGTH} obs in window", n_pos, n_neg)

        df_no_tx = pp.sample_non_transfused_patients(df, df_tx)
        add(f"Neg: >={cfg.MIN_OBSERVATION_LENGTH} obs in record", n_pos, n_ids(df_no_tx))

        fm = pp.create_feature_matrix(pd.concat([df_tx, df_no_tx], ignore_index=True))
        fm = pp.add_labels(fm, df)
        add(">=1 hemoglobin value", *counts(fm))

        fm = pp.filter_by_completeness(fm, cfg.COMPLETENESS_THRESHOLD)
        add(f"Completeness >= {cfg.COMPLETENESS_THRESHOLD}%", *counts(fm))

        if isinstance(ratio, float):
            fm = pp.balance_classes(fm, ratio)
            add(f"Class balancing to {ratio:.0%} positive", *counts(fm))

        metadata = pp.compute_patient_metadata(df, fm["person_id"])

    # Same outputs as process_database() + save_outputs()
    y = fm["blood_transfused"]
    X = fm.drop(columns=["person_id", "blood_transfused"])
    if isinstance(ratio, float):
        assert abs(y.mean() - ratio) < 0.01, f"Target ratio {ratio:.1%} not met! Got {y.mean():.1%}"
    pp.save_outputs(X, y, metadata, db, ratio)

    t = pd.DataFrame(steps, columns=["step", "pos", "neg"])
    t["total"] = t["pos"] + t["neg"]
    for c in ["pos", "neg", "total"]:
        t[f"excl_{c}"] = (-t[c].diff()).fillna(0).astype(int)
    t.insert(0, "db", db)
    return t, pre_icu_only


def run_version(ratio):
    """One loop of main(): all databases for one ratio setting."""
    suffix = cfg.get_ratio_suffix(ratio)
    print(f"\n{'=' * 60}\nVERSION: {suffix}\n{'=' * 60}")
    tables = []
    for db in cfg.DATABASES:
        np.random.seed(cfg.RANDOM_STATE)
        t, pre_icu = patient_flow(db, ratio)
        tables.append(t)
        print(f"\n=== {db} ({suffix}) ===")
        print(t.drop(columns="db").to_string(index=False))
        if pre_icu:
            print(f"  Note: {pre_icu} patients transfused only before ICU admission "
                  f"are in the negative pool")

    flow = pd.concat(tables, ignore_index=True)
    pooled = flow.groupby("step", sort=False)[NUM].sum().reset_index()
    print(f"\n=== Pooled ({suffix}) ===")
    print(pooled.to_string(index=False))

    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    flow.to_csv(cfg.OUTPUT_DIR / f"patient_flow_{suffix}.csv", index=False)
    pooled.to_csv(cfg.OUTPUT_DIR / f"patient_flow_pooled_{suffix}.csv", index=False)


if __name__ == "__main__":
    # Mirrors pp.main(): natural first, then balanced, so the global RNG stream
    # (used when sampling negative windows) is consumed in the same order.
    cfg.validate_config()
    run_version("natural")
    if isinstance(cfg.POSITIVE_RATIO, float):
        run_version(cfg.POSITIVE_RATIO)