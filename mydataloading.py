"""Data loading and preprocessing module."""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional

from general_config import DATA_DIR, CLIENT_NAMES, RANDOM_STATE, X_PATTERN, Y_PATTERN, DATA_BALANCE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPLIT_DIR = DATA_DIR / "splits" / DATA_BALANCE
SPLIT_PARTS = ("train", "val", "test")


def get_client_name(client_id: int) -> str:
    """Map client ID to name."""
    return CLIENT_NAMES.get(client_id, f"client_{client_id}")


def _extract_labels(y) -> np.ndarray:
    """Convert labels to numpy array."""
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0].values
    elif isinstance(y, pd.Series):
        return y.values
    return np.asarray(y)


def _impute_and_scale(
    X_train: np.ndarray, *arrays: np.ndarray, feature_names: list
) -> tuple[list[pd.DataFrame], SimpleImputer, StandardScaler]:
    """Fit imputer and scaler on X_train, transform all provided arrays."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()

    X_train_out = pd.DataFrame(
        scaler.fit_transform(imputer.fit_transform(X_train)), columns=feature_names
    )
    others = [
        pd.DataFrame(scaler.transform(imputer.transform(a)), columns=feature_names)
        for a in arrays
    ]
    return [X_train_out, *others], imputer, scaler


def _split_paths(client_name: str) -> dict:
    """File paths of the persisted split for one client."""
    return {
        part: (
            SPLIT_DIR / f"{client_name}_X_{part}.parquet",
            SPLIT_DIR / f"{client_name}_y_{part}.parquet",
        )
        for part in SPLIT_PARTS
    }


def _load_or_create_split(X: pd.DataFrame, y: pd.Series, client_name: str) -> dict:
    """Return {part: (X_df, y_series)} for a fixed 70/10/20 train/val/test split.

    Created once per client (unscaled, after trend features) and stored in
    SPLIT_DIR; reused afterwards by both FL and LL. Delete the client's files
    in SPLIT_DIR to force regeneration.
    """
    paths = _split_paths(client_name)

    if all(xp.exists() and yp.exists() for xp, yp in paths.values()):
        logger.info(f"Loading existing split for {client_name} from {SPLIT_DIR}")
        split = {
            part: (pd.read_parquet(xp), pd.read_parquet(yp).iloc[:, 0])
            for part, (xp, yp) in paths.items()
        }
        n_cached = sum(len(Xp) for Xp, _ in split.values())
        if split["train"][0].columns.tolist() != X.columns.tolist() or n_cached != len(X):
            raise ValueError(
                f"Stored split for {client_name} does not match current data "
                f"({n_cached} vs {len(X)} rows or differing columns). "
                f"Delete {SPLIT_DIR}/{client_name}_* to regenerate."
            )
        return split

    # Same two-stage procedure (and seed) as the previous FL split:
    # 70% train / 30% temp, then temp -> 2/3 test (20%) and 1/3 val (10%).
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=1 / 3, random_state=RANDOM_STATE, stratify=y_temp
    )
    split = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for part, (xp, yp) in paths.items():
        split[part][0].to_parquet(xp)
        split[part][1].to_frame().to_parquet(yp)
    logger.info(f"Created and stored split for {client_name} in {SPLIT_DIR}")

    return split


def prepare_data(client_id: int) -> tuple:
    """Load, process and split data for a given client."""
    client_name = get_client_name(client_id)
    logger.info(f"Loading data for client {client_id} ({client_name})")

    x_path = DATA_DIR / X_PATTERN.format(client=client_name)
    y_path = DATA_DIR / Y_PATTERN.format(client=client_name)

    logger.info(f"Loading from {x_path}")

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)

    logger.info(f"Original shape: {X.shape}")

    if any("Hemoglobin" in str(c) for c in X.columns):
        X, y = add_hemoglobin_trend_features(X, y)
        logger.info(f"After trend features: {X.shape}")

    feature_names = X.columns.tolist()
    logger.info(f"Final feature count: {len(feature_names)}")
    
    logger.info(f"Final features: {feature_names}")

    missingness = pd.DataFrame(
        {"n_missing": X.isna().sum(), "frac_missing": X.isna().mean()}
    )
    missingness.index.name = "feature"
    logger.info(f"Missingness before imputation ({client_name}):\n{missingness.to_string()}")
    missingness.to_csv(DATA_DIR / f"{client_name}_features_missingness_{DATA_BALANCE}.csv")

    # y may be a DataFrame or an ndarray (if trend features dropped rows)
    y = pd.Series(np.ravel(_extract_labels(y)), index=X.index, name="label")

    split = _load_or_create_split(X, y, client_name)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = (split[p] for p in SPLIT_PARTS)

    scaled, _, _ = _impute_and_scale(
        X_train.values, X_test.values, X_val.values, feature_names=feature_names
    )
    X_train, X_test, X_val = scaled

    if X_train.shape[1] != len(feature_names):
        raise ValueError(f"Shape mismatch: {X_train.shape[1]} vs {len(feature_names)}")

    logger.info(
        f"Client {client_id} ready: {len(X_train)} train, "
        f"{len(X_test)} test, {len(X_val)} val"
    )

    return (
        X_train, _extract_labels(y_train),
        X_test, _extract_labels(y_test),
        X_val, _extract_labels(y_val),
        feature_names,
    )


def add_hemoglobin_trend_features(
    df: pd.DataFrame,
    y: np.ndarray,
    hgb_base_name: str = "Hemoglobin [Mass/volume] in Blood",
    n_timepoints: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute hemoglobin trend features from temporal data.

    Args:
        df: DataFrame with columns like 'Hemoglobin [Mass/volume] in Blood_1', '_2', etc.
        y: Target array to filter alongside X.
        hgb_base_name: Base name of hemoglobin feature.
        n_timepoints: Number of timepoints. If None, auto-detect.

    Returns:
        Tuple of (DataFrame with added trend features, filtered y).
    """
    df = df.copy()

    hgb_cols = sorted(
        [c for c in df.columns if c.startswith(hgb_base_name)],
        key=lambda x: int(x.split("_")[-1]) if "_" in x else 0,
    )

    if not hgb_cols:
        logger.info(f"No hemoglobin columns found — skipping trend features.")
        return df, y

    if n_timepoints is None:
        n_timepoints = len(hgb_cols)

    logger.info(f"Found {len(hgb_cols)} hemoglobin timepoints: {hgb_cols[:3]}...{hgb_cols[-1]}")

    hgb_values = df[hgb_cols].values

    valid_mask = ~np.isnan(hgb_values).all(axis=1)
    n_removed = (~valid_mask).sum()
    if n_removed > 0:
        logger.info(f"Removing {n_removed} patients with no hemoglobin values")
        df = df[valid_mask].reset_index(drop=True)
        hgb_values = hgb_values[valid_mask]
        y = np.asarray(y)[valid_mask]

    df["hgb_min"] = np.nanmin(hgb_values, axis=1)
    df["hgb_range"] = np.nanmax(hgb_values, axis=1) - np.nanmin(hgb_values, axis=1)
    df.loc[(~np.isnan(hgb_values)).sum(axis=1) < 2, "hgb_range"] = np.nan
    df["hgb_last"] = pd.DataFrame(hgb_values).ffill(axis=1).iloc[:, -1].to_numpy()
    df["hgb_n_obs"] = (~np.isnan(hgb_values)).sum(axis=1)

    x = np.arange(n_timepoints)

    def compute_slope(row):
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            return 0.0
        return np.polyfit(x[valid], row[valid], 1)[0]

    df["hgb_slope"] = np.apply_along_axis(compute_slope, 1, hgb_values)

    logger.info("Added 5 hemoglobin trend features")

    return df, y