"""Blood transfusion prediction data preprocessing pipeline."""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple
import config_data as cfg

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

def load_data(db: str) -> pd.DataFrame:
    """Load raw data and perform initial preprocessing."""
    path = f"{cfg.DATA_DIR}/ds_extracted_correct{cfg.PROJECT_NAME}_{db}.parquet"
    df = pd.read_parquet(path)
    
    df = df.sort_values(by=["person_id", "visit_occurrence_id", "time"])
    df = df.drop(columns="visit_occurrence_id")
    df.rename(columns={'Packed blood cell transfusion': 'blood_transfused'}, inplace=True)
    df["blood_transfused"] = df["blood_transfused"].fillna(0)
    
    meta_cols = ['begin_icu', 'end_icu', 'los_icu', 'los_hospital']
    cols = ['person_id', 'time', 'blood_transfused'] + cfg.VITAL_SIGNS + cfg.STATIC_FEATURES + meta_cols
    return df[[c for c in cols if c in df.columns]]


def handle_outliers(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Replace outliers with NaN using percentile-based approach."""
    df = df.copy()
    Q1, Q3 = df[variables].quantile(0.02), df[variables].quantile(0.98)
    
    df.loc[df['weight'] > 150, 'weight'] = np.nan
    df.loc[df['height'] > 220, 'height'] = np.nan
    
    for feat in variables:
        mask = (df[feat] < Q1[feat]) | (df[feat] > Q3[feat])
        df.loc[mask, feat] = np.nan
    return df


# =============================================================================
# TIME-BASED PROCESSING
# =============================================================================

def compute_time_to_transfusion(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hours until first transfusion (after ICU admission) for each patient."""
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    if 'begin_icu' in df.columns:
        df['begin_icu'] = pd.to_datetime(df['begin_icu'])
    df = df.sort_values(by=['person_id', 'time'])
    
    # Only consider transfusions AFTER ICU admission
    tx_mask = (df['blood_transfused'] == 1)
    if 'begin_icu' in df.columns:
        tx_mask = tx_mask & (df['time'] >= df['begin_icu'])
    
    first_tx = df[tx_mask].groupby('person_id')['time'].first()
    first_tx = first_tx.reset_index().rename(columns={'time': 'first_tx_time'})
    
    df = df.merge(first_tx, on='person_id', how='left')
    df['hours_to_tx'] = (df['time'] - df['first_tx_time']).dt.total_seconds() / 3600
    
    import sys
    h = df['hours_to_tx'].dropna()
    print("transfused rows, whole-hour share:", (h % 1 == 0).mean(), file=sys.__stdout__)

    w = h[(h > -6) & (h <= -2)]
    print("in-window, whole-hour share:", (w % 1 == 0).mean(), file=sys.__stdout__)
    print("fractional parts:", (w % 1).round(2).value_counts().head(10), file=sys.__stdout__)
    
    return df


def filter_observation_window(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to observation window (e.g., -5h to -2h before event)."""
    mask = (df['hours_to_tx'] <= -cfg.OBSERVATION_END) & \
           (df['hours_to_tx'] > -cfg.OBSERVATION_START)
    filtered = df[mask].copy()
    
    print(f"  Observation window: [{filtered['hours_to_tx'].min():.1f}h, {filtered['hours_to_tx'].max():.1f}h]")
    return filtered


def filter_min_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter patients who have at least MIN_OBSERVATION_LENGTH observations."""
    obs_counts = df.groupby('person_id').size()
    valid_patients = obs_counts[(obs_counts >= cfg.MIN_OBSERVATION_LENGTH) & 
                                (obs_counts <= cfg.OBSERVATION_LENGTH)].index
    filtered = df[df['person_id'].isin(valid_patients)].copy()
    
    removed = df['person_id'].nunique() - len(valid_patients)
    print(f"  Removed {removed} patients with obs outside [{cfg.MIN_OBSERVATION_LENGTH}, {cfg.OBSERVATION_LENGTH}]")
    return filtered


# =============================================================================
# NON-TRANSFUSED PATIENT SAMPLING
# =============================================================================

def sample_non_transfused_patients(df_all: pd.DataFrame, df_transfused: pd.DataFrame) -> pd.DataFrame:
    """Sample observation windows for non-transfused patients matching transfused distribution."""
    # Get observation counts from transfused patients
    tx_patients = df_transfused['person_id'].unique()
    tx_counts = df_transfused.groupby('person_id').size()
    valid_counts = tx_counts[tx_counts >= 1]
    
    if len(valid_counts) == 0:
        return pd.DataFrame()
    
    # Get non-transfused patients not already in observation set
    excluded = set(tx_patients) | set(df_transfused['person_id'])
    df_no_tx = df_all[~df_all['person_id'].isin(excluded) & df_all['first_tx_time'].isna()].copy()
    
    if df_no_tx.empty:
        return pd.DataFrame()
    
    # Sample observation lengths from transfused distribution
    unique_ids = df_no_tx['person_id'].unique()
    sampled_lengths = np.random.choice(valid_counts.values, size=len(unique_ids), replace=True)
    length_map = dict(zip(unique_ids, sampled_lengths))
    df_no_tx['target_length'] = df_no_tx['person_id'].map(length_map)
    
    # Select beginning slice for each patient
    result = []
    for pid, group in df_no_tx.groupby('person_id'):
        n = min(len(group), int(group['target_length'].iloc[0]))
        if n >= cfg.MIN_OBSERVATION_LENGTH:
            result.append(group.iloc[:n])
    
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame()


# =============================================================================
# FEATURE MATRIX CREATION
# =============================================================================

def create_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transform time-series data into feature matrix (one row per patient)."""
    df = df.sort_values(by=["person_id", "time"]).copy()
    
    # Static features (first observation per patient)
    static = df[['person_id'] + cfg.STATIC_FEATURES].groupby('person_id').first().reset_index()
    
    # Time-varying features (pivot to wide format)
    time_cols = ['person_id'] + cfg.VITAL_SIGNS
    df_time = df[time_cols].copy()
    df_time = df_time.groupby('person_id').head(cfg.OBSERVATION_LENGTH)
    df_time['timepoint'] = df_time.groupby('person_id').cumcount() + 1
    
    pivoted = df_time.pivot(index='person_id', columns='timepoint', values=cfg.VITAL_SIGNS)
    pivoted.columns = [f'{col}_{t}' for col, t in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    merge = pivoted.merge(static, on='person_id', how='left')
    #Exclude patients with no hemoglobin values
    hgb_cols = [c for c in merge.columns if c.startswith("Hemoglobin")]
    if len(hgb_cols) > 0:
        valid_mask = ~merge[hgb_cols].isna().all(axis=1)
        n_removed = (~valid_mask).sum()
        if n_removed > 0:
            print(f"Removing {n_removed} patients with no hemoglobin values")
            merge = merge[valid_mask].reset_index(drop=True)
    return merge


def add_labels(feature_matrix: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    """Add transfusion labels to feature matrix."""
    labels = df_all.groupby('person_id').agg({
        'blood_transfused': 'max',
        'first_tx_time': 'first'
    }).reset_index()
    
    labels['blood_transfused'] = labels['first_tx_time'].notna().astype(int)
    return feature_matrix.merge(labels[['person_id', 'blood_transfused']], on='person_id', how='left')


# =============================================================================
# CLASS BALANCING 
# =============================================================================

def calculate_observation_length(row: pd.Series) -> int:
    """Calculate the observation window length for a patient row."""
    max_tp = 0
    for col in row.index:
        if '_' in col and col.split('_')[-1].isdigit():
            if pd.notna(row[col]):
                max_tp = max(max_tp, int(col.split('_')[-1]))
    return max_tp


def balance_classes(df: pd.DataFrame, target_ratio: float) -> pd.DataFrame:
    """
    Balance dataset to target positive ratio while preserving observation length distributions.
    
    Strategy:
    - If too many positives: downsample positives first, then downsample negatives
    - If too many negatives: only downsample negatives (matching obs length distribution)
    """
    df = df.copy()
    
    # Calculate observation length for ALL patients upfront
    df['obs_length'] = df.apply(calculate_observation_length, axis=1)
    
    pos_df = df[df['blood_transfused'] == 1].copy()
    neg_df = df[df['blood_transfused'] == 0].copy()
    
    current_pos_count = len(pos_df)
    current_neg_count = len(neg_df)
    total_count = current_pos_count + current_neg_count
    pos_ratio = current_pos_count / total_count
    
    print(f"    Before balancing: {current_pos_count} pos, {current_neg_count} neg, ratio={pos_ratio:.3f}")
    
    # Step 1: If pos_ratio > target, downsample positives first
    if pos_ratio > target_ratio:
        # Calculate how many positives we need IF we keep all negatives
        # pos / (pos + neg) = target_ratio
        # pos = target_ratio * (pos + neg)
        # pos = target_ratio * pos + target_ratio * neg
        # pos * (1 - target_ratio) = target_ratio * neg
        # pos = neg * target_ratio / (1 - target_ratio)
        target_pos_count = int(current_neg_count * target_ratio / (1 - target_ratio))
        
        # But we might not have that many, so cap it
        target_pos_count = min(target_pos_count, current_pos_count)
        
        pos_df = pos_df.sample(n=target_pos_count, random_state=42)
        current_pos_count = len(pos_df)
        
        print(f"    After pos downsampling: {current_pos_count} pos, {current_neg_count} neg")
    
    # Step 2: Calculate required negatives for exact target ratio
    # neg / (pos + neg) = (1 - target_ratio)
    # neg = pos * (1 - target_ratio) / target_ratio
    target_neg_count = int(current_pos_count * (1 - target_ratio) / target_ratio)
    
    if current_neg_count > target_neg_count:
        print(f"    Target neg count: {target_neg_count} (have {current_neg_count})")
        
        # Get distribution of observation lengths from CURRENT positive patients
        pos_length_dist = pos_df['obs_length'].value_counts(normalize=True)
        
        # Sample negatives to match positive observation length distribution
        sampled_neg_list = []
        used_indices = set()
        
        for length in sorted(pos_length_dist.index):
            proportion = pos_length_dist[length]
            target_count = int(proportion * target_neg_count)
            
            available = neg_df[(neg_df['obs_length'] == length) & (~neg_df.index.isin(used_indices))]
            
            if len(available) >= target_count:
                sampled = available.sample(n=target_count, random_state=42)
                sampled_neg_list.append(sampled)
                used_indices.update(sampled.index)
            else:
                sampled_neg_list.append(available)
                used_indices.update(available.index)
                shortfall = target_count - len(available)
                
                if shortfall > 0:
                    longer = neg_df[(neg_df['obs_length'] > length) & (~neg_df.index.isin(used_indices))]
                    
                    if len(longer) > 0:
                        n_additional = min(shortfall, len(longer))
                        additional = longer.sample(n=n_additional, random_state=42).copy()
                        
                        for tp in range(length + 1, cfg.OBSERVATION_LENGTH + 1):
                            cols_to_null = [c for c in additional.columns if c.endswith(f'_{tp}')]
                            for col in cols_to_null:
                                additional[col] = np.nan
                        
                        additional['obs_length'] = length
                        sampled_neg_list.append(additional)
                        used_indices.update(additional.index)
        
        if sampled_neg_list:
            neg_df = pd.concat(sampled_neg_list, ignore_index=True)
        else:
            neg_df = pd.DataFrame(columns=df.columns)
    
    # Combine and clean up
    result = pd.concat([pos_df, neg_df], ignore_index=True)
    result = result.drop(columns=['obs_length'])
    
    final_pos = result['blood_transfused'].sum()
    final_total = len(result)
    print(f"    Final: {int(final_pos)} pos, {final_total - int(final_pos)} neg, ratio={final_pos/final_total:.3f}")
    
    return result.sort_values('person_id')


# =============================================================================
# DATA QUALITY FILTERING
# =============================================================================

def filter_by_completeness(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Filter rows by data completeness percentage."""
    if threshold <= 0:
        return df
    
    feature_cols = [c for c in df.columns if c not in ['person_id', 'blood_transfused']]
    completeness = df[feature_cols].notna().mean(axis=1) * 100
    return df[completeness >= threshold].copy()


def validate_timepoints(df: pd.DataFrame) -> None:
    """Validate that feature matrix has expected timepoints."""
    timepoint_cols = [c for c in df.columns if c.split('_')[-1].isdigit()]
    max_timepoint = max(int(c.split('_')[-1]) for c in timepoint_cols) if timepoint_cols else 0
    
    expected = cfg.OBSERVATION_LENGTH
    print(f"  Expected timepoints: 1 to {expected}")
    print(f"  Actual max timepoint: {max_timepoint}")
    
    if max_timepoint != expected:
        print(f"  WARNING: Timepoint mismatch!")


def compute_patient_metadata(df_raw: pd.DataFrame, final_patient_ids: pd.Index) -> pd.DataFrame:
    """Compute patient metadata for the final cohort."""
    patient_info = df_raw.groupby('person_id').first().reset_index()
    patient_info = patient_info[patient_info['person_id'].isin(final_patient_ids)].copy()
    
    meta = patient_info[['person_id']].copy()
    
    if 'los_icu' in patient_info.columns:
        meta['los_icu'] = patient_info['los_icu'].values
    if 'los_hospital' in patient_info.columns:
        meta['los_hospital'] = patient_info['los_hospital'].values
    
    meta['blood_transfused'] = patient_info['first_tx_time'].notna().astype(int).values
    
    if 'begin_icu' in patient_info.columns:
        hours_to_tx = (patient_info['first_tx_time'] - patient_info['begin_icu']).dt.total_seconds() / 3600
        meta['hours_icu_to_tx'] = hours_to_tx.values
        
        obs_window_start = patient_info['first_tx_time'] - pd.Timedelta(hours=cfg.OBSERVATION_START)
        hours_before_icu = (patient_info['begin_icu'] - obs_window_start).dt.total_seconds() / 3600
        meta['hours_obs_before_icu'] = hours_before_icu.clip(lower=0, upper=cfg.OBSERVATION_LENGTH).values
    
    return meta


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_database(db: str, positive_ratio) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Process a single database through the full pipeline."""
    print(f"\n{'='*60}\nProcessing: {db}\n{'='*60}")
    
    # Load and clean data
    df = load_data(db)
    print(f"  Loaded {len(df):,} rows, {df['person_id'].nunique():,} patients")
    
    df = handle_outliers(df, cfg.VITAL_SIGNS)
    df = compute_time_to_transfusion(df)
    df_with_times = df.copy()
    
    # Get transfused patient observations
    df_tx = filter_observation_window(df)
    df_tx = filter_min_observations(df_tx)
    print(f"  Transfused patients in window: {df_tx['person_id'].nunique():,}")
    
    # Get non-transfused patient observations
    df_no_tx = sample_non_transfused_patients(df, df_tx)
    print(f"  Non-transfused patients sampled: {df_no_tx['person_id'].nunique():,}")
    
    # Combine and create feature matrix
    observations = pd.concat([df_tx, df_no_tx], ignore_index=True)
    feature_matrix = create_feature_matrix(observations)
    feature_matrix = add_labels(feature_matrix, df)
    
    # Validate timepoints
    validate_timepoints(feature_matrix)
    
    # Filter by completeness
    feature_matrix = filter_by_completeness(feature_matrix, cfg.COMPLETENESS_THRESHOLD)
    print(f"  After completeness filter: {len(feature_matrix):,} patients")
    
    # Balance classes if specified
    if isinstance(positive_ratio, float):
        feature_matrix = balance_classes(feature_matrix, positive_ratio)
    
    # Compute patient metadata for final cohort
    final_patient_ids = feature_matrix['person_id']
    metadata = compute_patient_metadata(df_with_times, final_patient_ids)
    
    # Split features and labels
    y = feature_matrix['blood_transfused']
    X = feature_matrix.drop(columns=['person_id', 'blood_transfused'])
    
    # Final ratio check
    final_ratio = y.mean()
    print(f"  Final: {len(X):,} patients, {final_ratio:.1%} positive")
    if isinstance(positive_ratio, float):
        assert abs(final_ratio - positive_ratio) < 0.01, \
            f"Target ratio {positive_ratio:.1%} not met! Got {final_ratio:.1%}"
    
    return X, y, metadata


def save_outputs(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame, db: str, positive_ratio) -> None:
    """Save feature matrix, labels, and patient metadata to parquet files."""
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    suffix = cfg.get_ratio_suffix(positive_ratio)
    
    X.to_parquet(cfg.OUTPUT_DIR / f"X_{db}_{suffix}.parquet")
    pd.DataFrame(y).to_parquet(cfg.OUTPUT_DIR / f"y_{db}_{suffix}.parquet")
    metadata.to_parquet(cfg.OUTPUT_DIR / f"metadata_{db}_{suffix}.parquet")
    print(f"  Saved: X_{db}_{suffix}.parquet, y_{db}_{suffix}.parquet, metadata_{db}_{suffix}.parquet")


def main():
    """Run preprocessing pipeline for both natural and balanced outputs."""
    cfg.validate_config()
    
    # Process with natural class distribution
    print("\n" + "="*60)
    print("VERSION 1: NATURAL CLASS DISTRIBUTION")
    print("="*60)
    
    for db in cfg.DATABASES:
        np.random.seed(cfg.RANDOM_STATE)
        X, y, metadata = process_database(db, "natural")
        save_outputs(X, y, metadata, db, "natural")
    
    # Process with balanced ratio from config
    if isinstance(cfg.POSITIVE_RATIO, float):
        print("\n" + "="*60)
        print(f"VERSION 2: BALANCED TO {int(cfg.POSITIVE_RATIO * 100)}% POSITIVE")
        print("="*60)
        
        for db in cfg.DATABASES:
            np.random.seed(cfg.RANDOM_STATE)
            X, y, metadata = process_database(db, cfg.POSITIVE_RATIO)
            save_outputs(X, y, metadata, db, cfg.POSITIVE_RATIO)


if __name__ == "__main__":
    main()