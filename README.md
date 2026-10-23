# Federated performance limits for ICU transfusion prediction

Code for "What can federated data achieve? Extending performance limit estimation to federated learning with a case study on ICU blood transfusion" (Schwinn et al., PLOS Digital Health, under review).

The code estimates the maximum achievable Cohen's κ (κmax, Crow & Watts) for each database and, from aggregated client statistics, for all databases together. It trains LR, MLP, LSTM and XGBoost with local learning (LL) and federated learning (FL) on eICU-CRD, MIMIC-IV, UMCdb, HiRID and UKA, and validates the federated κmax estimator on simulated data with known limits.

## Requirements

- Python 3.11 and the packages in `requirements.txt`.
- Docker with Compose for the FL runs (one container per site). The image is CUDA 12.1 but also runs on CPU.
- The patient data are not included. eICU-CRD, MIMIC-IV and HiRID are available on PhysioNet, UMCdb from Amsterdam UMC; UKA data are not public. All databases were mapped to the OMOP CDM before preprocessing.

Run all commands from the repository root with `export PYTHONPATH=.`.

## Configuration

| File | Setting | Manuscript value |
|---|---|---|
| `config_data.py` | `DATA_DIR` | folder with the extracted raw data |
| `general_config/general_config.py` | `DATA_BALANCE` | `"20"` (20% cohort) or `"natural"` |
| | `OPTUNA_TRIALS` (LL) | 50 |
| | `FL_NUM_ROUNDS`, `FL_HPO_FACTOR`, `FL_HPO_RATE` | 600, 0.8, 10 |
| | `MAX_FEATURES`, `N_BINS`, `N_ITERATIONS` (κmax) | 8, 16, 5 |

`DATA_BALANCE` is set only in `general_config.py`; preprocessing, training, κmax and figure scripts all read it. Hyperparameter search spaces are in `federated/<model>/<model>_config.py`, and `S1_reproducibility_table.md` lists every setting. Client IDs are 1 eICU-CRD, 2 MIMIC-IV, 3 UMCdb, 4 HiRID, 5 UKA.

## 1. Preprocessing

```bash
python preprocessing.py
```

Reads `<DATA_DIR>/ds_extracted_correctblood_fl_<db>.parquet` and writes `X_`, `y_` and `metadata_<db>_<suffix>.parquet` to `FLdata/`. The natural cohort (suffix `natural`) is always written, the 20% cohort (`balanced_morevariables_20pct`) when `DATA_BALANCE = "20"`. Each patient contributes a 4 h observation window that ends 2 h before the first transfusion after ICU admission; non-transfused patients get windows of matched length.

All later steps load data through `mydataloading.prepare_data(client_id)`. It adds five hemoglobin trend features, drops patients without a hemoglobin value and creates a stratified 70/10/20 train/validation/test split on first use. The split is stored in `FLdata/splits/<balance>/` and reused by LL, FL and κmax; delete it to draw a new one. Median imputation and z-scaling are fitted on the training split. Per-feature missingness is written to `FLdata/<db>_features_missingness_<balance>.csv`.

## 2. κmax

```bash
python federated/kappa/kappa_parallel.py --selftest   # checks against the reference estimators, a few minutes
python federated/kappa/kappa_parallel.py              # point estimates, confidence intervals, figures
```

For each database and for all databases pooled, the script computes the direct Crow & Watts estimate on the patient data and the federated estimate from per-class histograms, feature sums and XᵀX (t-copula, ν = 4, 20 synthetic draws, median κmax). Confidence intervals come from up to 1,000 half-samples within a 6.5 h budget (`--n_rep`, `--hours`). Results go to `federated/kappa/kappa_parallel/<balance>/` (`kappa_point.csv`, `kappa_ci_summary.csv`, figures). An interrupted run resumes from `tasks.csv`; `--n_rep 0` computes point estimates only, `--plot_only` rebuilds the report.

`cd federated/kappa && docker compose up --build` runs the same federated estimate as a Flower deployment in a single round and writes `FLresults/federated_kappa_results.csv` and `FLresults/federated_kappa_<balance>.json`.

The per-database κmax stored with each trained model is computed on the first LL or FL run and cached in `FLhyperparams/theoretical_kappa_<db>_<balance>.json`.

## 3. MLflow

```bash
docker network create mlflow-limits-net
cd _mlflow && docker compose up -d
```

The UI runs on http://localhost:5055. The FL compose files need this network. Scripts run outside Docker without `MLFLOW_TRACKING_URI` log to `./mlruns`.

## 4. Local learning

```bash
python federated/<model>/local.py        # model: lr, mlp, lstm, xgboost
```

Trains one model per database. Optuna (TPE, 50 trials) minimizes the validation log loss, training stops early after 10 epochs (XGBoost: boosting rounds) without improvement, and the decision threshold maximizes F1 on the validation split. Test metrics come with 95% CIs from 1,000 class-stratified bootstrap resamples. Each compose file also contains a commented-out `local-<model>` service for running this in Docker.

## 5. Federated learning

```bash
cd federated/<model> && docker compose up --build                  # lr, lstm, xgboost
cd federated/mlp && docker compose -f docker-compose.mlp.yml up --build
```

Starts one server and five clients. LR uses FedProx with a tuned μ, MLP and LSTM run the same strategy with μ = 0 (FedAvg), and XGBoost uses histogram aggregation. Rounds 1-480 tune hyperparameters: every 10 rounds the server takes a new Optuna configuration (48 trials) and scores it by the sample-weighted validation log loss over all clients. Rounds 481-600 train with the best configuration, and the threshold maximizes the pooled validation F1. In the last round each client evaluates the global model on its test split, then again after one local training pass (XGBoost: 2 local trees) with a threshold re-tuned on its validation split. The manuscript reports these locally boosted results (`boost_*` columns).

LR, LSTM and XGBoost all use port 8097, so run them one after another. Containers run as UID/GID 1000; if yours differ, `export HOST_UID=$(id -u) HOST_GID=$(id -g)` first.

## 6. Simulation study

```bash
python simulation_validation/simulate_known_limits_v06.py --profile fast --part sweep models
```

Needs no patient data. `sweep` varies the class separation of a Gaussian mean shift (S3 Fig, `fig1_sweep_decomposition`), `models` covers the 17 data models at the baseline separation (S4 Fig, `fig3_datamodels`). Both use N = 5,000 entries over five clients, 5 replicates and 5 copula draws. Results go to `simulation_validation/results-v06/`. `--part all` adds coverage, site heterogeneity, dimension and histogram-window analyses that are not in the manuscript; `--dry_run` prints the plan and times one estimate.

## 7. Figures and tables

The figure scripts read `DATA_BALANCE` and the result folders from `general_config` and save their figures to the current directory.

- `makefigures-v3.py`: κmax per database against the federated κmax, achieved κ for LL and FL, FL AUPRC with and without local boost, AUROC for the 20% and the natural cohort. The federated κmax and its CI are hard-coded in `KMAX_FL_ALL`; copy them from `kappa_ci_summary.csv` (scope `ALL`).
- `fig_selected_features_kappa.py`: features selected by the κmax estimator, from `kappa_point.csv`.
- `missingness.py`: missingness heatmap from the `FLdata/*_features_missingness_*.csv` files.

From the repository root, `cohort-table.py` writes the cohort table and `longtable.py` the transfused vs. non-transfused tables (S3 and S4 Tables) to `Table1/<balance>/`. `preprocessing-count.py` writes the patient counts per preprocessing step (S2 Fig) to `FLdata/patient_flow_*.csv`.

## Outputs

```
FLresults/
  local_metrics_<model>_<balance>.csv                 LL: train/val/test metrics, test CIs, κmax
  FL_metrics_<MODEL>_client_<id>_<balance>.csv        FL: validation metrics per round
  FL_metrics_<MODEL>_client_<id>_<balance>_test.csv   FL: final test metrics with CIs and boost_*
  federated_metrics_aggregated.csv                    FL server metrics per round (LR, MLP, LSTM append here)
  FL_histogram_hpo_history_<balance>.csv              same for XGBoost
  optuna_trials_<balance>_<model>.csv                 FL Optuna trials
  models/                                             trained models
FLhyperparams/                                        best hyperparameters, thresholds, cached κmax
```

Reported metrics are AUROC (column `AUC`), AUPRC, F1, precision, recall, sensitivity, specificity, Brier score, log loss and κ (`actual_kappa`). κ efficiency is κ divided by the applicable κmax.


## References

Crow L, Watts SJ. Limits to classification performance by relating Kullback-Leibler divergence to Cohen's Kappa. arXiv:2403.01571, 2024.
