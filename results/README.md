# Results

This folder contains compact, Git-friendly outputs from the completed run.

Authoritative result files:

- `validation_scores.csv` — 2023 model-selection results
- `test_scores_2024.csv` — final unseen 2024 test results
- `feature_importance.csv` — permutation importance calculated on validation data
- `monthly_model_vs_persistence_2024.csv` — monthly MAE comparison
- `daily_model_vs_persistence_2024.csv` — daily MAE comparison
- `cell_error_2024.csv` — grid-cell MAE and signed bias in LV95
- `cell_error_2024_wgs84.csv` — grid-cell results transformed to longitude/latitude
- `tigge_benchmark_scores.csv` — aggregate 28-day ECMWF benchmark
- `tigge_daily_mae.csv` — day-by-day benchmark results

The selected model in `models/selected_model.json` records:

```text
HGB extended + MAE loss
2023 validation MAE: 1.492 °C
2024 test MAE:       1.567 °C
```
