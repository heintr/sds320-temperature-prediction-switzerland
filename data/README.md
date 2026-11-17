# Data

Large raw and processed datasets are intentionally excluded from GitHub. The project is designed so the main data can be acquired or regenerated from the notebooks/code.

## Raw data

### MeteoSwiss TabsD and RhiresD

`src/project_utils.py` discovers yearly assets from the Swiss federal STAC collection:

```text
https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-surface-derived-grid/items
```

The first run of `notebooks/01_main_project.ipynb` downloads yearly NetCDF files for:

- `TabsD` — daily mean temperature
- `RhiresD` — daily precipitation

for 2015–2024 into:

```text
data/raw/TabsD/
data/raw/RhiresD/
```

The native gridded data use Swiss LV95 coordinates (EPSG:2056). Every eighth grid cell is sampled for this project.

### Elevation

Elevation is added once per sampled grid cell using the Swiss federal height service:

```text
https://api3.geo.admin.ch/rest/services/height
```

The resulting cache is stored as:

```text
data/processed/elevation_by_cell.csv
```

### ECMWF TIGGE

`notebooks/02_ecmwf_benchmark.ipynb` uses the Copernicus Climate Data Store API to request the `tigge-forecasts` dataset for ECMWF control forecasts.

Configuration used in this project:

```text
variable: 2 m temperature
initialization: 00 UTC
lead times: 0, 6, 12, 18 hours
forecast type: control forecast
grid: 0.2° x 0.2°
benchmark dates: Jan 15–21, Apr 15–21, Jul 1–7, Oct 15–21 2024
```

The downloaded GRIB files are stored in:

```text
data/raw/tigge_ecmwf/
```

A personal CDS API configuration (`~/.cdsapirc`) is required and must never be committed.

## Processed data

The main modelling cache is:

```text
data/processed/model_table_2015_2024_stride8.parquet
```

If this file is absent, notebook 01 rebuilds it from the source data. The completed table contains 730 sampled grid cells and 2,660,850 modelling rows.

Other processed outputs include:

```text
data/processed/test_predictions_2024.parquet
data/processed/tigge_comparison_2024.parquet
```

These are generated artifacts and are not required in version control.

## Why data are excluded from Git

The raw archive and processed Parquet files are substantially larger than the source code and compact project outputs. Keeping them outside the repository avoids duplicating provider data and keeps the public project lightweight.

The repository instead contains:

- acquisition/preprocessing code;
- environment information;
- compact CSV result tables;
- final figures;
- the selected fitted model.

Always consult the original provider documentation and terms before redistributing source data.
