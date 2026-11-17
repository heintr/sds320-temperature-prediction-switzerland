from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import re
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


STAC_ITEMS = (
    "https://data.geo.admin.ch/api/stac/v1/collections/"
    "ch.meteoschweiz.ogd-surface-derived-grid/items"
)
HEIGHT_API = "https://api3.geo.admin.ch/rest/services/height"


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

def project_dirs(root: Path):
    root = Path(root)

    out = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "models": root / "models",
        "figures": root / "figures",
        "results": root / "results",
    }

    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)

    return out


# -------------------------------------------------------------------
# MeteoSwiss download
# -------------------------------------------------------------------

def _year(text):
    m = re.search(r"(?<!\d)((?:19|20)\d{2})\d{10}", text)

    if not m:
        m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)

    return int(m.group(1)) if m else None


def _stac_items():
    url = STAC_ITEMS
    params = {"limit": 100}
    items = []

    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()

        payload = r.json()
        items.extend(payload.get("features", []))

        url = next(
            (
                link["href"]
                for link in payload.get("links", [])
                if link.get("rel") == "next"
            ),
            None,
        )

        params = None

    return items


def yearly_assets(product, start_year, end_year):
    items = _stac_items()

    archive = next(
        item
        for item in items
        if "archive"
        in " ".join(
            [
                str(item.get("id", "")),
                str(item.get("title", "")),
                str(item.get("properties", {}).get("title", "")),
                str(item.get("properties", {}).get("description", "")),
            ]
        ).lower()
    )

    rows = []

    for key, asset in archive.get("assets", {}).items():
        text = (
            f"{key} {asset.get('href', '')} "
            f"{asset.get('title', '')}"
        ).lower()

        year = _year(text)

        if (
            product.lower() in text
            and year is not None
            and start_year <= year <= end_year
        ):
            rows.append(
                {
                    "year": year,
                    "href": asset["href"],
                }
            )

    table = (
        pd.DataFrame(rows)
        .sort_values("year")
        .drop_duplicates("year")
    )

    expected = set(range(start_year, end_year + 1))
    found = set(table["year"]) if len(table) else set()
    missing = sorted(expected - found)

    if missing:
        raise RuntimeError(
            f"{product}: missing archive years {missing}"
        )

    return table


def _open_netcdf(path):
    try:
        return xr.open_dataset(
            path,
            engine="netcdf4",
            chunks=None,
        )
    except Exception:
        return xr.open_dataset(
            path,
            engine="h5netcdf",
            chunks=None,
        )


def netcdf_ok(path):
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        ds = _open_netcdf(path)
        _ = dict(ds.sizes)
        ds.close()
        return True
    except Exception:
        return False


def download_file(url, path):
    """
    Download to a temporary file first and validate it.

    This stops incomplete NetCDF files from being reused later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if netcdf_ok(path):
        return path

    if path.exists():
        print("Removing broken file:", path.name)
        path.unlink()

    part = path.with_suffix(path.suffix + ".part")

    if part.exists():
        part.unlink()

    print("Downloading:", path.name)

    with requests.get(url, stream=True, timeout=240) as r:
        r.raise_for_status()

        with part.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    if not netcdf_ok(part):
        part.unlink(missing_ok=True)

        raise OSError(
            f"Downloaded file is invalid: {path.name}"
        )

    os.replace(part, path)

    return path


def download_years(asset_table, out_dir):
    out_dir = Path(out_dir)
    files = []

    for row in asset_table.itertuples(index=False):
        name = Path(
            str(row.href).split("?")[0]
        ).name

        path = out_dir / name

        files.append(
            download_file(
                row.href,
                path,
            )
        )

    return files


def open_product(files, variable):
    datasets = [
        _open_netcdf(path)
        for path in files
    ]

    ds = xr.combine_by_coords(
        datasets,
        combine_attrs="override",
        data_vars="all",
    )

    if variable in ds.data_vars:
        return ds[variable]

    match = next(
        (
            v
            for v in ds.data_vars
            if variable.lower() in v.lower()
        ),
        None,
    )

    if match is None:
        raise KeyError(
            f"{variable} not found. "
            f"Available: {list(ds.data_vars)}"
        )

    return ds[match]


def load_weather(
    raw_dir,
    start_year=2015,
    end_year=2024,
    stride=8,
):
    raw_dir = Path(raw_dir)

    t_assets = yearly_assets(
        "TabsD",
        start_year,
        end_year,
    )

    p_assets = yearly_assets(
        "RhiresD",
        start_year,
        end_year,
    )

    t_files = download_years(
        t_assets,
        raw_dir / "TabsD",
    )

    p_files = download_years(
        p_assets,
        raw_dir / "RhiresD",
    )

    temp = open_product(
        t_files,
        "TabsD",
    )

    precip = open_product(
        p_files,
        "RhiresD",
    )

    temp, precip = xr.align(
        temp,
        precip,
        join="inner",
    )

    spatial = [
        d
        for d in temp.dims
        if d.lower() != "time"
    ]

    if set(map(str.upper, spatial)) != {"N", "E"}:
        raise RuntimeError(
            f"Expected N/E dimensions, got {spatial}"
        )

    indexers = {
        d: slice(None, None, stride)
        for d in spatial
    }

    return (
        temp.isel(indexers),
        precip.isel(indexers),
    )


# -------------------------------------------------------------------
# Table + elevation
# -------------------------------------------------------------------

def weather_table(temp, precip):
    spatial = [
        d
        for d in temp.dims
        if d.lower() != "time"
    ]

    e_dim = next(
        d for d in spatial
        if d.upper() == "E"
    )

    n_dim = next(
        d for d in spatial
        if d.upper() == "N"
    )

    time_dim = next(
        d for d in temp.dims
        if d.lower() == "time"
    )

    t = (
        temp
        .to_dataframe(name="temp")
        .reset_index()
    )

    p = (
        precip
        .to_dataframe(name="precip")
        .reset_index()
    )

    keys = [
        time_dim,
        n_dim,
        e_dim,
    ]

    df = t[
        keys + ["temp"]
    ].merge(
        p[keys + ["precip"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    df = (
        df
        .dropna(subset=["temp"])
        .copy()
    )

    df["date"] = pd.to_datetime(
        df[time_dim]
    )

    df["cell_id"] = (
        df
        .groupby(
            [e_dim, n_dim],
            sort=False,
        )
        .ngroup()
        .astype("int32")
    )

    return df, e_dim, n_dim


def _height(
    easting,
    northing,
    retries=3,
):
    params = {
        "easting": float(easting),
        "northing": float(northing),
        "sr": 2056,
    }

    for attempt in range(retries):
        try:
            r = requests.get(
                HEIGHT_API,
                params=params,
                timeout=30,
            )

            r.raise_for_status()

            value = r.json().get("height")

            return (
                np.nan
                if value in (None, "", "null")
                else float(value)
            )

        except Exception:
            if attempt == retries - 1:
                raise

            time.sleep(
                0.8 * (attempt + 1)
            )


def add_elevation(
    df,
    e_dim,
    n_dim,
    cache_file,
    workers=6,
):
    cache_file = Path(cache_file)

    cells = (
        df[
            ["cell_id", e_dim, n_dim]
        ]
        .drop_duplicates("cell_id")
        .copy()
    )

    if cache_file.exists():
        cache = pd.read_csv(
            cache_file
        )
    else:
        cache = pd.DataFrame(
            columns=[
                "cell_id",
                e_dim,
                n_dim,
                "elevation_m",
            ]
        )

    cache["cell_id"] = pd.to_numeric(
        cache.get("cell_id"),
        errors="coerce",
    )

    cached = set(
        cache["cell_id"]
        .dropna()
        .astype(int)
    )

    todo = cells[
        ~cells["cell_id"].isin(cached)
    ]

    if len(todo):
        print(
            f"Downloading elevation "
            f"for {len(todo)} cells..."
        )

        records = []

        with ThreadPoolExecutor(
            max_workers=workers
        ) as pool:

            futures = {
                pool.submit(
                    _height,
                    row[e_dim],
                    row[n_dim],
                ): row
                for _, row in todo.iterrows()
            }

            for future in as_completed(
                futures
            ):
                row = futures[future]

                records.append(
                    {
                        "cell_id": int(
                            row["cell_id"]
                        ),
                        e_dim: float(
                            row[e_dim]
                        ),
                        n_dim: float(
                            row[n_dim]
                        ),
                        "elevation_m": future.result(),
                    }
                )

        cache = pd.concat(
            [
                cache,
                pd.DataFrame(records),
            ],
            ignore_index=True,
        )

        cache = (
            cache
            .drop_duplicates(
                "cell_id",
                keep="last",
            )
            .sort_values("cell_id")
        )

        cache_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache.to_csv(
            cache_file,
            index=False,
        )

    cache["elevation_m"] = pd.to_numeric(
        cache["elevation_m"],
        errors="coerce",
    )

    return df.merge(
        cache[
            ["cell_id", "elevation_m"]
        ],
        on="cell_id",
        how="left",
        validate="many_to_one",
    )


# -------------------------------------------------------------------
# Features
# -------------------------------------------------------------------

def add_features(
    df,
    e_dim,
    n_dim,
):
    """
    Features use only information available on or before predictor day t.
    """
    df = (
        df
        .sort_values(
            ["cell_id", "date"]
        )
        .copy()
    )

    g = df.groupby(
        "cell_id",
        sort=False,
    )

    for lag in (
        1,
        2,
        3,
        7,
    ):
        df[f"temp_lag{lag}"] = (
            g["temp"]
            .shift(lag)
        )

    df["temp_t"] = df["temp"]

    df["precip_t"] = (
        df["precip"]
        .fillna(0)
    )

    df["precip_lag1"] = (
        df
        .groupby("cell_id")["precip_t"]
        .shift(1)
    )

    df["temp_mean_3d"] = (
        g["temp"]
        .rolling(
            3,
            min_periods=3,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    df["temp_mean_7d"] = (
        g["temp"]
        .rolling(
            7,
            min_periods=7,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    df["temp_std_7d"] = (
        g["temp"]
        .rolling(
            7,
            min_periods=7,
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    temp_min_7d = (
        g["temp"]
        .rolling(
            7,
            min_periods=7,
        )
        .min()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    temp_max_7d = (
        g["temp"]
        .rolling(
            7,
            min_periods=7,
        )
        .max()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    df["temp_range_7d"] = (
        temp_max_7d
        - temp_min_7d
    )

    df["precip_sum_3d"] = (
        df
        .groupby(
            "cell_id",
            sort=False,
        )["precip_t"]
        .rolling(
            3,
            min_periods=3,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    df["temp_change_1d"] = (
        df["temp_t"]
        - df["temp_lag1"]
    )

    # Same-day nationwide context.
    # This is still available at predictor day t.
    by_date = df.groupby(
        "date",
        sort=False,
    )

    df["ch_temp_mean_t"] = (
        by_date["temp_t"]
        .transform("mean")
    )

    df["ch_precip_mean_t"] = (
        by_date["precip_t"]
        .transform("mean")
    )

    df["temp_anomaly_ch"] = (
        df["temp_t"]
        - df["ch_temp_mean_t"]
    )

    df["target_temp_tomorrow"] = (
        g["temp"]
        .shift(-1)
    )

    df["target_date"] = (
        df["date"]
        + pd.Timedelta(days=1)
    )

    doy = df["date"].dt.dayofyear

    df["sin_doy"] = np.sin(
        2 * np.pi * doy / 365.25
    )

    df["cos_doy"] = np.cos(
        2 * np.pi * doy / 365.25
    )

    df["coord_1"] = (
        df[e_dim]
        .astype(float)
    )

    df["coord_2"] = (
        df[n_dim]
        .astype(float)
    )

    needed = [
        "temp_t",
        "temp_lag1",
        "temp_lag2",
        "temp_lag3",
        "temp_lag7",
        "precip_t",
        "precip_lag1",
        "temp_mean_3d",
        "temp_mean_7d",
        "temp_std_7d",
        "temp_range_7d",
        "precip_sum_3d",
        "temp_change_1d",
        "ch_temp_mean_t",
        "ch_precip_mean_t",
        "temp_anomaly_ch",
        "target_temp_tomorrow",
        "elevation_m",
    ]

    return (
        df
        .dropna(subset=needed)
        .copy()
    )


def build_or_load_model_table(
    dirs,
    start_year=2015,
    end_year=2024,
    stride=8,
):
    """
    First run:
        NetCDF -> table -> elevation -> features -> parquet cache

    Later runs:
        load one parquet file directly.

    This is the main quality-of-life improvement for model iteration.
    """
    cache_file = (
        Path(dirs["processed"])
        / (
            f"model_table_"
            f"{start_year}_{end_year}_"
            f"stride{stride}.parquet"
        )
    )

    if cache_file.exists():
        print(
            "Loading cached model table:",
            cache_file.name,
        )

        df = pd.read_parquet(
            cache_file
        )

        e_dim = next(
            c
            for c in df.columns
            if str(c).upper() == "E"
        )

        n_dim = next(
            c
            for c in df.columns
            if str(c).upper() == "N"
        )

        return df, e_dim, n_dim

    print(
        "No processed cache found. "
        "Building it once..."
    )

    temp, precip = load_weather(
        dirs["raw"],
        start_year=start_year,
        end_year=end_year,
        stride=stride,
    )

    df, e_dim, n_dim = weather_table(
        temp,
        precip,
    )

    df = add_elevation(
        df,
        e_dim,
        n_dim,
        Path(dirs["processed"])
        / "elevation_by_cell.csv",
    )

    df = add_features(
        df,
        e_dim,
        n_dim,
    )

    df.to_parquet(
        cache_file,
        index=False,
    )

    print(
        "Saved processed cache:",
        cache_file.name,
    )

    return df, e_dim, n_dim


def split_years(
    df,
    train_end=2022,
    valid_year=2023,
    test_year=2024,
):
    """
    Split by target date, not predictor date.

    This keeps the forecast target years completely separated.
    """
    target_year = (
        pd.to_datetime(
            df["target_date"]
        )
        .dt.year
    )

    train = df[
        target_year <= train_end
    ].copy()

    valid = df[
        target_year == valid_year
    ].copy()

    test = df[
        target_year == test_year
    ].copy()

    return train, valid, test


# -------------------------------------------------------------------
# Forecast models
# -------------------------------------------------------------------

def metrics(
    y_true,
    y_pred,
    name,
):
    return {
        "model": name,
        "MAE_C": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "RMSE_C": mean_squared_error(
            y_true,
            y_pred,
        ) ** 0.5,
        "R2": r2_score(
            y_true,
            y_pred,
        ),
    }


def _fit_one(
    data,
    target,
    spec,
):
    estimator, features, mode = spec

    model = clone(estimator)

    if mode == "delta":
        y = data[target] - data["temp_t"]
    elif mode == "seasonal_residual":
        if "seasonal_persistence_C" not in data.columns:
            raise KeyError(
                "seasonal_residual mode requires seasonal_persistence_C"
            )
        y = data[target] - data["seasonal_persistence_C"]
    else:
        y = data[target]

    model.fit(data[features], y)

    return model, features, mode


def forecast(
    fitted_spec,
    data,
):
    model, features, mode = fitted_spec

    pred = model.predict(data[features])

    if mode == "delta":
        pred = data["temp_t"].to_numpy() + pred
    elif mode == "seasonal_residual":
        pred = data["seasonal_persistence_C"].to_numpy() + pred

    return pred


def fit_forecasters(
    train,
    valid,
    target,
    specs,
):
    fitted = {}

    rows = [
        metrics(
            valid[target],
            valid["temp_t"],
            "Persistence",
        )
    ]

    for name, spec in specs.items():
        fitted[name] = _fit_one(
            train,
            target,
            spec,
        )

        pred = forecast(
            fitted[name],
            valid,
        )

        rows.append(
            metrics(
                valid[target],
                pred,
                name,
            )
        )

    scores = (
        pd.DataFrame(rows)
        .sort_values("MAE_C")
        .reset_index(drop=True)
    )

    persistence_mae = float(
        scores.loc[
            scores["model"] == "Persistence",
            "MAE_C",
        ].iloc[0]
    )

    scores["skill_vs_persistence_pct"] = (
        100
        * (
            1
            - scores["MAE_C"]
            / persistence_mae
        )
    )

    return (
        fitted,
        scores,
    )


def refit_forecasters(
    train,
    test,
    target,
    specs,
):
    fitted = {}

    rows = [
        metrics(
            test[target],
            test["temp_t"],
            "Persistence",
        )
    ]

    for name, spec in specs.items():
        fitted[name] = _fit_one(
            train,
            target,
            spec,
        )

        pred = forecast(
            fitted[name],
            test,
        )

        rows.append(
            metrics(
                test[target],
                pred,
                name,
            )
        )

    scores = (
        pd.DataFrame(rows)
        .sort_values("MAE_C")
        .reset_index(drop=True)
    )

    persistence_mae = float(
        scores.loc[
            scores["model"] == "Persistence",
            "MAE_C",
        ].iloc[0]
    )

    scores["skill_vs_persistence_pct"] = (
        100
        * (
            1
            - scores["MAE_C"]
            / persistence_mae
        )
    )

    return (
        fitted,
        scores,
    )


def make_test_results(
    test,
    target,
    fitted_spec,
    e_dim,
    n_dim,
):
    out = test[
        [
            "date",
            "target_date",
            "cell_id",
            e_dim,
            n_dim,
            "elevation_m",
            "temp_t",
            target,
        ]
    ].copy()

    out["prediction"] = forecast(
        fitted_spec,
        test,
    )

    out["error"] = (
        out["prediction"]
        - out[target]
    )

    out["abs_error"] = (
        out["error"]
        .abs()
    )

    return out


def permutation_mae_importance(
    fitted_spec,
    data,
    target,
    sample_size=40000,
    random_state=42,
):
    """
    Simple validation-set permutation importance in MAE units.

    Larger values mean the feature mattered more.
    """
    model, features, mode = fitted_spec

    sample = data.sample(
        n=min(
            sample_size,
            len(data),
        ),
        random_state=random_state,
    ).copy()

    actual = (
        sample[target]
        .to_numpy()
    )

    baseline_pred = forecast(
        fitted_spec,
        sample,
    )

    baseline_mae = mean_absolute_error(
        actual,
        baseline_pred,
    )

    rng = np.random.default_rng(
        random_state
    )

    rows = []

    for feature in features:
        shuffled = sample.copy()

        shuffled[feature] = (
            rng.permutation(
                shuffled[feature].to_numpy()
            )
        )

        pred = model.predict(
            shuffled[features]
        )

        if mode == "delta":
            pred = (
                sample["temp_t"].to_numpy()
                + pred
            )

        shuffled_mae = (
            mean_absolute_error(
                actual,
                pred,
            )
        )

        rows.append(
            {
                "feature": feature,
                "MAE_increase_C": (
                    shuffled_mae
                    - baseline_mae
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "MAE_increase_C",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# -------------------------------------------------------------------
# Static visuals
# -------------------------------------------------------------------

def _finish_figure(fig, out_file=None):
    """Apply consistent report styling, optionally save, then display."""
    fig.tight_layout()
    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.show()


def score_bar(
    scores,
    title="Model comparison",
    out_file=None,
):
    """Horizontal MAE comparison with direct value labels."""
    plot = scores.sort_values("MAE_C", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bars = ax.barh(plot["model"], plot["MAE_C"])
    ax.set_xlabel("MAE (°C)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)

    xmax = float(plot["MAE_C"].max())
    ax.set_xlim(0, xmax * 1.18)

    for bar, value in zip(bars, plot["MAE_C"]):
        ax.text(
            value + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )

    _finish_figure(fig, out_file)


def importance_bar(
    importance,
    top_n=10,
    out_file=None,
):
    """Permutation importance in interpretable MAE units."""
    plot = (
        importance.head(top_n)
        .sort_values("MAE_increase_C", ascending=True)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.barh(plot["feature"], plot["MAE_increase_C"])
    ax.axvline(0, linewidth=0.8, color="0.35")
    ax.set_xlabel("Increase in validation MAE after shuffling (°C)")
    ax.set_title("Which predictors matter most to the selected model?")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)

    _finish_figure(fig, out_file)


def input_data_figure(
    data,
    e_dim,
    n_dim,
    figure_dir,
    date="2024-07-01",
):
    """Create one compact input-data figure for a representative predictor day."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    work = data.copy()
    work["date"] = pd.to_datetime(work["date"])
    target_date = pd.Timestamp(date)
    day = work.loc[work["date"] == target_date].copy()

    if day.empty:
        target_date = work["date"].sort_values().iloc[len(work) // 2]
        day = work.loc[work["date"] == target_date].copy()

    panels = [
        ("temp_t", "Temperature (°C)", "viridis"),
        ("precip_t", "Precipitation (mm/day)", "Blues"),
        ("elevation_m", "Elevation (m)", "terrain"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)

    for ax, (column, label, cmap) in zip(axes, panels):
        sc = ax.scatter(
            day[e_dim] / 1000,
            day[n_dim] / 1000,
            c=day[column],
            s=15,
            cmap=cmap,
            linewidths=0,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.82)
        cb.set_label(label)
        ax.set_xlabel("LV95 Easting (km)")
        ax.set_ylabel("LV95 Northing (km)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.08)
        ax.set_title(label.split(" (")[0])

    fig.suptitle(
        f"Input data across sampled Swiss grid cells — {pd.Timestamp(target_date):%d %b %Y}",
        fontsize=13,
    )
    fig.savefig(
        figure_dir / "00_input_data_example.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    return pd.Timestamp(target_date)


# -------------------------------------------------------------------
# Leakage-safe climatology features for model improvement
# -------------------------------------------------------------------

def add_climatology_features(
    reference,
    *frames,
    target="target_temp_tomorrow",
):
    """
    Add simple historical-climatology features without using future data.

    `reference` is the period allowed to inform the climatology.  For the
    reference rows themselves, leave-one-out values are used so each target
    does not directly contribute to its own feature.  Evaluation frames use
    means learned from the reference period only.

    Features:
      - cell_month_clim_C: historical mean target temperature for the same
        grid cell and target month.
      - seasonal_delta_C: historical mean next-day temperature change for
        the target day-of-year across Switzerland.
      - temp_vs_clim_C: today's local temperature relative to the cell/month
        climatology.
      - seasonal_persistence_C: today's temperature plus the historical
        seasonal day-to-day change.
    """
    ref = reference.copy()
    eval_frames = [frame.copy() for frame in frames]

    def add_time_keys(data):
        target_date = pd.to_datetime(data["target_date"])
        data["_target_month"] = target_date.dt.month.astype("int8")
        data["_target_doy"] = target_date.dt.dayofyear.astype("int16")
        return data

    ref = add_time_keys(ref)
    eval_frames = [add_time_keys(frame) for frame in eval_frames]

    # Cell x month climatology.  Training rows use leave-one-out means.
    cell_keys = ["cell_id", "_target_month"]
    cell_group = ref.groupby(cell_keys, sort=False)[target]
    cell_sum = cell_group.transform("sum")
    cell_count = cell_group.transform("count")
    overall_mean = float(ref[target].mean())

    ref["cell_month_clim_C"] = np.where(
        cell_count > 1,
        (cell_sum - ref[target]) / (cell_count - 1),
        overall_mean,
    )

    cell_lookup = (
        ref.groupby(cell_keys, as_index=False)[target]
        .mean()
        .rename(columns={target: "cell_month_clim_C"})
    )

    # Mean seasonal day-to-day change.  Again use leave-one-out on ref.
    ref["_target_delta_C"] = ref[target] - ref["temp_t"]
    doy_group = ref.groupby("_target_doy", sort=False)["_target_delta_C"]
    doy_sum = doy_group.transform("sum")
    doy_count = doy_group.transform("count")
    overall_delta = float(ref["_target_delta_C"].mean())

    ref["seasonal_delta_C"] = np.where(
        doy_count > 1,
        (doy_sum - ref["_target_delta_C"]) / (doy_count - 1),
        overall_delta,
    )

    doy_lookup = (
        ref.groupby("_target_doy", as_index=False)["_target_delta_C"]
        .mean()
        .rename(columns={"_target_delta_C": "seasonal_delta_C"})
    )

    def finish(data, use_lookup):
        if use_lookup:
            data = data.merge(
                cell_lookup,
                on=cell_keys,
                how="left",
                validate="many_to_one",
            )
            data = data.merge(
                doy_lookup,
                on="_target_doy",
                how="left",
                validate="many_to_one",
            )

            data["cell_month_clim_C"] = data["cell_month_clim_C"].fillna(
                overall_mean
            )
            data["seasonal_delta_C"] = data["seasonal_delta_C"].fillna(
                overall_delta
            )

        data["temp_vs_clim_C"] = (
            data["temp_t"] - data["cell_month_clim_C"]
        )
        data["seasonal_persistence_C"] = (
            data["temp_t"] + data["seasonal_delta_C"]
        )

        return data.drop(
            columns=["_target_month", "_target_doy", "_target_delta_C"],
            errors="ignore",
        )

    ref = finish(ref, use_lookup=False)
    eval_frames = [finish(frame, use_lookup=True) for frame in eval_frames]

    return (ref, *eval_frames)


def report_figures(
    test_results,
    test_scores,
    e_dim,
    n_dim,
    figure_dir,
):
    """Create focused, report-ready evaluation figures.

    The figure set follows the course logic: summary metrics plus spatial
    residual/error views. Observed and predicted maps use the same scale,
    while signed error uses a symmetric diverging scale around zero.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = test_results.copy()
    results["target_date"] = pd.to_datetime(results["target_date"])
    results["persistence_abs_error"] = (
        results["temp_t"] - results["target_temp_tomorrow"]
    ).abs()

    # 1) Final model comparison.
    plot = test_scores.sort_values("MAE_C", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.barh(plot["model"], plot["MAE_C"])
    ax.set_xlabel("MAE (°C)")
    ax.set_title("2024 final-test error")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    xmax = float(plot["MAE_C"].max())
    ax.set_xlim(0, xmax * 1.18)
    for bar, value in zip(bars, plot["MAE_C"]):
        ax.text(
            value + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f} °C",
            va="center",
            fontsize=9,
        )
    _finish_figure(fig, figure_dir / "01_model_comparison_2024.png")

    # 2) Monthly error: selected model vs persistence.
    monthly = (
        results.assign(month=results["target_date"].dt.month)
        .groupby("month", as_index=False)
        .agg(
            model_MAE_C=("abs_error", "mean"),
            persistence_MAE_C=("persistence_abs_error", "mean"),
        )
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    ax.plot(
        monthly["month"], monthly["model_MAE_C"],
        marker="o", linewidth=2, label="Selected model"
    )
    ax.plot(
        monthly["month"], monthly["persistence_MAE_C"],
        marker="o", linewidth=1.6, label="Persistence"
    )
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Target month")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("When does the ML model improve on persistence?")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    _finish_figure(fig, figure_dir / "02_monthly_error_2024.png")

    # 3) Daily error with a rolling trend so the main pattern is readable.
    daily = (
        results.groupby("target_date", as_index=False)
        .agg(
            model_MAE_C=("abs_error", "mean"),
            persistence_MAE_C=("persistence_abs_error", "mean"),
        )
        .sort_values("target_date")
    )
    daily["model_14d_MAE_C"] = daily["model_MAE_C"].rolling(14, center=True, min_periods=5).mean()
    daily["persistence_14d_MAE_C"] = daily["persistence_MAE_C"].rolling(14, center=True, min_periods=5).mean()

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.plot(daily["target_date"], daily["model_MAE_C"], linewidth=0.7, alpha=0.25)
    ax.plot(daily["target_date"], daily["persistence_MAE_C"], linewidth=0.7, alpha=0.18)
    ax.plot(daily["target_date"], daily["model_14d_MAE_C"], linewidth=2.0, label="Selected model — 14-day mean")
    ax.plot(daily["target_date"], daily["persistence_14d_MAE_C"], linewidth=1.7, label="Persistence — 14-day mean")
    ax.set_ylabel("Daily MAE (°C)")
    ax.set_xlabel("2024 target date")
    ax.set_title("Forecast difficulty changes through the year")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, ncol=2)
    _finish_figure(fig, figure_dir / "03_daily_error_2024.png")

    # Per-cell summary used by static and interactive maps.
    cell = (
        results.groupby(["cell_id", e_dim, n_dim, "elevation_m"], as_index=False)
        .agg(MAE_C=("abs_error", "mean"), bias_C=("error", "mean"))
    )

    # 4) Spatial MAE: sequential colour scale.
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sc = ax.scatter(
        cell[e_dim] / 1000,
        cell[n_dim] / 1000,
        c=cell["MAE_C"],
        s=28,
        cmap="viridis",
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("2024 MAE (°C)")
    ax.set_xlabel("LV95 Easting (km)")
    ax.set_ylabel("LV95 Northing (km)")
    ax.set_title("Where is next-day temperature prediction less accurate?")
    ax.set_aspect("equal")
    ax.grid(alpha=0.08)
    _finish_figure(fig, figure_dir / "04_spatial_mae_2024.png")

    # 5) Spatial bias: diverging scale with a meaningful centre at zero.
    bias_limit = max(0.05, float(cell["bias_C"].abs().quantile(0.99)))
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sc = ax.scatter(
        cell[e_dim] / 1000,
        cell[n_dim] / 1000,
        c=cell["bias_C"],
        s=28,
        cmap="RdBu_r",
        vmin=-bias_limit,
        vmax=bias_limit,
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax, extend="both")
    cb.set_label("Mean prediction error (prediction − observed, °C)")
    ax.set_xlabel("LV95 Easting (km)")
    ax.set_ylabel("LV95 Northing (km)")
    ax.set_title("Is the model systematically too warm or too cold?")
    ax.set_aspect("equal")
    ax.grid(alpha=0.08)
    _finish_figure(fig, figure_dir / "05_spatial_bias_2024.png")

    # 6) Elevation relationship with a robust binned trend.
    cell = cell.sort_values("elevation_m").copy()
    cell["elevation_bin"] = pd.qcut(cell["elevation_m"], q=10, duplicates="drop")
    elev_trend = (
        cell.groupby("elevation_bin", observed=True, as_index=False)
        .agg(elevation_m=("elevation_m", "mean"), MAE_C=("MAE_C", "mean"))
    )
    corr = float(cell[["elevation_m", "MAE_C"]].corr().iloc[0, 1])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(cell["elevation_m"], cell["MAE_C"], s=15, alpha=0.28, label="Grid cells")
    ax.plot(
        elev_trend["elevation_m"], elev_trend["MAE_C"],
        marker="o", linewidth=2.0, label="Elevation-bin mean"
    )
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("2024 MAE (°C)")
    ax.set_title(f"Does prediction error increase with elevation?  r = {corr:.2f}")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    _finish_figure(fig, figure_dir / "06_elevation_vs_error_2024.png")

    # 7) Course-style three-panel regression comparison for a representative day.
    day_stats = results.groupby("target_date")["abs_error"].mean().sort_values()
    median_error = float(day_stats.median())
    representative_date = (day_stats - median_error).abs().idxmin()
    day = results.loc[results["target_date"] == representative_date].copy()

    temp_lo = float(min(day["target_temp_tomorrow"].min(), day["prediction"].min()))
    temp_hi = float(max(day["target_temp_tomorrow"].max(), day["prediction"].max()))
    resid_limit = max(0.5, float(day["error"].abs().quantile(0.98)))

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), constrained_layout=True)
    panels = [
        ("target_temp_tomorrow", "Observed", "viridis", temp_lo, temp_hi),
        ("prediction", "Predicted", "viridis", temp_lo, temp_hi),
        ("error", "Prediction error", "RdBu_r", -resid_limit, resid_limit),
    ]
    for ax, (column, title, cmap, vmin, vmax) in zip(axes, panels):
        sc = ax.scatter(
            day[e_dim] / 1000,
            day[n_dim] / 1000,
            c=day[column],
            s=18,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.82)
        cb.set_label("Temperature (°C)" if column != "error" else "Prediction − observed (°C)")
        ax.set_xlabel("LV95 Easting (km)")
        ax.set_ylabel("LV95 Northing (km)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.06)
        ax.set_title(title)

    fig.suptitle(
        f"Observed, predicted and residual pattern — {pd.Timestamp(representative_date):%d %b %Y}\n"
        "Representative day selected near the median daily MAE",
        fontsize=13,
    )
    fig.savefig(
        figure_dir / "07_observed_predicted_residual_2024.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()

    # 8) Predicted vs observed density with standard metrics in the panel.
    fig, ax = plt.subplots(figsize=(6.3, 6.0))
    hb = ax.hexbin(
        results["target_temp_tomorrow"],
        results["prediction"],
        gridsize=70,
        mincnt=1,
        cmap="viridis",
    )
    lo = float(min(results["target_temp_tomorrow"].min(), results["prediction"].min()))
    hi = float(max(results["target_temp_tomorrow"].max(), results["prediction"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="0.2", label="1:1 line")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Observations per hexagon")
    ax.set_xlabel("Observed next-day mean temperature (°C)")
    ax.set_ylabel("Predicted temperature (°C)")
    ax.set_title("Observed vs predicted — all 2024 grid-day observations")
    selected_mask = test_scores["model"].astype(str).str.startswith("Selected:")
    selected_row = test_scores.loc[selected_mask].iloc[0] if selected_mask.any() else test_scores.iloc[0]
    ax.text(
        0.03, 0.97,
        f"MAE = {selected_row['MAE_C']:.3f} °C\nRMSE = {selected_row['RMSE_C']:.3f} °C\nR² = {selected_row['R2']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    ax.legend(frameon=False, loc="lower right")
    _finish_figure(fig, figure_dir / "08_observed_vs_predicted_2024.png")

    return monthly, daily, cell, corr


# -------------------------------------------------------------------
# TIGGE
# -------------------------------------------------------------------

def open_tigge_grib(path):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
        },
    )

    variable = next(
        (
            v
            for v in (
                "t2m",
                "2t",
            )
            if v in ds.data_vars
        ),
        None,
    )

    if (
        variable is None
        and len(ds.data_vars) == 1
    ):
        variable = list(
            ds.data_vars
        )[0]

    if variable is None:
        raise KeyError(
            "Could not identify "
            "2 m temperature. "
            f"Variables: {list(ds.data_vars)}"
        )

    return ds, variable


def tigge_daily_fields(
    files,
    target_dates,
):
    daily_fields = []

    for target_date, path in zip(
        target_dates,
        files,
    ):
        ds, var = open_tigge_grib(
            path
        )

        da = (
            ds[var]
            .squeeze(drop=True)
            - 273.15
        )

        if "step" in da.dims:
            daily = da.mean(
                "step"
            )

        elif "valid_time" in da.dims:
            daily = da.mean(
                "valid_time"
            )

        else:
            ds.close()

            raise RuntimeError(
                "Unexpected TIGGE dimensions: "
                f"{da.dims}"
            )

        daily = daily.rename(
            "tigge_temp_C"
        )

        daily = daily.expand_dims(
            target_date=[
                np.datetime64(
                    pd.Timestamp(
                        target_date
                    ).date()
                )
            ]
        )

        daily_fields.append(
            daily
        )

        ds.close()

    return xr.concat(
        daily_fields,
        dim="target_date",
        coords="minimal",
        compat="override",
    )


def comparison_scores(
    comparison,
    target="target_temp_tomorrow",
):
    rows = []

    for name, col in [
        (
            "Persistence",
            "temp_t",
        ),
        (
            "Our ML model",
            "prediction",
        ),
        (
            "ECMWF TIGGE",
            "tigge_prediction",
        ),
    ]:
        part = comparison[
            [target, col]
        ].dropna()

        rows.append(
            {
                "model": name,
                "n": len(part),
                "MAE_C": mean_absolute_error(
                    part[target],
                    part[col],
                ),
                "RMSE_C": (
                    mean_squared_error(
                        part[target],
                        part[col],
                    )
                    ** 0.5
                ),
                "R2": r2_score(
                    part[target],
                    part[col],
                ),
            }
        )

    scores = (
        pd.DataFrame(rows)
        .sort_values("MAE_C")
        .reset_index(drop=True)
    )

    persistence_mae = float(
        scores.loc[
            scores["model"] == "Persistence",
            "MAE_C",
        ].iloc[0]
    )

    scores["skill_vs_persistence_pct"] = (
        100
        * (
            1
            - scores["MAE_C"]
            / persistence_mae
        )
    )

    return scores
