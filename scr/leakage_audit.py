from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import re

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_combined_panel_models as pipeline
from combine_aqi_datasets import pm25_to_us_aqi


OUTPUT_PATH = Path("data/processed/california_aqi_leakage_audit.csv")


def ridge_score(X_train, y_train, X_test, y_test):
    X_train_scaled, _, X_test_scaled, _, _, _, _ = pipeline.prepare_model_matrices(X_train, X_test, X_test)
    model = Ridge(alpha=10.0)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    return {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
    }


def run_audit():
    raw = pd.read_csv(pipeline.DATA_PATH)
    pm25_proxy = pm25_to_us_aqi(raw["pm25_ug_m3"])
    target_proxy_max_abs_error = float(np.nanmax(np.abs(pm25_proxy - raw[pipeline.TARGET])))

    rows = [
        {
            "configuration": "dataset",
            "check": "target_aqi_is_pm25_proxy",
            "value": target_proxy_max_abs_error,
            "interpretation": "Near-zero value means target_aqi is deterministically derived from pm25_ug_m3.",
        }
    ]

    for configuration in pipeline.CONFIGURATIONS:
        engineered = pipeline.add_features(raw, configuration=configuration)
        X, y, y_time = pipeline.build_feature_matrix(engineered, configuration=configuration)
        station_ids = engineered.loc[X.index, "station_id"].reset_index(drop=True)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        y_time = y_time.reset_index(drop=True)

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            train_station_ids,
            val_station_ids,
            test_station_ids,
        ) = pipeline.climate_context_aware_split(X, y, y_time, station_ids, configuration)

        guard_hours = (
            pipeline.CONFIGURATIONS[configuration]["horizon"]
            + pipeline.max_history_context_hours(configuration)
        )

        rows.extend(
            [
                {
                    "configuration": configuration,
                    "check": "split_strategy",
                    "value": "climate_context_aware",
                    "interpretation": "Train/validation/test years are assigned by California climate context, with 2025 held out for final stress testing.",
                },
                {
                    "configuration": configuration,
                    "check": "train_years",
                    "value": sorted(y_train.index.year.unique().tolist()),
                    "interpretation": "Expected train years: 2018, 2019, 2021, 2022, 2023, 2024.",
                },
                {
                    "configuration": configuration,
                    "check": "validation_years",
                    "value": sorted(y_val.index.year.unique().tolist()),
                    "interpretation": "Expected validation year: 2020, isolated as an extreme wildfire-context holdout for early stopping.",
                },
                {
                    "configuration": configuration,
                    "check": "test_years",
                    "value": sorted(y_test.index.year.unique().tolist()),
                    "interpretation": "Expected independent test year: 2025.",
                },
                {
                    "configuration": configuration,
                    "check": "post_validation_guard_hours",
                    "value": guard_hours,
                    "interpretation": "Early 2021 train targets inside this guard are removed so lag/rolling features do not reference the 2020 validation holdout.",
                },
                {
                    "configuration": configuration,
                    "check": "train_target_period",
                    "value": f"{y_train.index.min()} to {y_train.index.max()}",
                    "interpretation": "Climate-context-aware target split.",
                },
                {
                    "configuration": configuration,
                    "check": "validation_target_period",
                    "value": f"{y_val.index.min()} to {y_val.index.max()}",
                    "interpretation": "Climate-context-aware target split.",
                },
                {
                    "configuration": configuration,
                    "check": "test_target_period",
                    "value": f"{y_test.index.min()} to {y_test.index.max()}",
                    "interpretation": "Climate-context-aware target split.",
                },
                {
                    "configuration": configuration,
                    "check": "test_station_distribution",
                    "value": test_station_ids.value_counts().to_dict(),
                    "interpretation": "Shows whether the independent test set is station-balanced.",
                },
            ]
        )

        future_like_features = [c for c in X.columns if "future" in c.lower()]
        pollutant_proxy_features = [c for c in ["pm25_ug_m3", "pm10_ug_m3"] if c in X.columns]
        lag_numbers = []
        local_target_lag_numbers = []
        spatial_lag_numbers = []
        for column in X.columns:
            match = re.search(r"_lag_(\d+)$", column)
            if match:
                lag = int(match.group(1))
                lag_numbers.append(lag)
                if column.startswith("target_aqi_lag_"):
                    local_target_lag_numbers.append(lag)
                if column.startswith("spatial_"):
                    spatial_lag_numbers.append(lag)
        recent_local_lag_features = [
            c for c in X.columns
            if c.startswith("target_aqi_lag_") and (m := re.search(r"_lag_(\d+)$", c)) and int(m.group(1)) < 24
        ]
        intermediate_spatial_lag_features = [
            c for c in X.columns
            if c.startswith("spatial_") and (m := re.search(r"_lag_(\d+)$", c)) and int(m.group(1)) in [6, 12]
        ]
        rows.append(
            {
                "configuration": configuration,
                "check": "future_like_feature_names",
                "value": future_like_features,
                "interpretation": "Should be empty. Non-empty values indicate direct future leakage.",
            }
        )
        rows.append(
            {
                "configuration": configuration,
                "check": "current_pollutant_proxy_features_present",
                "value": pollutant_proxy_features,
                "interpretation": "Should be empty because final configurations use AQI history, not current pollutant proxy variables.",
            }
        )
        rows.append(
            {
                "configuration": configuration,
                "check": "minimum_lag_hours",
                "value": min(lag_numbers) if lag_numbers else np.nan,
                "interpretation": "Minimum lag across local and spatial history features.",
            }
        )
        rows.append(
            {
                "configuration": configuration,
                "check": "minimum_local_target_lag_hours",
                "value": min(local_target_lag_numbers) if local_target_lag_numbers else np.nan,
                "interpretation": "Should be 1 for short-term nowcasting and >= 24 for long-term forecasting.",
            }
        )
        rows.append(
            {
                "configuration": configuration,
                "check": "minimum_spatial_lag_hours",
                "value": min(spatial_lag_numbers) if spatial_lag_numbers else np.nan,
                "interpretation": "Spatial lags may include 6h and 12h in the 24h-ahead setting to model cross-city transport progression.",
            }
        )
        rows.append(
            {
                "configuration": configuration,
                "check": "recent_local_target_lag_features_under_24h",
                "value": recent_local_lag_features,
                "interpretation": "Should be empty for Long-term Forecasting (Lag 24h). Short-term nowcasting intentionally allows local lag 1-3h.",
            }
        )
        rows.append(
            {
                "configuration": configuration,
                "check": "intermediate_spatial_lag_features_6_12h",
                "value": intermediate_spatial_lag_features,
                "interpretation": "Expected in Long-term Forecasting (Lag 24h) after spatial-lag tuning; these features are cross-station transport signals.",
            }
        )

        for feature in ["pm25_ug_m3", "pm10_ug_m3", "target_aqi_lag_1", "target_aqi_lag_24"]:
            if feature in X_test.columns:
                rows.append(
                    {
                        "configuration": configuration,
                        "check": f"corr_y_test_with_{feature}",
                        "value": float(X_test[feature].corr(y_test)),
                        "interpretation": "High absolute correlation indicates strong target/proxy persistence.",
                    }
                )

        all_score = ridge_score(X_train, y_train, X_test, y_test)
        drop_patterns = ["pm25", "pm10", "target_aqi_lag", "target_aqi_roll", "spatial_target_aqi"]
        strict_features = [c for c in X_train.columns if not any(p in c.lower() for p in drop_patterns)]
        strict_score = ridge_score(X_train[strict_features], y_train, X_test[strict_features], y_test)

        rows.extend(
            [
                {
                    "configuration": configuration,
                    "check": "ridge_all_features_r2",
                    "value": all_score["r2"],
                    "interpretation": "Baseline score with the current feature set.",
                },
                {
                    "configuration": configuration,
                    "check": "ridge_all_features_mae",
                    "value": all_score["mae"],
                    "interpretation": "Baseline score with the current feature set.",
                },
                {
                    "configuration": configuration,
                    "check": "ridge_without_pm_or_target_history_r2",
                    "value": strict_score["r2"],
                    "interpretation": "Large drop indicates that PM and target-history proxy features drive most predictive power.",
                },
                {
                    "configuration": configuration,
                    "check": "ridge_without_pm_or_target_history_mae",
                    "value": strict_score["mae"],
                    "interpretation": "Large increase indicates harder meteorology-only forecasting.",
                },
            ]
        )

    report = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)
    print(report.to_string(index=False))
    print(f"\nSaved leakage audit report: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_audit()
