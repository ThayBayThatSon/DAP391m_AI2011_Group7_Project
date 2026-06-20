import os
import time
import warnings

os.environ["MPLBACKEND"] = "agg"

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)


RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 30
TRAIN_YEARS = [2018, 2019, 2021, 2022, 2023, 2024]
VALIDATION_YEARS = [2020]
TEST_YEARS = [2025]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "california_aqi_model_ready.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
LEADERBOARD_PATH = os.path.join(OUTPUT_DIR, "california_aqi_model_leaderboard.csv")
SCENARIO_PATH = os.path.join(OUTPUT_DIR, "california_aqi_scenario_evaluation.csv")
PREDICTION_PATH = os.path.join(OUTPUT_DIR, "california_aqi_model_predictions.csv")

TARGET = "target_aqi"
ID_COLS = ["source", "station_id", "station_name"]
CAT_FEATURES = ["source", "station_id"]

CONFIGURATIONS = {
    "Short-term Autoregressive (Lag 1-3h)": {
        "horizon": 1,
        "target_lags": [1, 2, 3],
        "rolling_shift": 1,
        "rolling_windows": [3],
        "spatial_lags": [1, 2, 3],
        "recent_lag_allowed": True,
    },
    "Long-term Forecasting (Lag 24h)": {
        "horizon": 24,
        "target_lags": [24, 48, 72, 168],
        "rolling_shift": 24,
        "rolling_windows": [24, 72, 168],
        "spatial_lags": [6, 12, 24, 48, 72],
        "recent_lag_allowed": False,
    },
}


def max_history_context_hours(configuration: str) -> int:
    config = CONFIGURATIONS[configuration]
    lag_context = max(config["target_lags"] + config["spatial_lags"])
    rolling_context = config["rolling_shift"] + max(config["rolling_windows"]) - 1
    return int(max(lag_context, rolling_context))


def safe_station_name(value: str) -> str:
    return (
        str(value)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace(",", "")
    )


def vapor_pressure_deficit_kpa(temperature_c: pd.Series, relative_humidity_pct: pd.Series) -> pd.Series:
    saturation_vapor_pressure = 0.6108 * np.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    actual_vapor_pressure = saturation_vapor_pressure * (relative_humidity_pct / 100.0)
    return (saturation_vapor_pressure - actual_vapor_pressure).clip(lower=0)


def add_spatial_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    group = df.groupby("station_id", group_keys=False)
    pivot = df.pivot_table(index="time", columns="station_id", values=TARGET, aggfunc="mean").sort_index()
    for lag in lags:
        shifted = pivot.shift(lag)
        spatial_mean = df["time"].map(shifted.mean(axis=1))
        local_fallback = group[TARGET].shift(lag)
        for station in shifted.columns:
            feature_name = f"spatial_{TARGET}_{safe_station_name(station)}_lag_{lag}"
            df[feature_name] = df["time"].map(shifted[station]).fillna(spatial_mean).fillna(local_fallback)
    return df


def add_features(df: pd.DataFrame, configuration: str) -> pd.DataFrame:
    config = CONFIGURATIONS[configuration]
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values(["station_id", "time"]).reset_index(drop=True)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    wd_rad = np.deg2rad(df["wind_direction_10m"])
    df["wind_u"] = df["wind_speed_10m"] * np.cos(wd_rad)
    df["wind_v"] = df["wind_speed_10m"] * np.sin(wd_rad)

    eps = 1e-6
    df["temp_humidity_ratio"] = df["temperature_2m"] / (df["relative_humidity_2m"] + eps)
    df["dry_heat_index"] = df["temperature_2m"] * (1 - df["relative_humidity_2m"] / 100.0)
    df["wind_speed_pressure"] = df["wind_speed_10m"] / (df["surface_pressure"] + eps)
    df["dry_wind_index"] = df["wind_speed_10m"] * (1 - df["relative_humidity_2m"] / 100.0)
    df["vpd_kpa"] = vapor_pressure_deficit_kpa(df["temperature_2m"], df["relative_humidity_2m"])

    group = df.groupby("station_id", group_keys=False)

    horizon = config["horizon"]
    # Future target: row at time t predicts AQI proxy at t + horizon for the same station.
    df[f"{TARGET}_future"] = group[TARGET].shift(-horizon)
    df["future_time"] = group["time"].shift(-horizon)

    for lag in config["target_lags"]:
        df[f"{TARGET}_lag_{lag}"] = group[TARGET].shift(lag)

    shifted_target = group[TARGET].shift(config["rolling_shift"])
    for window in config["rolling_windows"]:
        df[f"{TARGET}_roll_mean_from_lag{config['rolling_shift']}_{window}"] = (
            shifted_target.groupby(df["station_id"]).rolling(window).mean().reset_index(level=0, drop=True)
        )
        df[f"{TARGET}_roll_std_from_lag{config['rolling_shift']}_{window}"] = (
            shifted_target.groupby(df["station_id"]).rolling(window).std().reset_index(level=0, drop=True)
        )

    df = add_spatial_lag_features(df, lags=config["spatial_lags"])

    return df


def build_feature_matrix(df: pd.DataFrame, configuration: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Do not include current PM2.5/PM10 because target_aqi is a PM2.5-derived
    # AQI proxy. The two final configurations use weather/time/location plus
    # allowed historical AQI features only.
    exogenous_features = [
        "lat",
        "lon",
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "cloud_cover",
        "wind_speed_10m",
        "surface_pressure",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "wind_u",
        "wind_v",
        "temp_humidity_ratio",
        "dry_heat_index",
        "wind_speed_pressure",
        "dry_wind_index",
        "vpd_kpa",
    ]
    base_features = exogenous_features.copy()

    history_features = [c for c in df.columns if "_lag_" in c or "_roll_" in c or c.startswith("spatial_")]

    X = pd.concat([df[base_features + history_features], df[CAT_FEATURES].astype(str)], axis=1)
    y = df[f"{TARGET}_future"]
    y_time = pd.to_datetime(df["future_time"])

    numeric_features = [c for c in X.columns if c not in CAT_FEATURES]
    valid = X[numeric_features].notna().all(axis=1) & X[CAT_FEATURES].notna().all(axis=1) & y.notna() & y_time.notna()
    X_valid = X.loc[valid].copy()
    X_valid[numeric_features] = X_valid[numeric_features].astype(float)
    return X_valid, y.loc[valid].astype(float), y_time.loc[valid]


def prepare_model_matrices(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    numeric_features = [c for c in X_train.columns if c not in CAT_FEATURES]

    scaler = StandardScaler()
    train_numeric = pd.DataFrame(
        scaler.fit_transform(X_train[numeric_features]),
        columns=numeric_features,
        index=X_train.index,
    )
    val_numeric = pd.DataFrame(
        scaler.transform(X_val[numeric_features]),
        columns=numeric_features,
        index=X_val.index,
    )
    test_numeric = pd.DataFrame(
        scaler.transform(X_test[numeric_features]),
        columns=numeric_features,
        index=X_test.index,
    )

    train_cat = X_train[CAT_FEATURES].astype(str)
    val_cat = X_val[CAT_FEATURES].astype(str)
    test_cat = X_test[CAT_FEATURES].astype(str)

    train_ohe = pd.get_dummies(train_cat, prefix=CAT_FEATURES, dtype=float)
    ohe_columns = train_ohe.columns
    val_ohe = pd.get_dummies(val_cat, prefix=CAT_FEATURES, dtype=float).reindex(columns=ohe_columns, fill_value=0.0)
    test_ohe = pd.get_dummies(test_cat, prefix=CAT_FEATURES, dtype=float).reindex(columns=ohe_columns, fill_value=0.0)

    X_train_general = pd.concat([train_numeric, train_ohe], axis=1)
    X_val_general = pd.concat([val_numeric, val_ohe], axis=1)
    X_test_general = pd.concat([test_numeric, test_ohe], axis=1)

    X_train_catboost = pd.concat([train_numeric, train_cat], axis=1)
    X_val_catboost = pd.concat([val_numeric, val_cat], axis=1)
    X_test_catboost = pd.concat([test_numeric, test_cat], axis=1)

    return (
        X_train_general,
        X_val_general,
        X_test_general,
        X_train_catboost,
        X_val_catboost,
        X_test_catboost,
        CAT_FEATURES.copy(),
    )


def climate_context_aware_split(
    X: pd.DataFrame,
    y: pd.Series,
    y_time: pd.Series,
    station_ids: pd.Series,
    configuration: str,
):
    y_time = pd.to_datetime(y_time)
    year = y_time.dt.year

    train_mask = year.isin(TRAIN_YEARS)
    val_mask = year.isin(VALIDATION_YEARS)
    test_mask = year.isin(TEST_YEARS)

    # Validation year 2020 is intentionally isolated as the climate-extreme
    # early-stopping holdout. Because target-history features are built before
    # splitting, we conservatively remove early 2021 train targets whose lag or
    # rolling windows could still reference late-2020 validation observations.
    guard_hours = CONFIGURATIONS[configuration]["horizon"] + max_history_context_hours(configuration)
    validation_to_train_boundary = pd.Timestamp("2021-01-01")
    validation_guard_end = validation_to_train_boundary + pd.Timedelta(hours=guard_hours)
    train_mask &= ~((y_time >= validation_to_train_boundary) & (y_time < validation_guard_end))

    X_train, y_train, t_train = X.loc[train_mask], y.loc[train_mask], y_time.loc[train_mask]
    X_val, y_val, t_val = X.loc[val_mask], y.loc[val_mask], y_time.loc[val_mask]
    X_test, y_test, t_test = X.loc[test_mask], y.loc[test_mask], y_time.loc[test_mask]
    station_train = station_ids.loc[train_mask].reset_index(drop=True)
    station_val = station_ids.loc[val_mask].reset_index(drop=True)
    station_test = station_ids.loc[test_mask].reset_index(drop=True)

    y_train.index = pd.DatetimeIndex(t_train)
    y_val.index = pd.DatetimeIndex(t_val)
    y_test.index = pd.DatetimeIndex(t_test)
    X_train.index = y_train.index
    X_val.index = y_val.index
    X_test.index = y_test.index

    return X_train, X_val, X_test, y_train, y_val, y_test, station_train, station_val, station_test


def regression_metrics(y_true, y_pred):
    return {
        "N": int(len(y_true)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2_Score": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan,
    }


def scenario_masks(y_true: pd.Series):
    return {
        "Full Test Year 2025": (y_true.index >= "2025-01-01") & (y_true.index <= "2025-12-31 23:59:59"),
        "Non-Event Baseline (AQI <= 100)": y_true <= 100,
        "January 2025 Stress-Test Window": (
            (y_true.index >= "2025-01-07") & (y_true.index <= "2025-01-31 23:59:59")
        ),
        "Extreme AQI in Test 2025 (Top 5%)": y_true >= y_true.quantile(0.95),
        "Wildfire Season 2025 (Jul-Sep)": (
            (y_true.index >= "2025-07-01") & (y_true.index <= "2025-09-30 23:59:59")
        ),
    }


def evaluate_scenarios(configuration: str, model_name: str, y_true: pd.Series, y_pred: pd.Series):
    rows = []
    for scenario, mask in scenario_masks(y_true).items():
        scenario_y = y_true.loc[mask]
        scenario_pred = y_pred.loc[mask]
        if len(scenario_y) == 0:
            rows.append(
                {
                    "Configuration": configuration,
                    "Model": model_name,
                    "Scenario": scenario,
                    "Scenario_Status": "Unavailable in test split",
                    "N": 0,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2_Score": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "Configuration": configuration,
                    "Model": model_name,
                    "Scenario": scenario,
                    "Scenario_Status": "Available",
                    **regression_metrics(scenario_y, scenario_pred),
                }
            )
    return rows


def model_dictionary():
    return {
        "Linear Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.01,
            max_depth=4,
            num_leaves=15,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=700,
            learning_rate=0.01,
            depth=4,
            l2_leaf_reg=5,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            min_child_weight=5,
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def run_configuration(configuration: str):
    print("\n" + "=" * 90, flush=True)
    print(f"COMBINED PANEL MODEL TRAINING: {configuration}", flush=True)
    print("=" * 90, flush=True)

    raw = pd.read_csv(DATA_PATH)
    engineered = add_features(raw, configuration=configuration)
    X, y, y_time = build_feature_matrix(engineered, configuration=configuration)

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
    ) = climate_context_aware_split(X, y, y_time, station_ids, configuration)

    print(f"Dataset: {DATA_PATH}", flush=True)
    print(
        "Climate-context split: "
        f"Train years={TRAIN_YEARS}, Validation years={VALIDATION_YEARS}, Test years={TEST_YEARS}",
        flush=True,
    )
    print(
        f"Post-validation lag guard: {CONFIGURATIONS[configuration]['horizon'] + max_history_context_hours(configuration)} hours",
        flush=True,
    )
    print(f"Train period: {y_train.index.min()} to {y_train.index.max()} | rows={len(y_train):,}", flush=True)
    print(f"Val period:   {y_val.index.min()} to {y_val.index.max()} | rows={len(y_val):,}", flush=True)
    print(f"Test period:  {y_test.index.min()} to {y_test.index.max()} | rows={len(y_test):,}", flush=True)
    print(f"Raw feature count: {X_train.shape[1]}", flush=True)

    (
        X_train_general,
        X_val_general,
        X_test_general,
        X_train_catboost,
        X_val_catboost,
        X_test_catboost,
        cat_features,
    ) = prepare_model_matrices(X_train, X_val, X_test)
    print(f"General model feature count: {X_train_general.shape[1]}", flush=True)
    print(f"CatBoost categorical features: {cat_features}", flush=True)

    predictions = {}
    leaderboard_rows = []

    for name, model in model_dictionary().items():
        print(f"\nTraining {name}...", flush=True)
        start = time.time()
        if isinstance(model, lgb.LGBMRegressor):
            model.fit(
                X_train_general,
                y_train,
                eval_set=[(X_val_general, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
        elif isinstance(model, CatBoostRegressor):
            model.fit(
                X_train_catboost,
                y_train,
                eval_set=(X_val_catboost, y_val),
                cat_features=cat_features,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
        elif isinstance(model, XGBRegressor):
            model.fit(
                X_train_general,
                y_train,
                eval_set=[(X_val_general, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train_general, y_train)

        if isinstance(model, CatBoostRegressor):
            test_matrix = X_test_catboost
        else:
            test_matrix = X_test_general
        pred = pd.Series(model.predict(test_matrix), index=y_test.index, name="prediction")
        predictions[name] = (y_test, pred)
        leaderboard_rows.append(
            {
                "Configuration": configuration,
                "Model": name,
                **regression_metrics(y_test, pred),
                "Runtime_sec": round(time.time() - start, 2),
            }
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("R2_Score", ascending=False).reset_index(drop=True)

    scenario_rows = []
    for model_name, (model_y_true, pred) in predictions.items():
        scenario_rows.extend(evaluate_scenarios(configuration, model_name, model_y_true, pred))
    scenario_report = pd.DataFrame(scenario_rows)

    print("\nSorted Leaderboard", flush=True)
    print(leaderboard.to_string(index=False), flush=True)
    print("\nScenario Report", flush=True)
    print(scenario_report.sort_values(["Scenario", "R2_Score"], ascending=[True, False]).to_string(index=False), flush=True)

    prediction_rows = []
    for model_name, (model_y_true, pred) in predictions.items():
        prediction_rows.append(
            pd.DataFrame(
                {
                    "Configuration": configuration,
                    "Model": model_name,
                    "time": model_y_true.index,
                    "Actual_AQI": model_y_true.to_numpy(),
                    "Predicted_AQI": pred.to_numpy(),
                }
            )
        )
    prediction_report = pd.concat(prediction_rows, ignore_index=True)

    return leaderboard, scenario_report, prediction_report


if __name__ == "__main__":
    all_leaderboards = []
    all_scenarios = []
    all_predictions = []

    for configuration in CONFIGURATIONS:
        leaderboard, scenario_report, prediction_report = run_configuration(configuration)
        all_leaderboards.append(leaderboard)
        all_scenarios.append(scenario_report)
        all_predictions.append(prediction_report)

    final_leaderboard = pd.concat(all_leaderboards, ignore_index=True)
    final_scenarios = pd.concat(all_scenarios, ignore_index=True)
    final_predictions = pd.concat(all_predictions, ignore_index=True)

    final_leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    final_scenarios.to_csv(SCENARIO_PATH, index=False)
    final_predictions.to_csv(PREDICTION_PATH, index=False)

    print("\n" + "=" * 90, flush=True)
    print("FINAL COMBINED PANEL LEADERBOARD", flush=True)
    print("=" * 90, flush=True)
    print(final_leaderboard.sort_values(["Configuration", "R2_Score"], ascending=[True, False]).to_string(index=False), flush=True)

    print(f"\nSaved leaderboard: {LEADERBOARD_PATH}")
    print(f"Saved scenario report: {SCENARIO_PATH}")
    print(f"Saved predictions: {PREDICTION_PATH}")
