import time

import lightgbm as lgb
import pandas as pd

from train_combined_panel_models import (
    CONFIGURATIONS,
    DATA_PATH,
    EARLY_STOPPING_ROUNDS,
    OUTPUT_DIR,
    add_features,
    build_feature_matrix,
    climate_context_aware_split,
    model_dictionary,
    prepare_model_matrices,
    regression_metrics,
)


ABLATION_PATH = f"{OUTPUT_DIR}/california_aqi_lightgbm_ablation.csv"


FEATURE_SETS = {
    "Full feature set": [],
    "Without VPD": ["vpd_kpa"],
    "Without spatial lags": ["spatial_"],
}


def drop_ablation_features(X: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    drop_rules = FEATURE_SETS[feature_set]
    if not drop_rules:
        return X.copy()

    drop_columns = []
    for column in X.columns:
        for rule in drop_rules:
            if rule.endswith("_"):
                if column.startswith(rule):
                    drop_columns.append(column)
            elif column == rule:
                drop_columns.append(column)

    return X.drop(columns=sorted(set(drop_columns)), errors="ignore")


def run_lightgbm_ablation(configuration: str) -> list[dict]:
    raw = pd.read_csv(DATA_PATH)
    engineered = add_features(raw, configuration=configuration)
    X, y, y_time = build_feature_matrix(engineered, configuration=configuration)
    station_ids = engineered.loc[X.index, "station_id"].reset_index(drop=True)

    rows = []
    for feature_set in FEATURE_SETS:
        X_variant = drop_ablation_features(X, feature_set).reset_index(drop=True)
        y_variant = y.reset_index(drop=True)
        y_time_variant = y_time.reset_index(drop=True)

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            _train_station_ids,
            _val_station_ids,
            _test_station_ids,
        ) = climate_context_aware_split(
            X_variant,
            y_variant,
            y_time_variant,
            station_ids,
            configuration,
        )

        (
            X_train_general,
            X_val_general,
            X_test_general,
            _X_train_catboost,
            _X_val_catboost,
            _X_test_catboost,
            _cat_features,
        ) = prepare_model_matrices(X_train, X_val, X_test)

        model = model_dictionary()["LightGBM"]
        start = time.time()
        model.fit(
            X_train_general,
            y_train,
            eval_set=[(X_val_general, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        pred = pd.Series(model.predict(X_test_general), index=y_test.index)
        rows.append(
            {
                "Configuration": configuration,
                "Model": "LightGBM",
                "Feature_Set": feature_set,
                "Feature_Count": int(X_train_general.shape[1]),
                **regression_metrics(y_test, pred),
                "Runtime_sec": round(time.time() - start, 2),
            }
        )

    return rows


def main() -> None:
    rows = []
    for configuration in CONFIGURATIONS:
        rows.extend(run_lightgbm_ablation(configuration))

    result = pd.DataFrame(rows)
    full = result[result["Feature_Set"].eq("Full feature set")][
        ["Configuration", "MAE", "RMSE", "R2_Score"]
    ].rename(columns={"MAE": "Full_MAE", "RMSE": "Full_RMSE", "R2_Score": "Full_R2"})
    result = result.merge(full, on="Configuration", how="left")
    result["Delta_MAE_vs_Full"] = result["MAE"] - result["Full_MAE"]
    result["Delta_R2_vs_Full"] = result["R2_Score"] - result["Full_R2"]
    result = result.drop(columns=["Full_MAE", "Full_RMSE", "Full_R2"])
    result.to_csv(ABLATION_PATH, index=False)
    print(result.to_string(index=False))
    print(f"Saved ablation report: {ABLATION_PATH}")


if __name__ == "__main__":
    main()
