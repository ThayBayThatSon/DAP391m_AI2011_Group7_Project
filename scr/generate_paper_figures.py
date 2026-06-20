from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED = BASE_DIR / "data" / "processed"
PLOTS = BASE_DIR / "data" / "plots"
EDA_PLOTS = PLOTS / "eda"

MODEL_READY = PROCESSED / "california_aqi_model_ready.csv"
PREDICTIONS = PROCESSED / "california_aqi_model_predictions.csv"

AQI_COL = "target_aqi"


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def vapor_pressure_deficit_kpa(temperature_c: pd.Series, relative_humidity_pct: pd.Series) -> pd.Series:
    saturation_vapor_pressure = 0.6108 * np.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    actual_vapor_pressure = saturation_vapor_pressure * (relative_humidity_pct / 100.0)
    return (saturation_vapor_pressure - actual_vapor_pressure).clip(lower=0)


def short_config(configuration: str) -> str:
    if "Short-term" in configuration:
        return "Short-term nowcasting"
    if "Long-term" in configuration:
        return "24h forecasting"
    return configuration


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    columns = [
        AQI_COL,
        "pm25_ug_m3",
        "pm10_ug_m3",
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
    ]
    labels = {
        AQI_COL: "AQI Target",
        "pm25_ug_m3": "PM2.5",
        "pm10_ug_m3": "PM10",
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Relative Humidity",
        "rain": "Rainfall",
        "cloud_cover": "Cloud Cover",
        "wind_speed_10m": "Wind Speed",
        "wind_direction_10m": "Wind Direction",
        "surface_pressure": "Surface Pressure",
    }
    corr = df[columns].rename(columns=labels).corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        annot_kws={"size": 8.5},
        cbar_kws={"label": "Pearson correlation", "shrink": 0.82},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", labelrotation=0)
    save_figure(fig, PLOTS / "paper_correlation_heatmap_en.png")


def plot_vpd_diagnostic(df: pd.DataFrame) -> None:
    work = df[["time", AQI_COL, "temperature_2m", "relative_humidity_2m"]].copy()
    work["time"] = pd.to_datetime(work["time"])
    work["vpd_kpa"] = vapor_pressure_deficit_kpa(work["temperature_2m"], work["relative_humidity_2m"])
    monthly = (
        work.assign(month=work["time"].dt.month)
        .groupby("month", as_index=False)
        .agg(mean_aqi=(AQI_COL, "mean"), mean_vpd=("vpd_kpa", "mean"))
    )

    fig, ax1 = plt.subplots(figsize=(8.3, 4.3))
    ax2 = ax1.twinx()
    ax1.plot(monthly["month"], monthly["mean_aqi"], marker="o", color="#4C78A8", label="Mean AQI")
    ax2.plot(monthly["month"], monthly["mean_vpd"], marker="s", color="#D9822B", label="Mean VPD")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean AQI", color="#4C78A8")
    ax2.set_ylabel("Mean VPD (kPa)", color="#D9822B")
    ax1.set_xticks(range(1, 13))
    ax1.grid(color="#E6E6E6")
    ax1.set_title("Monthly AQI and Vapor Pressure Deficit Diagnostic")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=False)
    save_figure(fig, EDA_PLOTS / "07_vpd_monthly_diagnostic.png")


def plot_lightgbm_predicted_vs_actual(predictions: pd.DataFrame) -> None:
    lightgbm = predictions[predictions["Model"].eq("LightGBM")].copy()
    lightgbm["Task"] = lightgbm["Configuration"].map(short_config)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.9), sharex=True, sharey=True)
    for ax, task in zip(axes, ["Short-term nowcasting", "24h forecasting"]):
        sub = lightgbm[lightgbm["Task"].eq(task)]
        if len(sub) > 7000:
            sub = sub.sample(7000, random_state=42)
        ax.scatter(
            sub["Actual_AQI"],
            sub["Predicted_AQI"],
            s=7,
            alpha=0.28,
            color="#4C78A8" if task.startswith("Short") else "#D9822B",
            edgecolor="none",
        )
        lim = float(max(sub["Actual_AQI"].max(), sub["Predicted_AQI"].max()))
        ax.plot([0, lim], [0, lim], color="#333333", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Actual AQI")
        ax.text(0.03, 0.94, task, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
        ax.grid(color="#E6E6E6")
    axes[0].set_ylabel("Predicted AQI")
    sns.despine(fig=fig)
    save_figure(fig, PLOTS / "paper_lightgbm_predicted_vs_actual.png")


def plot_extreme_heatmap(df: pd.DataFrame) -> None:
    work = df[["time", AQI_COL]].copy()
    work["time"] = pd.to_datetime(work["time"])
    threshold = work[AQI_COL].quantile(0.99)
    extreme = work[work[AQI_COL] > threshold].copy()
    extreme["year"] = extreme["time"].dt.year
    extreme["month"] = extreme["time"].dt.month
    matrix = (
        extreme.pivot_table(index="year", columns="month", values=AQI_COL, aggfunc="size", fill_value=0)
        .reindex(index=sorted(work["time"].dt.year.unique()), columns=range(1, 13), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Reds",
        linewidths=0,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Extreme hours"},
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Top 1% AQI Event Hours by Year and Month")
    save_figure(fig, EDA_PLOTS / "06_extreme_aqi_events_heatmap_year_month.png")


def main() -> None:
    configure_style()
    df = pd.read_csv(MODEL_READY)
    predictions = pd.read_csv(PREDICTIONS)

    plot_correlation_heatmap(df)
    plot_vpd_diagnostic(df)
    plot_lightgbm_predicted_vs_actual(predictions)
    plot_extreme_heatmap(df)


if __name__ == "__main__":
    main()
