# -*- coding: utf-8 -*-
"""EDA toàn diện cho dữ liệu chất lượng không khí và khí tượng California.

Chạy mặc định:
    python scr/eda_california_air_quality.py

Chạy với file CSV khác:
    python scr/eda_california_air_quality.py --data-path data/processed/your_file.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_PATH = Path("data/processed/california_aqi_merged_panel.csv")
DEFAULT_OUTPUT_DIR = Path("data/plots/eda")
DEFAULT_REPORT_DIR = Path("data/processed/eda_reports")

TIME_COLUMNS = [
    "time",
    "hour",
    "dayofweek",
    "month",
    "year",
    "station_id",
    "station_name",
    "lat",
    "lon",
    "source",
]

AIR_QUALITY_COLUMNS = [
    "pm25_ug_m3",
    "pm10_ug_m3",
    "target_aqi",
    "target_type",
]

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]

IMPORTANT_UNIVARIATE_COLUMNS = [
    "target_aqi",
    "pm25_ug_m3",
    "temperature_2m",
    "wind_speed_10m",
]


def configure_plot_style() -> None:
    """Thiết lập style chung để biểu đồ dễ đọc và không bị đè chữ."""
    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["axes.unicode_minus"] = False


def configure_console_encoding() -> None:
    """Giúp Windows console/redirect in được tiếng Việt có dấu ổn định."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_data(data_path: Path) -> pd.DataFrame:
    """Đọc CSV và parse cột thời gian nếu có."""
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    df = pd.read_csv(data_path)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa kiểu dữ liệu cho các cột số và tạo cột thời gian phụ nếu cần."""
    df = df.copy()

    numeric_candidates = [
        "hour",
        "dayofweek",
        "month",
        "year",
        "lat",
        "lon",
        *AIR_QUALITY_COLUMNS,
        *WEATHER_COLUMNS,
    ]

    for col in numeric_candidates:
        if col in df.columns and col != "target_type":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "time" in df.columns:
        valid_time = df["time"].notna()
        if "hour" not in df.columns:
            df["hour"] = df["time"].dt.hour.where(valid_time)
        if "dayofweek" not in df.columns:
            df["dayofweek"] = df["time"].dt.dayofweek.where(valid_time)
        if "month" not in df.columns:
            df["month"] = df["time"].dt.month.where(valid_time)
        if "year" not in df.columns:
            df["year"] = df["time"].dt.year.where(valid_time)

    return df


def print_section(title: str) -> None:
    """In tiêu đề từng phần báo cáo."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_table(table: pd.DataFrame, output_path: Path) -> None:
    """Lưu bảng kết quả EDA thành CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=True, encoding="utf-8-sig")


def overview_analysis(df: pd.DataFrame, report_dir: Path) -> None:
    """1. Kiểm tra tổng quan: shape, dtype, missing values, describe."""
    print_section("1. KIỂM TRA TỔNG QUAN DỮ LIỆU")

    print(f"Kích thước dữ liệu: {df.shape[0]:,} dòng x {df.shape[1]:,} cột")

    print("\nKiểu dữ liệu từng cột:")
    print(df.dtypes.to_string())

    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
        }
    ).sort_values(["missing_percent", "missing_count"], ascending=False)

    print("\nGiá trị thiếu theo cột:")
    print(missing.to_string())
    save_table(missing, report_dir / "missing_values_report.csv")

    print("\nThống kê mô tả cho biến số:")
    numeric_describe = df.describe(include="number").T.round(3)
    print(numeric_describe.to_string())
    save_table(numeric_describe, report_dir / "numeric_descriptive_statistics.csv")

    print("\nThống kê mô tả tổng quát:")
    all_describe = df.describe(include="all").T
    print(all_describe.to_string())
    save_table(all_describe, report_dir / "all_descriptive_statistics.csv")


def plot_univariate_analysis(df: pd.DataFrame, output_dir: Path, show: bool = False) -> None:
    """2. Vẽ histogram/KDE và boxplot cho các biến quan trọng."""
    print_section("2. PHÂN TÍCH ĐƠN BIẾN VÀ NGOẠI LAI")

    available_cols = [col for col in IMPORTANT_UNIVARIATE_COLUMNS if col in df.columns]
    missing_cols = sorted(set(IMPORTANT_UNIVARIATE_COLUMNS) - set(available_cols))

    if missing_cols:
        print(f"Bỏ qua các cột không tồn tại: {missing_cols}")

    if not available_cols:
        print("Không có cột nào phù hợp để vẽ phân tích đơn biến.")
        return

    fig, axes = plt.subplots(
        nrows=len(available_cols),
        ncols=2,
        figsize=(15, 4.2 * len(available_cols)),
        constrained_layout=True,
    )

    if len(available_cols) == 1:
        axes = [axes]

    for row_index, col in enumerate(available_cols):
        series = df[col].dropna()
        ax_hist, ax_box = axes[row_index]

        sns.histplot(series, kde=True, bins=40, color="#4C78A8", ax=ax_hist)
        ax_hist.set_title(f"Phân phối của {col}")
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel("Tần suất")

        sns.boxplot(x=series, color="#F58518", ax=ax_box)
        ax_box.set_title(f"Boxplot kiểm tra ngoại lai của {col}")
        ax_box.set_xlabel(col)

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((series < lower) | (series > upper)).sum()

        print(
            f"{col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
            f"ngưỡng ngoại lai=({lower:.2f}, {upper:.2f}), "
            f"số ngoại lai={outlier_count:,}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_univariate_histogram_boxplot.png"
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Đã lưu biểu đồ đơn biến: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def summarize_target_by_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "target_aqi",
) -> pd.DataFrame:
    """Thống kê target_aqi theo một cột nhóm."""
    summary = (
        df.dropna(subset=[group_col, target_col])
        .groupby(group_col, observed=True)[target_col]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .round(3)
    )
    return summary


def time_series_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    report_dir: Path,
    show: bool = False,
) -> None:
    """3. Phân tích xu hướng target_aqi theo giờ, tháng và năm."""
    print_section("3. PHÂN TÍCH CHUỖI THỜI GIAN")

    target_col = "target_aqi"
    if target_col not in df.columns:
        raise KeyError("Dataset cần có cột target_aqi để phân tích chuỗi thời gian.")

    group_specs = [
        ("hour", "Giờ trong ngày", "Giờ", list(range(24))),
        ("month", "Tháng trong năm", "Tháng", list(range(1, 13))),
        ("year", "Năm 2018-2025", "Năm", list(range(2018, 2026))),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 14), constrained_layout=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ax, (group_col, title, xlabel, full_index) in zip(axes, group_specs):
        if group_col not in df.columns:
            ax.axis("off")
            ax.set_title(f"Không có cột {group_col}")
            print(f"Bỏ qua phân tích theo {group_col} vì cột không tồn tại.")
            continue

        plot_df = df.copy()
        if group_col == "year":
            plot_df = plot_df[(plot_df["year"] >= 2018) & (plot_df["year"] <= 2025)]

        summary = summarize_target_by_group(plot_df, group_col, target_col)
        summary = summary.reindex(full_index)
        save_table(summary, report_dir / f"target_aqi_by_{group_col}.csv")

        print(f"\nThống kê target_aqi theo {group_col}:")
        print(summary.to_string())

        sns.lineplot(
            data=summary.reset_index(names=group_col),
            x=group_col,
            y="mean",
            marker="o",
            linewidth=2.3,
            color="#D62728",
            ax=ax,
        )
        ax.fill_between(
            summary.index,
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            color="#D62728",
            alpha=0.12,
            label="mean +/- std",
        )
        ax.set_title(f"Xu hướng AQI trung bình theo {title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("target_aqi trung bình")
        ax.set_xticks(full_index)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    output_path = output_dir / "02_target_aqi_time_patterns.png"
    fig.savefig(output_path, bbox_inches="tight")
    print(f"\nĐã lưu biểu đồ chuỗi thời gian: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def correlation_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    report_dir: Path,
    show: bool = False,
) -> None:
    """4. Tính và vẽ heatmap tương quan giữa các biến số."""
    print_section("4. PHÂN TÍCH TƯƠNG QUAN")

    target_col = "target_aqi"
    if target_col not in df.columns:
        raise KeyError("Dataset cần có cột target_aqi để phân tích tương quan.")

    preferred_order = [
        "target_aqi",
        "pm25_ug_m3",
        "pm10_ug_m3",
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
        "hour",
        "dayofweek",
        "month",
        "year",
        "lat",
        "lon",
    ]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    ordered_cols = [col for col in preferred_order if col in numeric_cols]
    ordered_cols += [col for col in numeric_cols if col not in ordered_cols]

    corr = df[ordered_cols].corr(method="pearson")
    save_table(corr.round(4), report_dir / "correlation_matrix.csv")

    print("\nTương quan của các biến với target_aqi:")
    corr_with_target = corr[target_col].drop(target_col).sort_values(ascending=False).round(4)
    print(corr_with_target.to_string())
    save_table(
        corr_with_target.to_frame("correlation_with_target_aqi"),
        report_dir / "correlation_with_target_aqi.csv",
    )

    weather_corr_cols = [col for col in WEATHER_COLUMNS if col in corr.index]
    if weather_corr_cols:
        weather_corr = corr.loc[weather_corr_cols, target_col].sort_values(ascending=False)
        print("\nLiên hệ tương quan giữa thời tiết và target_aqi:")
        for col, value in weather_corr.items():
            strength = describe_correlation_strength(value)
            direction = "dương" if value > 0 else "âm" if value < 0 else "gần như bằng 0"
            print(f"- {col}: r={value:.4f} ({strength}, tương quan {direction})")
        print(
            "\nLưu ý: tương quan chỉ cho thấy mức độ đồng biến/nghịch biến, "
            "không chứng minh quan hệ nhân quả trực tiếp."
        )

    fig_height = max(8, 0.55 * len(ordered_cols))
    fig_width = max(12, 0.7 * len(ordered_cols))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        square=False,
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Ma trận tương quan giữa các biến số")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "03_correlation_heatmap.png"
    fig.savefig(output_path, bbox_inches="tight")
    print(f"\nĐã lưu heatmap tương quan: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def describe_correlation_strength(value: float) -> str:
    """Diễn giải nhanh độ mạnh/yếu của hệ số tương quan Pearson."""
    abs_value = abs(value)
    if abs_value >= 0.7:
        return "mạnh"
    if abs_value >= 0.4:
        return "trung bình"
    if abs_value >= 0.2:
        return "yếu"
    return "rất yếu"


def station_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    report_dir: Path,
    top_n: int = 20,
    show: bool = False,
) -> None:
    """5. So sánh target_aqi trung bình giữa các trạm đo."""
    print_section("5. PHÂN TÍCH THEO TRẠM ĐO")

    target_col = "target_aqi"
    station_col = "station_name"

    if target_col not in df.columns:
        raise KeyError("Dataset cần có cột target_aqi để phân tích theo trạm.")
    if station_col not in df.columns:
        raise KeyError("Dataset cần có cột station_name để phân tích theo trạm.")

    station_stats = (
        df.dropna(subset=[station_col, target_col])
        .groupby(station_col, observed=True)[target_col]
        .agg(
            observation_count="count",
            mean_target_aqi="mean",
            median_target_aqi="median",
            std_target_aqi="std",
            min_target_aqi="min",
            max_target_aqi="max",
        )
        .sort_values("mean_target_aqi", ascending=False)
        .round(3)
    )

    print("\nBảng xếp hạng trạm đo theo target_aqi trung bình:")
    print(station_stats.to_string())
    save_table(station_stats, report_dir / "station_target_aqi_ranking.csv")

    most_polluted_station = station_stats.index[0]
    most_polluted_aqi = station_stats.iloc[0]["mean_target_aqi"]
    print(
        f"\nTrạm có AQI trung bình cao nhất: {most_polluted_station} "
        f"({most_polluted_aqi:.2f})"
    )

    top_station_stats = station_stats.head(top_n).reset_index()
    fig_height = max(6, 0.45 * len(top_station_stats))
    fig, ax = plt.subplots(figsize=(13, fig_height), constrained_layout=True)
    sns.barplot(
        data=top_station_stats,
        y=station_col,
        x="mean_target_aqi",
        hue=station_col,
        dodge=False,
        palette="Reds_r",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Top {len(top_station_stats)} trạm đo có target_aqi trung bình cao nhất")
    ax.set_xlabel("target_aqi trung bình")
    ax.set_ylabel("Trạm đo")

    max_value = top_station_stats["mean_target_aqi"].max()
    for index, value in enumerate(top_station_stats["mean_target_aqi"]):
        ax.text(
            value + max_value * 0.01,
            index,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_xlim(0, max_value * 1.15)
    ax.grid(True, axis="x", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "04_station_target_aqi_ranking.png"
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Đã lưu biểu đồ xếp hạng trạm đo: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="EDA cho dataset chất lượng không khí và khí tượng California."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Đường dẫn file CSV đầu vào. Mặc định: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục lưu biểu đồ. Mặc định: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Thư mục lưu bảng kết quả CSV. Mặc định: {DEFAULT_REPORT_DIR}",
    )
    parser.add_argument(
        "--top-stations",
        type=int,
        default=20,
        help="Số trạm đo hiển thị trong biểu đồ xếp hạng.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị biểu đồ sau khi lưu file PNG.",
    )
    return parser.parse_args()


def main() -> None:
    """Chạy toàn bộ quy trình EDA."""
    args = parse_args()
    configure_console_encoding()
    configure_plot_style()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_rows", 120)

    print(f"Đọc dữ liệu từ: {args.data_path}")
    df = load_data(args.data_path)
    df = prepare_columns(df)

    if "target_aqi" not in df.columns:
        raise KeyError("Dataset bắt buộc cần cột target_aqi.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    overview_analysis(df, args.report_dir)
    plot_univariate_analysis(df, args.output_dir, show=args.show)
    time_series_analysis(df, args.output_dir, args.report_dir, show=args.show)
    correlation_analysis(df, args.output_dir, args.report_dir, show=args.show)
    station_analysis(
        df,
        args.output_dir,
        args.report_dir,
        top_n=args.top_stations,
        show=args.show,
    )

    print_section("HOÀN TẤT EDA")
    print(f"Các biểu đồ đã lưu trong: {args.output_dir}")
    print(f"Các bảng báo cáo đã lưu trong: {args.report_dir}")


if __name__ == "__main__":
    main()
