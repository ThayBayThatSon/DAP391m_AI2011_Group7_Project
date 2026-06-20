import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

CSV_PATH = os.path.join('data', 'processed', 'openmeteo_california_hourly_raw.csv')
PLOT_DIR = os.path.join('data', 'plots')
PLOT_FILE = os.path.join(PLOT_DIR, 'aqi_visualization.png')

AQI_CATEGORIES = [
    {
        'label': 'Good',
        'range': '0-50',
        'low': 0,
        'high': 50,
        'color': '#00E400',
        'meaning': 'Good air quality; little or no health risk.',
    },
    {
        'label': 'Moderate',
        'range': '51-100',
        'low': 51,
        'high': 100,
        'color': '#FFFF00',
        'meaning': 'Acceptable; unusually sensitive people may be affected.',
    },
    {
        'label': 'Unhealthy for Sensitive Groups (USG)',
        'range': '101-150',
        'low': 101,
        'high': 150,
        'color': '#FF7E00',
        'meaning': 'Older adults, children, and people with heart/lung disease may be affected.',
    },
    {
        'label': 'Unhealthy',
        'range': '151-200',
        'low': 151,
        'high': 200,
        'color': '#FF0000',
        'meaning': 'Health effects may begin for the general population.',
    },
    {
        'label': 'Very Unhealthy',
        'range': '201-300',
        'low': 201,
        'high': 300,
        'color': '#8F3F97',
        'meaning': 'Health alert: increased risk for everyone.',
    },
    {
        'label': 'Hazardous',
        'range': '301-500',
        'low': 301,
        'high': 500,
        'color': '#7E0023',
        'meaning': 'Emergency health warning.',
    },
]


def load_data(filepath: str = CSV_PATH) -> pd.DataFrame:
    """Load the processed data file for visualization."""
    df = pd.read_csv(filepath, parse_dates=['time'])
    return df.sort_values('time').reset_index(drop=True)


def find_pm25_column(df: pd.DataFrame) -> str | None:
    """Find a PM2.5 column even when unit text is encoded differently."""
    for col in df.columns:
        normalized = ''.join(ch.lower() for ch in col if ch.isalnum())
        if 'pm25' in normalized or 'pm2' in normalized:
            return col
    return None


def aqi_color(value: float) -> str:
    """Return the official US AQI category color for a numeric AQI value."""
    for category in AQI_CATEGORIES:
        if category['low'] <= value <= category['high']:
            return category['color']
    if value > AQI_CATEGORIES[-1]['high']:
        return AQI_CATEGORIES[-1]['color']
    return AQI_CATEGORIES[0]['color']


def add_aqi_background_bands(ax) -> None:
    """Add subtle horizontal US AQI category bands."""
    for category in AQI_CATEGORIES:
        ax.axhspan(
            category['low'],
            category['high'],
            color=category['color'],
            alpha=0.08,
            linewidth=0,
        )


def plot_aqi_colored_line(ax, df: pd.DataFrame) -> None:
    """Plot AQI as connected line segments colored by US AQI category."""
    plot_df = df[['time', 'us_aqi']].dropna().sort_values('time')
    if len(plot_df) < 2:
        ax.plot(plot_df['time'], plot_df['us_aqi'], color='tab:red', linewidth=1.2)
        return

    x = mdates.date2num(plot_df['time'])
    y = plot_df['us_aqi'].to_numpy()
    points = list(zip(x, y))
    segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
    segment_values = (y[:-1] + y[1:]) / 2.0
    segment_colors = [aqi_color(value) for value in segment_values]

    collection = LineCollection(segments, colors=segment_colors, linewidths=1.4, alpha=0.95)
    ax.add_collection(collection)
    ax.xaxis_date()
    ax.set_xlim(plot_df['time'].min(), plot_df['time'].max())
    ax.set_ylim(max(0, y.min() - 10), min(500, max(160, y.max() + 20)))


def aqi_legend_handles() -> list[Line2D]:
    """Build legend handles for US AQI health categories."""
    return [
        Line2D(
            [0],
            [0],
            color=category['color'],
            lw=3,
            label=f"{category['range']} | {category['label']}",
        )
        for category in AQI_CATEGORIES
    ]


def plot_aqi(df: pd.DataFrame, output_path: str = PLOT_FILE) -> None:
    """Draw US AQI visualizations and save the plot to disk."""
    if 'us_aqi' not in df.columns:
        raise KeyError('Column us_aqi not found in dataset')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pm25_col = find_pm25_column(df)

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), constrained_layout=True)

    add_aqi_background_bands(axs[0])
    plot_aqi_colored_line(axs[0], df)
    if pm25_col is not None:
        axs[0].plot(df['time'], df[pm25_col], label='PM2.5', color='tab:blue', linewidth=1, alpha=0.7)

    axs[0].set_title('US AQI over Time Colored by Official US AQI Health Category')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('US AQI / PM2.5 Value')
    handles = aqi_legend_handles()
    if pm25_col is not None:
        handles.append(Line2D([0], [0], color='tab:blue', lw=2, label='PM2.5'))
    axs[0].legend(handles=handles, loc='upper right', fontsize=8, frameon=True)
    axs[0].grid(True, alpha=0.25)

    if pm25_col is not None:
        scatter_colors = df['us_aqi'].apply(aqi_color)
        add_aqi_background_bands(axs[1])
        axs[1].scatter(df[pm25_col], df['us_aqi'], s=10, alpha=0.65, color=scatter_colors)
        axs[1].set_title('US AQI vs PM2.5 Colored by US AQI Category')
        axs[1].set_xlabel('PM2.5')
        axs[1].set_ylabel('US AQI')
        axs[1].set_ylim(max(0, df['us_aqi'].min() - 10), min(500, max(160, df['us_aqi'].max() + 20)))
        axs[1].grid(True, alpha=0.25)
    else:
        axs[1].axis('off')
        axs[1].text(0.5, 0.5, 'PM2.5 column not found', ha='center', va='center', fontsize=14)

    fig.savefig(output_path, dpi=200)
    print(f'Saved AQI visualization to: {output_path}')
    plt.show()


def print_summary(df: pd.DataFrame) -> None:
    """Print basic US AQI statistics to the console."""
    if 'us_aqi' not in df.columns:
        raise KeyError('Column us_aqi not found in dataset')

    summary = df['us_aqi'].describe()
    print('US AQI summary:')
    print(summary)


if __name__ == '__main__':
    df = load_data()
    print_summary(df)
    plot_aqi(df)
