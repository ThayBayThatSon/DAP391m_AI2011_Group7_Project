from __future__ import annotations

import os
import tempfile
from pathlib import Path

RUNTIME_CACHE = Path(tempfile.gettempdir()) / "aqi_station_map_runtime_cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("CARTOPY_DATA_DIR", str(RUNTIME_CACHE / "cartopy"))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from cartopy.io import shapereader

cartopy.config["data_dir"] = RUNTIME_CACHE / "cartopy"


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_READY = BASE_DIR / "data" / "processed" / "california_aqi_model_ready.csv"
OUTPUT = BASE_DIR / "data" / "plots" / "paper_station_map.png"

MAP_EXTENT = (-124.8, -113.8, 32.0, 42.4)
SOURCE_MARKERS = {
    "epa_aqs_station": "o",
    "openmeteo": "^",
}
SOURCE_LABELS = {
    "epa_aqs_station": "EPA AQS station",
    "openmeteo": "Open-Meteo point",
}


def load_california_boundary() -> gpd.GeoDataFrame:
    """Load the official California boundary from Natural Earth admin-1 data."""
    shp_path = shapereader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_1_states_provinces",
    )
    states = gpd.read_file(shp_path).to_crs("EPSG:4326")
    california = states[
        (states["admin"] == "United States of America")
        & (states["name"].str.lower() == "california")
    ].copy()
    if california.empty:
        raise RuntimeError("California boundary was not found in Natural Earth admin-1 data.")
    return california


def load_station_summary() -> gpd.GeoDataFrame:
    """Read true station coordinates from the model-ready panel and aggregate mean AQI."""
    usecols = ["source", "station_id", "station_name", "lat", "lon", "target_aqi"]
    df = pd.read_csv(MODEL_READY, usecols=usecols)
    summary = (
        df.groupby(["source", "station_id", "station_name", "lat", "lon"], dropna=False)
        .agg(rows=("target_aqi", "size"), mean_aqi=("target_aqi", "mean"))
        .reset_index()
        .sort_values(["source", "station_id"])
    )
    return gpd.GeoDataFrame(
        summary,
        geometry=gpd.points_from_xy(summary["lon"], summary["lat"]),
        crs="EPSG:4326",
    )


def annotate_stations(ax: plt.Axes, stations: gpd.GeoDataFrame) -> None:
    """Add compact, publication-friendly labels with small offsets."""
    offsets = {
        "FRES": (-0.95, 0.25),
        "FRES_OPENMETEO": (0.25, -0.35),
        "LA": (-0.85, -0.38),
        "LA_OPENMETEO": (0.25, 0.32),
        "SJ_OPENMETEO": (0.22, 0.32),
    }
    for row in stations.itertuples(index=False):
        dx, dy = offsets.get(row.station_id, (0.22, 0.22))
        label = f"{row.station_id}\nMean AQI={row.mean_aqi:.1f}"
        ax.annotate(
            label,
            xy=(row.lon, row.lat),
            xytext=(row.lon + dx, row.lat + dy),
            textcoords="data",
            fontsize=7.5,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#8a8a8a", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7),
            transform=ccrs.PlateCarree(),
            zorder=6,
        )


def plot_station_map() -> None:
    california = load_california_boundary()
    stations = load_station_summary()

    fig = plt.figure(figsize=(7.2, 7.8))
    ax = fig.add_axes([0.06, 0.08, 0.76, 0.86], projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dcecf7", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f4f1e8", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#b3b3b3", linewidth=0.6, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#4f6f8f", linewidth=0.8, zorder=2)

    ax.add_geometries(
        california.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="#f7f1e4",
        edgecolor="#222222",
        linewidth=1.25,
        zorder=3,
    )

    vmin = stations["mean_aqi"].min()
    vmax = stations["mean_aqi"].max()
    scatter_for_colorbar = None
    for source, group in stations.groupby("source"):
        scatter = ax.scatter(
            group["lon"],
            group["lat"],
            c=group["mean_aqi"],
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            s=145,
            marker=SOURCE_MARKERS.get(source, "o"),
            edgecolors="#222222",
            linewidths=0.9,
            label=SOURCE_LABELS.get(source, source),
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        scatter_for_colorbar = scatter

    annotate_stations(ax, stations)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.45,
        color="#bdbdbd",
        alpha=0.65,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=8)

    if scatter_for_colorbar is not None:
        cax = fig.add_axes([0.86, 0.18, 0.035, 0.68])
        cbar = fig.colorbar(scatter_for_colorbar, cax=cax)
        cbar.set_label("Mean AQI Target", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    plot_station_map()
