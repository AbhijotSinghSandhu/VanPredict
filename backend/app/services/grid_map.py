from pathlib import Path
import math

import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import LinearColormap
from folium.plugins import HeatMap, MarkerCluster


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def create_grid_map():
    grid = gpd.read_file(
        PROCESSED_DIR / "vancouver_grid.geojson"
    ).to_crs("EPSG:4326")

    reports = pd.read_csv(PROCESSED_DIR / "311_gridded.csv")
    unassigned = pd.read_csv(
        PROCESSED_DIR / "311_outside_grid.csv"
    )

    # Recalculate counts from the assigned reports.
    counts = reports.groupby("grid_id").size()
    grid["report_count"] = (
        grid["grid_id"].map(counts).fillna(0).astype(int)
    )

    metric_grid = grid.to_crs("EPSG:32610")
    grid["area_m2"] = metric_grid.geometry.area.round(2)

    max_count = max(1, int(grid["report_count"].max()))

    # Log scale makes differences among low-count cells visible.
    colours = LinearColormap(
        ["#fff7bc", "#fec44f", "#f03b20", "#800026"],
        vmin=0,
        vmax=math.log1p(max_count),
        caption="Historical report count: log(1 + count)",
    )

    def grid_style(feature):
        count = feature["properties"]["report_count"]

        return {
            "fillColor": (
                "#cbd5e1" if count == 0
                else colours(math.log1p(count))
            ),
            "color": "#475569",
            "weight": 0.6,
            "fillOpacity": 0.45,
        }

    map_view = folium.Map(
        location=[49.25, -123.12],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    folium.GeoJson(
        grid.to_json(),
        name="500 m grid — historical counts",
        style_function=grid_style,
        highlight_function=lambda feature: {
            "weight": 2,
            "color": "#0f172a",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["grid_id", "report_count", "area_m2"],
            aliases=["Grid:", "Historical reports:", "Area (m²):"],
            localize=True,
        ),
    ).add_to(map_view)

    colours.add_to(map_view)

    # All assigned reports; repeated locations contribute repeatedly.
    heat_layer = folium.FeatureGroup(
        name=f"All {len(reports):,} reports — heatmap",
        show=False,
    )

    HeatMap(
        reports[["latitude", "longitude"]].values.tolist(),
        radius=12,
        blur=15,
    ).add_to(heat_layer)

    heat_layer.add_to(map_view)

    # A reproducible sample provides inspectable markers.
    sample = reports.sample(
        n=min(2000, len(reports)),
        random_state=42,
    )

    marker_layer = MarkerCluster(
        name=f"Report markers — sample of {len(sample):,}",
        show=False,
    ).add_to(map_view)

    for _, report in sample.iterrows():
        folium.Marker(
            location=[report["latitude"], report["longitude"]],
            tooltip=(
                f"{report['grid_id']} | "
                f"{report['service_request_type']} | "
                f"{report['service_request_open_timestamp']}"
            ),
        ).add_to(marker_layer)

    outside_layer = folium.FeatureGroup(
        name=f"Unassigned reports ({len(unassigned)})",
        show=True,
    ).add_to(map_view)

    for _, report in unassigned.iterrows():
        folium.CircleMarker(
            location=[report["latitude"], report["longitude"]],
            radius=7,
            color="#7c3aed",
            fill=True,
            fill_color="#7c3aed",
            fill_opacity=1,
            tooltip=(
                f"UNASSIGNED | {report['service_request_type']} | "
                f"{report['service_request_open_timestamp']}"
            ),
        ).add_to(outside_layer)

    # Diagnostic threshold: less than 1% of a full 500 m square.
    tiny_cells = grid.loc[grid["area_m2"] < 2500]

    if not tiny_cells.empty:
        folium.GeoJson(
            tiny_cells.to_json(),
            name="Tiny boundary cells — under 2,500 m²",
            show=False,
            style_function=lambda feature: {
                "color": "#2563eb",
                "weight": 3,
                "fillColor": "#2563eb",
                "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["grid_id", "area_m2", "report_count"],
            ),
        ).add_to(map_view)

    # Include the unassigned locations in the initial view.
    min_x, min_y, max_x, max_y = grid.total_bounds

    if not unassigned.empty:
        min_x = min(min_x, unassigned["longitude"].min())
        max_x = max(max_x, unassigned["longitude"].max())
        min_y = min(min_y, unassigned["latitude"].min())
        max_y = max(max_y, unassigned["latitude"].max())

    map_view.fit_bounds([[min_y, min_x], [max_y, max_x]])

    folium.LayerControl(collapsed=False).add_to(map_view)

    return map_view.get_root().render()