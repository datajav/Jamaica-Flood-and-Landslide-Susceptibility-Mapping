import argparse
import sys

import geopandas as gpd
import pandas as pd

from paths import PROCESSED, RAW

CRS_M = 3448  # JAD2001 / Jamaica Metric Grid, for distances and lengths in meters

NON_VEHICLE = {"footway", "path", "cycleway", "pedestrian", "steps", "bridleway"}
MAIN = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
}


def load_roads(path):
    if not path.exists():
        sys.exit(f"Roads file not found: {path}")
    roads = gpd.read_file(path)
    roads = roads[roads.geometry.notna() & roads.geom_type.isin(["LineString", "MultiLineString"])]
    if "fclass" in roads.columns:
        roads = roads[~roads["fclass"].isin(NON_VEHICLE)]
    else:
        print("Warning: no 'fclass' column; using every line feature as a road.")
    print(f"{len(roads):,} road features after filtering")
    return roads.to_crs(CRS_M)


def nearest_distance(points, lines, name):
    j = gpd.sjoin_nearest(points[["cell", "geometry"]], lines[["geometry"]], distance_col=name, how="left")
    return j.drop_duplicates("cell")[["cell", name]]


def main(roads_path, res):
    grid_path = PROCESSED / f"grid_res{res}.parquet"
    if not grid_path.exists():
        sys.exit(f"Grid not found: {grid_path}")

    roads = load_roads(roads_path)
    cells = gpd.read_parquet(grid_path)[["cell", "geometry"]].to_crs(CRS_M)

    points = cells.copy()
    points["geometry"] = cells.geometry.centroid

    out = cells[["cell"]].copy()
    out = out.merge(nearest_distance(points, roads, "dist_road_m"), on="cell", how="left")

    if "fclass" in roads.columns and roads["fclass"].isin(MAIN).any():
        out = out.merge(
            nearest_distance(points, roads[roads["fclass"].isin(MAIN)], "dist_main_road_m"),
            on="cell", how="left",
        )
    else:
        print("Warning: no main roads found; dist_main_road_m left empty.")
        out["dist_main_road_m"] = float("nan")

    clipped = gpd.overlay(roads[["geometry"]], cells, how="intersection", keep_geom_type=True)
    km = clipped.length.groupby(clipped["cell"]).sum() / 1000
    area_km2 = pd.Series((cells.area / 1e6).values, index=cells["cell"].values)
    density = (km / area_km2).rename("road_km_per_km2")
    out = out.merge(density, left_on="cell", right_index=True, how="left")
    out["road_km_per_km2"] = out["road_km_per_km2"].fillna(0.0)

    out_path = PROCESSED / f"roads_res{res}.parquet"
    out.to_parquet(out_path)
    print(f"Saved {out_path} ({len(out)} rows)")
    print(out.describe().T[["count", "mean", "min", "max"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roads", default=str(RAW / "jamaica-osm" / "gis_osm_roads_free_1.shp"))
    ap.add_argument("--res", type=int, default=8)
    from pathlib import Path
    a = ap.parse_args()
    main(Path(a.roads), a.res)