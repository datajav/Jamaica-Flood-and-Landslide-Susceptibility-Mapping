import argparse
import sys

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Polygon

from paths import PARISHES, PROCESSED

NAME_COLUMNS = ["shapeName", "ADM1_NAME", "NAME_1", "parish", "Parish", "name"]


def find_parish_column(gdf, requested):
    if requested:
        if requested not in gdf.columns:
            sys.exit(f"Column '{requested}' not found. Columns: {list(gdf.columns)}")
        return requested
    for col in NAME_COLUMNS:
        if col in gdf.columns:
            return col
    sys.exit(f"Could not find a parish name column. Columns: {list(gdf.columns)}. Use --parish-col.")


def cells_for_geometry(geom, res):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    cells = set()
    for p in polys:
        cells |= set(h3.geo_to_cells(p, res))
    return cells


def cell_polygon(cell):
    # h3 returns (lat, lng); shapely wants (lng, lat)
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def main(boundary, res, parish_col, out):
    if not boundary.exists():
        sys.exit(f"Boundary file not found: {boundary}\nRun: python src/get_parishes.py --project <id>")

    parishes = gpd.read_file(boundary).to_crs(4326)
    col = find_parish_column(parishes, parish_col)

    frames = []
    for _, row in parishes.iterrows():
        cells = cells_for_geometry(row.geometry, res)
        frames.append(pd.DataFrame({"cell": list(cells), "parish": row[col]}))
    grid = pd.concat(frames).drop_duplicates("cell").reset_index(drop=True)

    grid["lat"], grid["lon"] = zip(*grid["cell"].map(h3.cell_to_latlng))
    grid["geometry"] = grid["cell"].map(cell_polygon)
    gdf = gpd.GeoDataFrame(grid, geometry="geometry", crs=4326)
    gdf.to_parquet(out)

    print(f"{len(gdf)} cells at res {res} across {gdf.parish.nunique()} parishes")
    print(f"Saved {out}")
    print(gdf.groupby("parish").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", default=str(PARISHES))
    ap.add_argument("--res", type=int, default=8)
    ap.add_argument("--parish-col", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from pathlib import Path
    out = Path(a.out) if a.out else PROCESSED / f"grid_res{a.res}.parquet"
    main(Path(a.boundary), a.res, a.parish_col, out)