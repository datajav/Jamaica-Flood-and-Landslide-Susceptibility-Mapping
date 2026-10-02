import argparse
import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Polygon


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
    parishes = gpd.read_file(boundary).to_crs(4326)
    frames = []
    for _, row in parishes.iterrows():
        cells = cells_for_geometry(row.geometry, res)
        frames.append(pd.DataFrame({"cell": list(cells), "parish": row[parish_col]}))
    grid = pd.concat(frames).drop_duplicates("cell").reset_index(drop=True)

    grid["lat"], grid["lon"] = zip(*grid["cell"].map(h3.cell_to_latlng))
    grid["geometry"] = grid["cell"].map(cell_polygon)
    gdf = gpd.GeoDataFrame(grid, geometry="geometry", crs=4326)
    gdf.to_parquet(out)
    print(f"{len(gdf)} cells at res {res} across {gdf.parish.nunique()} parishes -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--res", type=int, default=8)
    ap.add_argument("--parish-col", default="parish")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.boundary, a.res, a.parish_col, a.out or f"data/processed/grid_res{a.res}.parquet")