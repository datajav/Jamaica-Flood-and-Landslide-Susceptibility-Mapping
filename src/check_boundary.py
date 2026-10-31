"""Compare the parish boundary area with the H3 grid area, and measure edge overhang.

Usage (from any folder):
    python src/check_boundary.py
    python src/check_boundary.py --boundary data/raw/jamaica_parishes.geojson
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd

from paths import PROCESSED, RAW

CRS_M = 3448  # JAD2001 / Jamaica Metric Grid
NAME_COLUMNS = ["shapeName", "ADM1_NAME", "NAME_1", "parish", "Parish", "name"]


def main(boundary, res):
    grid_path = PROCESSED / f"grid_res{res}.parquet"
    for p in (boundary, grid_path):
        if not p.exists():
            sys.exit(f"Not found: {p}")

    b = gpd.read_file(boundary).to_crs(CRS_M)
    g = gpd.read_parquet(grid_path).to_crs(CRS_M)
    col = next((c for c in NAME_COLUMNS if c in b.columns), None)
    if col is None:
        sys.exit(f"No parish name column. Columns: {list(b.columns)}")

    b["boundary_km2"] = b.area / 1e6
    g["cell_km2"] = g.area / 1e6

    per = b.set_index(col)[["boundary_km2"]].join(
        g.groupby("parish").agg(cells=("cell", "size"), grid_km2=("cell_km2", "sum"))
    )
    per["grid/boundary"] = per["grid_km2"] / per["boundary_km2"]
    print(per.round(2).sort_values("boundary_km2", ascending=False).to_string())

    union = b.geometry.union_all() if hasattr(b.geometry, "union_all") else b.geometry.unary_union
    g["inside_frac"] = g.geometry.intersection(union).area / g.geometry.area
    outside_km2 = ((1 - g["inside_frac"]) * g["cell_km2"]).sum()

    print()
    print(f"Boundary area:            {b['boundary_km2'].sum():8.0f} km2   (Jamaica is about 10,990 km2)")
    print(f"Grid cells:               {len(g):8d}")
    print(f"Grid area (sum of cells): {g['cell_km2'].sum():8.0f} km2")
    print(f"  of which outside boundary: {outside_km2:6.0f} km2")
    print()
    print("Share of each cell inside the boundary:")
    print(g["inside_frac"].describe().round(2).to_string())
    for t in (0.25, 0.5, 0.75):
        print(f"cells with < {t:.0%} inside: {int((g['inside_frac'] < t).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", default=str(RAW / "jamaica_parishes.geojson"))
    ap.add_argument("--res", type=int, default=8)
    a = ap.parse_args()
    main(Path(a.boundary), a.res)