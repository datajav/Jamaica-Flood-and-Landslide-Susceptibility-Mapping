import argparse
import os
import sys

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

from ee_utils import run_batches
from paths import PROCESSED
from terrain_features import init

SCALE = 30  # meters; WorldCover is 10 m, 30 m sampling keeps queries fast

CLASSES = {
    10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
    60: "bare", 80: "water", 90: "wetland", 95: "mangrove",
}


def build_image():
    lc = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    return ee.Image.cat([lc.eq(v).rename(name) for v, name in CLASSES.items()])


def make_reduce_fn(image):
    reducer = ee.Reducer.mean()

    def reduce_fn(batch):
        feats = [
            ee.Feature(ee.Geometry(mapping(g), None, False), {"cell": c})
            for c, g in zip(batch["cell"], batch["geometry"])
        ]
        out = image.reduceRegions(
            collection=ee.FeatureCollection(feats),
            reducer=reducer, scale=SCALE, tileScale=4,
        )
        return pd.DataFrame([f["properties"] for f in out.getInfo()["features"]])

    return reduce_fn


def tidy(df):
    missing = [c for c in CLASSES.values() if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns from Earth Engine output: {missing}\nGot: {list(df.columns)}")
    out = df[["cell"] + list(CLASSES.values())].copy()
    return out.rename(columns={c: f"lc_{c}" for c in CLASSES.values()})


def main(project, res, batch_size):
    grid_path = PROCESSED / f"grid_res{res}.parquet"
    if not grid_path.exists():
        sys.exit(f"Grid not found: {grid_path}")
    init(project)
    grid = gpd.read_parquet(grid_path)[["cell", "geometry"]]
    out = run_batches(grid, make_reduce_fn(build_image()), tidy, "landcover", res, batch_size)
    print(out.describe().T[["count", "mean", "min", "max"]].to_string())
    print(f"Cells with no data: {out['lc_tree'].isna().sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EARTHENGINE_PROJECT"))
    ap.add_argument("--res", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=500)
    a = ap.parse_args()
    main(a.project, a.res, a.batch_size)