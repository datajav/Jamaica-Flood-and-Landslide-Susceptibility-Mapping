import argparse
import math
import os
import sys
import time

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

from paths import PROCESSED
from terrain_features import build_reducer, init

SCALE = 90  # MERIT Hydro native resolution, meters


def build_hydro_image():
    h = ee.Image("MERIT/Hydro/v1_0_1")
    # slope in degrees from MERIT elevation; floor avoids divide-by-zero on flat ground
    slope = ee.Terrain.slope(h.select("elv")).max(0.1)
    # specific catchment area (m): upstream area in km2 -> m2, divided by pixel width
    sca = h.select("upa").multiply(1e6).divide(SCALE)
    twi = sca.divide(slope.multiply(math.pi / 180).tan()).log().rename("twi")
    return (
        h.select("hnd").rename("hand")
        .addBands(h.select("upa"))
        .addBands(twi)
        .addBands(h.select("wat").rename("water"))
    )


def reduce_batch(image, reducer, batch):
    feats = [
        ee.Feature(ee.Geometry(mapping(geom), None, False), {"cell": cell})
        for cell, geom in zip(batch["cell"], batch["geometry"])
    ]
    out = image.reduceRegions(
        collection=ee.FeatureCollection(feats),
        reducer=reducer, scale=SCALE, tileScale=4,
    )
    return pd.DataFrame([f["properties"] for f in out.getInfo()["features"]])


def tidy(df):
    keep = {
        "cell": "cell",
        "hand_mean": "hand_mean",
        "hand_min": "hand_min",
        "hand_max": "hand_max",
        "upa_mean": "upa_mean",
        "upa_max": "upa_max",
        "twi_mean": "twi_mean",
        "twi_max": "twi_max",
        "water_mean": "hydro_water_frac",
    }
    missing = [c for c in keep if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns from Earth Engine output: {missing}\nGot: {list(df.columns)}")
    return df[list(keep)].rename(columns=keep)


def main(project, res, batch_size):
    grid_path = PROCESSED / f"grid_res{res}.parquet"
    if not grid_path.exists():
        sys.exit(f"Grid not found: {grid_path}")

    init(project)
    grid = gpd.read_parquet(grid_path)[["cell", "geometry"]]
    image, reducer = build_hydro_image(), build_reducer()

    cache = PROCESSED / "hydro_batches"
    cache.mkdir(exist_ok=True)
    n_batches = (len(grid) + batch_size - 1) // batch_size
    print(f"{len(grid)} cells in {n_batches} batches of up to {batch_size}")

    for i in range(n_batches):
        part = cache / f"part_{i:04d}.parquet"
        if part.exists():
            continue
        batch = grid.iloc[i * batch_size:(i + 1) * batch_size]
        for attempt in range(4):
            try:
                df = tidy(reduce_batch(image, reducer, batch))
                break
            except ee.EEException as e:
                wait = 5 * 2 ** attempt
                print(f"  batch {i} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            sys.exit(f"Batch {i} failed 4 times. Rerun to resume.")
        df.to_parquet(part)
        print(f"  batch {i + 1}/{n_batches} done")

    out = pd.concat(
        [pd.read_parquet(p) for p in sorted(cache.glob("part_*.parquet"))],
        ignore_index=True,
    )
    out_path = PROCESSED / f"hydro_res{res}.parquet"
    out.to_parquet(out_path)
    print(f"Saved {out_path} ({len(out)} rows)")
    print(out.describe().T[["count", "mean", "min", "max"]].to_string())
    print(f"Cells with no HAND value: {out['hand_mean'].isna().sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EARTHENGINE_PROJECT"))
    ap.add_argument("--res", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=500)
    a = ap.parse_args()
    main(a.project, a.res, a.batch_size)