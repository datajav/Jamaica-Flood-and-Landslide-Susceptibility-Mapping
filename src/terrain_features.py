import argparse
import os
import sys
import time

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

from paths import PROCESSED

SCALE = 30  # meters, native resolution of GLO-30


def init(project):
    if not project:
        sys.exit("No project ID. Pass --project or set EARTHENGINE_PROJECT.")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def build_terrain_image():
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30")
    proj = col.first().select("DEM").projection()
    img = col.mosaic().setDefaultProjection(proj)

    dem = img.select("DEM").rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    # WBM > 0 marks ocean, lake and river pixels in GLO-30
    water = img.select("WBM").gt(0).rename("water")
    return dem.addBands(slope).addBands(water)


def build_reducer():
    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
    )


def reduce_batch(image, reducer, batch):
    feats = [
        ee.Feature(ee.Geometry(mapping(geom), None, False), {"cell": cell})
        for cell, geom in zip(batch["cell"], batch["geometry"])
    ]
    fc = ee.FeatureCollection(feats)
    out = image.reduceRegions(
        collection=fc, reducer=reducer, scale=SCALE, tileScale=4
    )
    rows = [f["properties"] for f in out.getInfo()["features"]]
    return pd.DataFrame(rows)


def tidy(df):
    keep = {
        "cell": "cell",
        "elevation_mean": "elev_mean",
        "elevation_stdDev": "elev_std",
        "elevation_min": "elev_min",
        "elevation_max": "elev_max",
        "slope_mean": "slope_mean",
        "slope_stdDev": "slope_std",
        "slope_max": "slope_max",
        "water_mean": "water_frac",
    }
    missing = [c for c in keep if c not in df.columns]
    if missing:
        sys.exit(f"Expected columns missing from Earth Engine output: {missing}\nGot: {list(df.columns)}")
    return df[list(keep)].rename(columns=keep)


def main(project, res, batch_size):
    grid_path = PROCESSED / f"grid_res{res}.parquet"
    if not grid_path.exists():
        sys.exit(f"Grid not found: {grid_path}\nRun: python src/build_grid.py --res {res}")

    init(project)
    grid = gpd.read_parquet(grid_path)[["cell", "geometry"]]
    image, reducer = build_terrain_image(), build_reducer()

    cache = PROCESSED / "terrain_batches"
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
            sys.exit(f"Batch {i} failed 4 times. Rerun to resume from here.")
        df.to_parquet(part)
        print(f"  batch {i + 1}/{n_batches} done")

    out = pd.concat(
        [pd.read_parquet(p) for p in sorted(cache.glob("part_*.parquet"))],
        ignore_index=True,
    )
    out_path = PROCESSED / f"terrain_res{res}.parquet"
    out.to_parquet(out_path)
    print(f"Saved {out_path} ({len(out)} rows)")
    print(out.describe().T[["count", "mean", "min", "max"]].to_string())
    print(f"Cells with no data: {out['elev_mean'].isna().sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EARTHENGINE_PROJECT"))
    ap.add_argument("--res", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=500)
    a = ap.parse_args()
    main(a.project, a.res, a.batch_size)

