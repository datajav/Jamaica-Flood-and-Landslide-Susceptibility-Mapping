import sys
import time

import ee
import pandas as pd

from paths import PROCESSED


def run_batches(grid, reduce_fn, tidy_fn, name, res, batch_size=500, max_retries=4):

    cache = PROCESSED / f"{name}_batches"
    cache.mkdir(exist_ok=True)
    n_batches = (len(grid) + batch_size - 1) // batch_size
    print(f"{len(grid)} cells in {n_batches} batches of up to {batch_size}")

    for i in range(n_batches):
        part = cache / f"part_{i:04d}.parquet"
        if part.exists():
            continue
        batch = grid.iloc[i * batch_size:(i + 1) * batch_size]
        for attempt in range(max_retries):
            try:
                df = tidy_fn(reduce_fn(batch))
                break
            except ee.EEException as e:
                wait = 5 * 2 ** attempt
                print(f"  batch {i} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            sys.exit(f"Batch {i} failed {max_retries} times. Rerun to resume.")
        df.to_parquet(part)
        print(f"  batch {i + 1}/{n_batches} done")

    out = pd.concat(
        [pd.read_parquet(p) for p in sorted(cache.glob("part_*.parquet"))],
        ignore_index=True,
    )
    out_path = PROCESSED / f"{name}_res{res}.parquet"
    out.to_parquet(out_path)
    print(f"Saved {out_path} ({len(out)} rows)")
    return out