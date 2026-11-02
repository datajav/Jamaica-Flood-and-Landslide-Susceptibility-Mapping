import argparse
import os

import ee
import pandas as pd
import yaml

from paths import CONFIG
from terrain_features import init

COLUMNS = ["time_ms", "pass", "orbit", "platform"]


def summarize(rows):
    """rows: [[time_ms, pass, relative_orbit, platform], ...] -> one row per (pass, orbit)."""
    if not rows:
        return pd.DataFrame(columns=["pass", "orbit", "scenes", "first", "last", "dates"])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["time_ms"], unit="ms").dt.date
    out = (
        df.groupby(["pass", "orbit"])
        .agg(scenes=("date", "size"), first=("date", "min"), last=("date", "max"),
             dates=("date", lambda s: ", ".join(sorted({d.isoformat() for d in s}))))
        .reset_index()
    )
    return out


def find_pairs(pre, post):
    if pre.empty or post.empty:
        return pd.DataFrame()
    return pre.merge(post, on=["pass", "orbit"], suffixes=("_pre", "_post"))


def collection(jamaica, start, end):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(jamaica)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    )


def coverage(col, jamaica, pass_dir, orbit):
    sub = (col.filter(ee.Filter.eq("orbitProperties_pass", pass_dir))
              .filter(ee.Filter.eq("relativeOrbitNumber_start", int(orbit))))
    frac = (sub.select("VV").mosaic().mask()
               .reduceRegion(ee.Reducer.mean(), jamaica, 500).get("VV").getInfo())
    return round(frac or 0.0, 2)


def main(project):
    init(project)
    jamaica = ee.Geometry.Rectangle([-78.4, 17.6, -76.1, 18.6])
    events = yaml.safe_load((CONFIG / "events.yaml").read_text())["events"]

    for ev in events:
        print("=" * 70)
        print(f"{ev['name']} ({ev['id']})  event window {ev['event_window']}")
        tables = {}
        for w in ("pre_window", "post_window"):
            start, end = ev[w]
            col = collection(jamaica, start, end)
            rows = (col.reduceColumns(ee.Reducer.toList(4), [
                "system:time_start", "orbitProperties_pass",
                "relativeOrbitNumber_start", "platform_number",
            ]).get("list").getInfo())
            df = summarize(rows)
            if not df.empty:
                df["island_coverage"] = [
                    coverage(col, jamaica, p, o) for p, o in zip(df["pass"], df["orbit"])
                ]
            tables[w] = df
            print(f"\n{w} {start} to {end}: {len(rows)} scenes")
            print(df.drop(columns=["dates"]).to_string(index=False) if not df.empty else "  none")

        pairs = find_pairs(tables["pre_window"], tables["post_window"])
        print("\nUsable pairs (same pass and orbit before and after):")
        if pairs.empty:
            print("  NONE - widen the windows or use a different event")
        else:
            cols = ["pass", "orbit", "scenes_pre", "island_coverage_pre",
                    "scenes_post", "island_coverage_post", "first_post"]
            print(pairs[cols].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EARTHENGINE_PROJECT"))
    main(ap.parse_args().project)