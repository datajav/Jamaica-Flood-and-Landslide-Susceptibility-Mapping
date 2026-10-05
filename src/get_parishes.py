import argparse
import json
import os
import sys

import ee

from paths import PARISHES


def init(project):
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def main(project):
    if not project:
        sys.exit("No project ID. Pass --project or set EARTHENGINE_PROJECT.")
    init(project)

    parishes = (
        ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM1")
        .filter(ee.Filter.eq("shapeGroup", "JAM"))
    )
    n = parishes.size().getInfo()
    print(f"Found {n} parishes")
    if n != 14:
        print("Warning: expected 14. Check how this dataset splits parishes.")

    geojson = parishes.getInfo()
    PARISHES.write_text(json.dumps(geojson))
    names = sorted(f["properties"]["shapeName"] for f in geojson["features"])
    print(f"Saved {PARISHES}")
    print(names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EARTHENGINE_PROJECT"))
    main(ap.parse_args().project)