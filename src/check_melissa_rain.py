import os
import ee

from terrain_features import init

init(os.environ.get("EARTHENGINE_PROJECT"))

jamaica = ee.Geometry.Rectangle([-78.4, 17.6, -76.1, 18.6])
col = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
       .filterDate("2025-10-24", "2025-11-04")
       .select("precipitation"))
print("images:", col.size().getInfo(), "(expect 11)")


def daily(img):
    s = img.reduceRegion(ee.Reducer.mean(), jamaica, 5566)
    return ee.Feature(None, {
        "date": img.date().format("YYYY-MM-dd"),
        "mean_mm": s.get("precipitation"),
    })


for f in col.map(daily).getInfo()["features"]:
    print(f["properties"])