import os
import ee
import pandas as pd

from paths import PROCESSED
from terrain_features import init

init(os.environ.get("EARTHENGINE_PROJECT"))

jamaica = ee.Geometry.Rectangle([-78.4, 17.6, -76.1, 18.6])
reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)


def daily(img):
    s = img.reduceRegion(reducer, jamaica, 5566)
    return ee.Feature(None, {
        "date": img.date().format("YYYY-MM-dd"),
        "mean": s.get("precipitation_mean"),
        "max": s.get("precipitation_max"),
    })


rows = []
for year in range(2015, 2027):
    col = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .select("precipitation"))
    rows += [f["properties"] for f in col.map(daily).getInfo()["features"]]
    print(year, "done")

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()
df["mean_3d"] = df["mean"].rolling(3).sum()
df["max_3d"] = df["max"].rolling(3).sum()

picked = []
for date, row in df.sort_values("mean_3d", ascending=False).iterrows():
    if all(abs((date - p).days) > 14 for p, _ in picked):
        picked.append((date, row))
    if len(picked) == 15:
        break

out = pd.DataFrame([r for _, r in picked]).round(1)
out = out[["mean_3d", "max_3d"]].sort_values("mean_3d", ascending=False)
out.index = out.index.date
print(out.to_string())
out.to_csv(PROCESSED / "top_rain_episodes.csv")