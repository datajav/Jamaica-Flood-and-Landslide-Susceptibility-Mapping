import os
import ee

from terrain_features import init

init(os.environ.get("EARTHENGINE_PROJECT"))

jamaica = ee.Geometry.Rectangle([-78.4, 17.6, -76.1, 18.6])
START, END = "2025-10-25", "2025-10-31"  # END is exclusive: covers 25-30 Oct
reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)

chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
          .filterDate(START, END).select("precipitation").sum())

imerg_col = (ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
             .filterDate(START, END).select("precipitation"))
print("IMERG images:", imerg_col.size().getInfo(), "(expect 288)")
imerg = imerg_col.sum().multiply(0.5)  # mm/hr x half an hour per image

for name, img, scale in [("CHIRPS", chirps, 5566), ("IMERG", imerg, 11132)]:
    print(name, img.reduceRegion(reducer, jamaica, scale).getInfo())