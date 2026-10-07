import os
import ee

ee.Initialize(project=os.environ.get("EARTHENGINE_PROJECT"))

img = ee.Image("MERIT/Hydro/v1_0_1")
print(img.bandNames().getInfo())

kingston = ee.Geometry.Point([-76.79, 17.97]).buffer(500)
print(img.reduceRegion(ee.Reducer.first(), kingston, 90).getInfo())