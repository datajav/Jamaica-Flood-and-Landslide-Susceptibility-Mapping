import numpy as np
import geopandas as gpd
import pandas as pd

from paths import PROCESSED

RES = 8

grid = gpd.read_parquet(PROCESSED / f"grid_res{RES}.parquet")[["cell", "parish", "lat", "lon"]]
terrain = pd.read_parquet(PROCESSED / f"terrain_res{RES}.parquet")
hydro = pd.read_parquet(PROCESSED / f"hydro_res{RES}.parquet")
landcover = pd.read_parquet(PROCESSED / f"landcover_res{RES}.parquet")
roads = pd.read_parquet(PROCESSED / f"roads_res{RES}.parquet")

df = (
    grid.merge(terrain, on="cell", validate="1:1")
    .merge(hydro, on="cell", validate="1:1")
    .merge(landcover, on="cell", validate="1:1")
    .merge(roads, on="cell", validate="1:1")
)

df["elev_mean"] = df["elev_mean"].clip(lower=0)
df["log_upa_max"] = np.log10(df["upa_max"])
df["is_mostly_water"] = df["water_frac"] > 0.5
df["is_near_sea_level"] = df["elev_max"] < 3
df["lc_permanent_water"] = df["lc_water"] + df["lc_wetland"] + df["lc_mangrove"]

out = PROCESSED / f"static_features_res{RES}.parquet"
df.to_parquet(out)

print(f"Saved {out}: {df.shape[0]} rows, {df.shape[1]} columns")
print("Mostly water:", int(df["is_mostly_water"].sum()))
print("Near sea level (max elev < 3 m):", int(df["is_near_sea_level"].sum()))
print(df[df["is_mostly_water"] | df["is_near_sea_level"]].groupby("parish").size().sort_values(ascending=False).to_string())
print(df[["elev_mean", "slope_mean", "hand_mean", "twi_mean", "log_upa_max"]].corr().round(2).to_string())