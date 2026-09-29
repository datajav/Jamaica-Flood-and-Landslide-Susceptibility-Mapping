# Jamaica Flood and Landslide Susceptibility Mapping

Island-wide flood and landslide susceptibility modeling for Jamaica, using satellite imagery, terrain and rainfall data on a shared H3 hexagon grid.

## Approach

- **Labels:** Sentinel-1 SAR flood extent; Sentinel-2 pre/post change for landslide scars
- **Features:** DEM derivatives (slope, curvature, HAND, TWI), CHIRPS rainfall, NDVI, soil, distance to roads and rivers
- **Unit of analysis:** one row per H3 cell per event
- **Validation:** leave-one-event-out and leave-one-parish-out CV, scored with PR-AUC
- **Explainability:** SHAP, run separately for floods and landslides
- **Output:** hazard, exposure and vulnerability map by parish (Streamlit)

## Phases

1. Island grid and static features (current)
2. Event config and rainfall features
3. Flood labels and baseline model
4. Landslide labels and baseline model
5. Leave-one-event-out evaluation, SHAP, combined map, dashboard

## Setup

```bash
pip install geopandas h3 shapely pyyaml pandas pyarrow
python src/build_grid.py --boundary data/raw/parishes.geojson --res 8
```

## Structure

```
config/events.yaml    event definitions (dates, windows, label types)
src/build_grid.py     island H3 grid tagged by parish
data/raw/             input boundaries and downloads
data/processed/       generated grids and feature tables
```