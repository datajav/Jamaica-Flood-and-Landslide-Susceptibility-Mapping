from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "config"

for d in (RAW, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)

PARISHES = RAW / "parishes.geojson"