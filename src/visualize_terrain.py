import argparse
import sys

import branca.colormap as bcm
import folium
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shapely
from branca.element import MacroElement, Template

from paths import PARISHES, PROCESSED, ROOT

FIGURES = ROOT / "figures"

# column, label, matplotlib cmap, folium colors
LAYERS = [
    ("elev_mean", "Mean elevation (m)", "terrain", ["#2c7bb6", "#abd9a0", "#ffffbf", "#d7a45a", "#8c510a", "#ffffff"]),
    ("slope_mean", "Mean slope (deg)", "YlOrRd", ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c", "#800026"]),
    ("elev_std", "Elevation variability, std (m)", "viridis", ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]),
    ("slope_max", "Max slope (deg)", "magma", ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"]),
    ("water_frac", "Water fraction", "Blues", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]),
]


def load(res):
    g_path, t_path = PROCESSED / f"grid_res{res}.parquet", PROCESSED / f"terrain_res{res}.parquet"
    for p in (g_path, t_path):
        if not p.exists():
            sys.exit(f"Missing {p}. Run build_grid.py and terrain_features.py first.")
    d = gpd.read_parquet(g_path)[["cell", "parish", "geometry"]].merge(
        pd.read_parquet(t_path), on="cell"
    )
    d["elev_mean"] = d["elev_mean"].clip(lower=0)  # DEM dips slightly below 0 at the coast
    d["suspect"] = ((d["water_frac"] > 0.5) | (d["elev_max"] < 3)).astype(int)
    # Snap coordinates to ~10 m to keep the HTML file small
    d["geometry"] = shapely.set_precision(d.geometry.values, 1e-4)
    return d


def static_figure(d, res):
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    panels = [(c, lab, cm) for c, lab, cm, _ in LAYERS]
    panels.append(("suspect", "Suspect cells (>50% water or max elev < 3 m)", "Reds"))
    for ax, (col, label, cmap) in zip(axes.ravel(), panels):
        d.plot(column=col, cmap=cmap, linewidth=0, ax=ax, legend=True,
               legend_kwds={"shrink": 0.6})
        ax.set_title(label, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(f"Jamaica terrain features, H3 resolution {res} ({len(d):,} cells)", fontsize=14)
    fig.tight_layout()
    out = FIGURES / f"terrain_overview_res{res}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


class LayerSwitch(MacroElement):
    """Dropdown that restyles one GeoJSON layer by a chosen column (keeps the file small)."""

    _template = Template("""
    {% macro script(this, kwargs) %}
    (function () {
      var cfg = {{ this.config|tojson }};
      var layer = {{ this.layer_name }};
      var map = {{ this._parent.get_name() }};
      var ctl = L.control({position: "topright"});
      ctl.onAdd = function () {
        var div = L.DomUtil.create("div", "leaflet-bar");
        div.style.cssText = "background:white;padding:8px;font:12px sans-serif;";
        var opts = Object.keys(cfg).map(function (k) {
          return '<option value="' + k + '">' + cfg[k].label + "</option>";
        }).join("");
        div.innerHTML = "<b>Layer</b><br><select id='layersel'>" + opts +
                        "</select><div id='lg' style='margin-top:6px'></div>";
        L.DomEvent.disableClickPropagation(div);
        return div;
      };
      ctl.addTo(map);
      function apply(col) {
        layer.setStyle(function (f) {
          return {fillColor: f.properties["c_" + col], color: "none", fillOpacity: 0.8};
        });
        var c = cfg[col];
        document.getElementById("lg").innerHTML =
          "<div style='height:10px;width:200px;background:linear-gradient(to right," + c.colors.join(",") + ")'></div>" +
          "<div style='display:flex;justify-content:space-between;width:200px'><span>" + c.lo + "</span><span>" + c.hi + "</span></div>";
      }
      document.getElementById("layersel").addEventListener("change", function (e) { apply(e.target.value); });
      apply(Object.keys(cfg)[0]);
    })();
    {% endmacro %}
    """)

    def __init__(self, layer_name, config):
        super().__init__()
        self._name = "LayerSwitch"
        self.layer_name = layer_name
        self.config = config


def interactive_map(d, res):
    FIGURES.mkdir(exist_ok=True)
    m = folium.Map(location=[18.15, -77.3], zoom_start=9, tiles="OpenStreetMap")

    sub = d[["cell", "parish", "suspect"] + [c for c, *_ in LAYERS] + ["geometry"]].copy()
    config = {}
    for col, label, _, colors in LAYERS:
        hi = 1.0 if col == "water_frac" else float(d[col].quantile(0.99))
        lo = float(d[col].min())
        cm = bcm.LinearColormap(colors, vmin=lo, vmax=max(hi, lo + 1e-6))
        sub["c_" + col] = [cm(min(v, cm.vmax)) for v in sub[col]]
        config[col] = {"label": label, "colors": colors, "lo": round(lo, 2), "hi": round(hi, 2)}
    sub["c_suspect"] = ["#e31a1c" if v else "#dddddd" for v in sub["suspect"]]
    config["suspect"] = {"label": "Suspect cells (>50% water or max elev < 3 m)",
                         "colors": ["#dddddd", "#e31a1c"], "lo": "ok", "hi": "suspect"}
    sub[[c for c, *_ in LAYERS]] = sub[[c for c, *_ in LAYERS]].round(2)

    tooltip_cols = ["parish"] + [c for c, *_ in LAYERS]
    gj = folium.GeoJson(
        sub.to_json(), name="Terrain cells",
        style_function=lambda f: {"fillColor": f["properties"]["c_elev_mean"], "color": "none", "fillOpacity": 0.8},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_cols, localize=True),
    ).add_to(m)
    m.add_child(LayerSwitch(gj.get_name(), config))

    if PARISHES.exists():
        parishes = gpd.read_file(PARISHES)
        parishes["geometry"] = shapely.set_precision(parishes.geometry.values, 1e-3)
        folium.GeoJson(
            parishes.to_json(), name="Parish boundaries",
            style_function=lambda f: {"fillOpacity": 0, "color": "#333", "weight": 1.2},
        ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    out = FIGURES / f"terrain_map_res{res}.html"
    m.save(str(out))
    print(f"Saved {out}  ({out.stat().st_size / 1e6:.1f} MB) - open it in your browser")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=8)
    a = ap.parse_args()
    data = load(a.res)
    print(f"Suspect cells: {int(data['suspect'].sum())} of {len(data)}")
    static_figure(data, a.res)
    interactive_map(data, a.res)