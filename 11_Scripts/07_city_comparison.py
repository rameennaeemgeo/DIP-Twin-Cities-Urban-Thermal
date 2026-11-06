import os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import rasterio
import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import shape
from rasterio import features
from scipy.stats import pearsonr

ROOT = "DIP_Twin_Cities_Project"
RAW = ROOT + "/02_Raw_Data/"
CLS = ROOT + "/06_Classification/Classified_Maps/"
TH = ROOT + "/05_Thermal_Analysis/Thermal_Stress/"
CHARTS = ROOT + "/09_Charts/"
STATS = ROOT + "/07_Statistics/Tables/"
rng = np.random.default_rng(42)
NAMES = {1: "Built-up", 2: "Vegetation", 3: "Agriculture", 4: "Water", 5: "Bare land"}

print("Downloading district boundaries...")
ee.Initialize(project="ai-geo-navigators-task-2")
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
def get_gdf(key, name):
    fc = gaul.filter(ee.Filter.stringContains("ADM2_NAME", key))
    geo = fc.geometry().simplify(50).getInfo()
    return gpd.GeoDataFrame({"name": [name]}, geometry=[shape(geo)], crs="EPSG:4326").to_crs("EPSG:32643")
isb = get_gdf("Islamabad", "Islamabad")
rwp = get_gdf("Rawalpindi", "Rawalpindi")
isb.to_file(ROOT + "/01_Study_Area/Islamabad_Boundary/Islamabad.geojson", driver="GeoJSON")
rwp.to_file(ROOT + "/01_Study_Area/Rawalpindi_Boundary/Rawalpindi.geojson", driver="GeoJSON")

with rasterio.open(CLS + "LULC_RF_30m.tif") as f:
    lulc = f.read(1)
    transform = f.transform
city = np.zeros(lulc.shape, dtype="uint8")
features.rasterize([(rwp.geometry.iloc[0], 2)], out=city, transform=transform)
features.rasterize([(isb.geometry.iloc[0], 1)], out=city, transform=transform)
print("Pixels - Islamabad: %d | Rawalpindi: %d" % ((city == 1).sum(), (city == 2).sum()))

groups = {"Islamabad": city == 1, "Rawalpindi": city == 2, "Combined": city > 0}

def load(s):
    with rasterio.open(RAW + "S2_" + s + "_30m.tif") as f:
        a = f.read([3, 4, 6, 7, 8]).astype("float32")
    with rasterio.open(RAW + "LST_" + s + "_30m.tif") as f:
        lst = f.read(1).astype("float32")
    with rasterio.open(TH + "UTSI_" + s + ".tif") as f:
        u = f.read(1).astype("float32")
    valid = (a[0] != 0) & (a[1] != 0) & np.isfinite(a[2]) & (lst != 0) & np.isfinite(lst)
    return lst, a[2], a[3], a[4], u, valid

metrics = {}
lc_rows = []
box_rows = []
for s in ("pre", "post"):
    print("Season:", s)
    lst, ndvi, ndbi, ndwi, u, valid = load(s)
    uok = u > -9000
    m, sd = u[uok].mean(), u[uok].std()
    high = uok & (u > m + 0.5 * sd)
    for g, gm in groups.items():
        mk = gm & valid
        col = s + "_" + g
        r = {}
        r["Area analysed (km2)"] = round(mk.sum() * 900 / 1e6, 1)
        r["LST mean (C)"] = round(float(lst[mk].mean()), 2)
        r["LST std (C)"] = round(float(lst[mk].std()), 2)
        r["LST min (C)"] = round(float(lst[mk].min()), 2)
        r["LST max (C)"] = round(float(lst[mk].max()), 2)
        r["NDVI mean"] = round(float(ndvi[mk].mean()), 3)
        r["NDBI mean"] = round(float(ndbi[mk].mean()), 3)
        r["NDWI mean"] = round(float(ndwi[mk].mean()), 3)
        denom = (mk & (lulc > 0)).sum()
        for c in NAMES:
            r[NAMES[c] + " (%)"] = round(100.0 * (mk & (lulc == c)).sum() / denom, 1)
        r["High thermal stress (% of area)"] = round(100.0 * (mk & high).sum() / mk.sum(), 1)
        idx = np.flatnonzero(mk.ravel())
        pick = rng.choice(idx, size=min(50000, idx.size), replace=False)
        r["Pearson LST-NDVI"] = round(float(pearsonr(ndvi.ravel()[pick], lst.ravel()[pick])[0]), 3)
        r["Pearson LST-NDBI"] = round(float(pearsonr(ndbi.ravel()[pick], lst.ravel()[pick])[0]), 3)
        metrics[col] = r
        if g != "Combined":
            for c in NAMES:
                cm = mk & (lulc == c)
                if cm.sum() > 0:
                    lc_rows.append([s, g, NAMES[c], round(float(lst[cm].mean()), 2), int(cm.sum())])
            sel = rng.choice(idx, size=min(30000, idx.size), replace=False)
            box_rows.append(pd.DataFrame({"Season": "Pre-monsoon" if s == "pre" else "Post-monsoon",
                                          "City": g, "LST": lst.ravel()[sel]}))

t8 = pd.DataFrame(metrics)
t8.to_csv(STATS + "Table8_Islamabad_vs_Rawalpindi.csv")
lc = pd.DataFrame(lc_rows, columns=["Season", "City", "Land_cover", "Mean_LST_C", "Pixels"])
lc.to_csv(STATS + "Table_LST_by_landcover_by_city.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5.5))
sns.boxplot(data=pd.concat(box_rows), x="Season", y="LST", hue="City",
            palette={"Islamabad": "#2b83ba", "Rawalpindi": "#d7191c"}, ax=ax)
ax.set_ylabel("LST (deg C)")
ax.set_title("LST distribution: Islamabad vs Rawalpindi", fontweight="bold")
fig.tight_layout()
fig.savefig(CHARTS + "Chart9_city_LST_boxplot.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
colors = ["#d7191c", "#1a9641", "#fdd835", "#2b83ba", "#bdbdbd"]
for ax, s in zip(axes, ("pre", "post")):
    d = pd.DataFrame({g: [metrics[s + "_" + g][NAMES[c] + " (%)"] for c in NAMES]
                      for g in ("Islamabad", "Rawalpindi")}, index=list(NAMES.values())).T
    d.plot(kind="bar", stacked=True, color=colors, ax=ax, edgecolor="black", legend=(s == "post"))
    ax.set_title("Land cover - " + ("Pre-monsoon" if s == "pre" else "Post-monsoon"))
    ax.set_ylabel("Share of area (%)")
    ax.tick_params(axis="x", rotation=0)
axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
fig.tight_layout()
fig.savefig(CHARTS + "Chart10_city_landcover.png", dpi=200)
plt.close(fig)

print("DONE")
print(t8.to_string())
print(lc.to_string(index=False))
