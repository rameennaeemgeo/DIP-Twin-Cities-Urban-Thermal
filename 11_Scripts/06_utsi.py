import os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.stats import spearmanr

ROOT = "DIP_Twin_Cities_Project"
RAW = ROOT + "/02_Raw_Data/"
CLS = ROOT + "/06_Classification/Classified_Maps/"
TH = ROOT + "/05_Thermal_Analysis/Thermal_Stress/"
MAPS = ROOT + "/08_Final_Maps/"
CHARTS = ROOT + "/09_Charts/"
STATS = ROOT + "/07_Statistics/Tables/"
for d in (TH, MAPS, CHARTS, STATS):
    os.makedirs(d, exist_ok=True)

rng = np.random.default_rng(42)
TITLE = {"pre": "Pre-monsoon (May-June 2025)", "post": "Post-monsoon (Sep-Oct 2025)"}
NAMES = {1: "Built-up", 2: "Vegetation", 3: "Agriculture", 4: "Water", 5: "Bare land"}

with rasterio.open(CLS + "LULC_RF_30m.tif") as f:
    lulc = f.read(1)
    profile = f.profile
    extent = (f.bounds.left, f.bounds.right, f.bounds.bottom, f.bounds.top)

def load(season):
    with rasterio.open(RAW + "S2_" + season + "_30m.tif") as f:
        red = f.read(3)
        nir = f.read(4)
        ndvi = f.read(6).astype("float32")
        ndbi = f.read(7).astype("float32")
    with rasterio.open(RAW + "LST_" + season + "_30m.tif") as f:
        lst = f.read(1).astype("float32")
    valid = (red != 0) & (nir != 0) & np.isfinite(ndvi) & np.isfinite(ndbi) & (lst != 0) & np.isfinite(lst)
    return np.where(valid, lst, np.nan), np.where(valid, ndvi, np.nan), np.where(valid, ndbi, np.nan)

def mm(a, lo=1, hi=99):
    # min-max normalisation after clipping the extreme 1% tails so single outliers do not squash the scale
    p1, p2 = np.nanpercentile(a, [lo, hi])
    return np.clip((a - p1) / (p2 - p1), 0, 1)

def utsi(lstn, ndvin, ndbin, w):
    return w[0] * lstn + w[1] * (1 - ndvin) + w[2] * ndbin

def decorate(ax):
    left, right, bottom, top = extent
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("Easting (m), UTM 43N")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain")
    ax.tick_params(labelsize=7)
    ax.grid(color="grey", lw=0.3, alpha=0.5)
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.85), xycoords="axes fraction", ha="center",
                fontsize=14, fontweight="bold", arrowprops=dict(facecolor="black", width=4, headwidth=12))
    x0 = left + 0.05 * (right - left)
    y0 = bottom + 0.05 * (top - bottom)
    ax.plot([x0, x0 + 20000], [y0, y0], color="black", lw=4)
    ax.text(x0 + 10000, y0 + 0.015 * (top - bottom), "20 km", ha="center", fontsize=8)

BASE = (0.5, 0.3, 0.2)
CLASS_NAMES = {1: "Low thermal stress", 2: "Moderate thermal stress", 3: "High thermal stress"}
CLASS_COL = {1: "#1a9641", 2: "#fee08b", 3: "#d7191c"}

rows_area, rows_lulc, rows_crit = [], [], []
store = {}
for s in ("pre", "post"):
    print("Season:", s)
    lst, ndvi, ndbi = load(s)
    lstn, ndvin, ndbin = mm(lst), mm(ndvi), mm(ndbi)
    u = utsi(lstn, ndvin, ndbin, BASE)
    ok = np.isfinite(u)
    m, sd = np.nanmean(u), np.nanstd(u)
    cls = np.zeros(u.shape, dtype="uint8")
    cls[ok & (u < m - 0.5 * sd)] = 1
    cls[ok & (u >= m - 0.5 * sd) & (u <= m + 0.5 * sd)] = 2
    cls[ok & (u > m + 0.5 * sd)] = 3
    print("  UTSI mean %.3f sd %.3f" % (m, sd))

    prof = profile.copy()
    prof.update(dtype="float32", nodata=-9999, count=1, compress="lzw")
    with rasterio.open(TH + "UTSI_" + s + ".tif", "w", **prof) as dst:
        dst.write(np.where(ok, u, -9999).astype("float32"), 1)

    total = ok.sum()
    for c in (1, 2, 3):
        n = (cls == c).sum()
        rows_area.append([s, CLASS_NAMES[c], round(n * 900 / 1e6, 1), round(100.0 * n / total, 1)])

    high = cls == 3
    for c in NAMES:
        inside = ok & (lulc == c)
        nh = (high & (lulc == c)).sum()
        rows_lulc.append([s, NAMES[c], round(100.0 * nh / max(inside.sum(), 1), 1), round(100.0 * nh / max(high.sum(), 1), 1)])

    # critical hotspot = hottest 10% of LST AND least vegetated 25% (threshold segmentation)
    crit = ok & (lst >= np.nanpercentile(lst, 90)) & (ndvi <= np.nanpercentile(ndvi, 25))
    rows_crit.append([s, round(crit.sum() * 900 / 1e6, 1), round(100.0 * crit.sum() / total, 2)])
    for c in NAMES:
        rows_crit.append([s + " - in " + NAMES[c], round(((crit) & (lulc == c)).sum() * 900 / 1e6, 1), None])

    arr = np.where(cls == 0, np.nan, cls.astype("float32"))
    cmap = ListedColormap([CLASS_COL[c] for c in (1, 2, 3)])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
    decorate(ax)
    ax.set_title("Urban Thermal Stress Zones - " + TITLE[s], fontsize=13, fontweight="bold")
    ax.legend(handles=[Patch(facecolor=CLASS_COL[c], label=CLASS_NAMES[c]) for c in (1, 2, 3)],
              title="UTSI class", loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(MAPS + "Map6_Thermal_Stress_" + s + ".png", dpi=200)
    plt.close(fig)
    store[s] = (lstn, ndvin, ndbin, u)

pd.DataFrame(rows_area, columns=["Season", "UTSI_class", "Area_km2", "Percent"]).to_csv(STATS + "Table_UTSI_classes.csv", index=False)
pd.DataFrame(rows_lulc, columns=["Season", "Land_cover", "Pct_of_class_that_is_high_stress", "Pct_of_high_stress_in_class"]).to_csv(STATS + "Table_UTSI_by_landcover.csv", index=False)
pd.DataFrame(rows_crit, columns=["Season", "Critical_hotspot_km2", "Percent_of_area"]).to_csv(STATS + "Table_critical_hotspots.csv", index=False)

print("Sensitivity analysis...")
wsets = {"Baseline 0.5/0.3/0.2": BASE, "LST-heavy 0.6/0.2/0.2": (0.6, 0.2, 0.2),
         "Equal 0.33/0.33/0.34": (0.33, 0.33, 0.34), "Vegetation-heavy 0.4/0.4/0.2": (0.4, 0.4, 0.2),
         "Built-up-heavy 0.4/0.2/0.4": (0.4, 0.2, 0.4)}
rows_s = []
for s in ("pre", "post"):
    lstn, ndvin, ndbin, ub = store[s]
    base_top = ub >= np.nanpercentile(ub, 80)
    ok = np.isfinite(ub)
    idx = np.flatnonzero(ok.ravel())
    pick = rng.choice(idx, size=50000, replace=False)
    for name, w in wsets.items():
        u2 = utsi(lstn, ndvin, ndbin, w)
        top2 = u2 >= np.nanpercentile(u2, 80)
        kept = 100.0 * (base_top & top2).sum() / base_top.sum()
        rho = spearmanr(ub.ravel()[pick], u2.ravel()[pick])[0]
        rows_s.append([s, name, round(kept, 1), round(float(rho), 3)])
sens = pd.DataFrame(rows_s, columns=["Season", "Weights", "Top20pct_hotspot_overlap_with_baseline_pct", "Spearman_vs_baseline"])
sens.to_csv(STATS + "Table_UTSI_sensitivity.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
names = list(wsets.keys())[1:]
x = np.arange(len(names))
for i, (s, col) in enumerate((("pre", "#e6ab5a"), ("post", "#7fb8d6"))):
    v = [sens[(sens.Season == s) & (sens.Weights == n)].iloc[0, 2] for n in names]
    ax.bar(x + (i - 0.5) * 0.35, v, 0.35, label=TITLE[s], color=col, edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("Hotspot overlap with baseline (%)")
ax.set_ylim(0, 100)
ax.set_title("UTSI weight sensitivity (top 20% hotspot pixels)", fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(CHARTS + "Chart8_UTSI_sensitivity.png", dpi=200)
plt.close(fig)

print("DONE")
print(pd.DataFrame(rows_area, columns=["Season", "UTSI_class", "Area_km2", "Percent"]).to_string(index=False))
print(pd.DataFrame(rows_lulc, columns=["Season", "Land_cover", "PctClassHighStress", "PctOfHighStressInClass"]).to_string(index=False))
print(sens.to_string(index=False))
