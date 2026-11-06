import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import rasterio
import ee
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.stats import linregress

ee.Initialize(project="ai-geo-navigators-task-2")
ROOT = "DIP_Twin_Cities_Project"
RAW = ROOT + "/02_Raw_Data/"
CHARTS = ROOT + "/09_Charts/"
STATS = ROOT + "/07_Statistics/Tables/"
rng = np.random.default_rng(42)
B = 33  # 33 x 30 m = 990 m, close to the MODIS 1 km pixel
periods = {"pre": ("2025-05-01", "2025-07-01", "Pre-monsoon"), "post": ("2025-09-01", "2025-11-01", "Post-monsoon")}
tf = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)

def modis(start, end):
    def prep(img):
        good = img.select("QC_Day").bitwiseAnd(3).eq(0)
        return img.select("LST_Day_1km").multiply(0.02).subtract(273.15).updateMask(good).rename("MODIS")
    col = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(start, end)
    proj = col.first().select("LST_Day_1km").projection()
    return col.map(prep).mean().setDefaultProjection(proj), proj

rows, fig_data = [], {}
for key, (start, end, label) in periods.items():
    print("Season:", key)
    with rasterio.open(RAW + "LST_" + key + "_30m.tif") as f:
        a = f.read(1).astype("float32")
        t = f.transform
    a = np.where((a != 0) & np.isfinite(a), a, np.nan)
    H, W = (a.shape[0] // B) * B, (a.shape[1] // B) * B
    blocks = a[:H, :W].reshape(H // B, B, W // B, B)
    frac = np.mean(np.isfinite(blocks), axis=(1, 3))
    mean = np.nanmean(blocks, axis=(1, 3))
    ok = frac >= 0.8
    ii, jj = np.nonzero(ok)
    pick = rng.choice(len(ii), size=min(2500, len(ii)), replace=False)
    ii, jj = ii[pick], jj[pick]
    x = t.c + (jj * B + B / 2) * t.a
    y = t.f + (ii * B + B / 2) * t.e
    lon, lat = tf.transform(x, y)
    feats = [ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"k": int(n)}) for n, (lo, la) in enumerate(zip(lon, lat))]
    img, proj = modis(start, end)
    res = img.sampleRegions(collection=ee.FeatureCollection(feats), properties=["k"], projection=proj, geometries=False).getInfo()
    d = pd.DataFrame([r["properties"] for r in res["features"]]).dropna()
    d["Landsat"] = mean[ii[d["k"].values], jj[d["k"].values]]
    d = d[np.isfinite(d.Landsat) & np.isfinite(d.MODIS)]
    reg = linregress(d.MODIS, d.Landsat)
    diff = d.Landsat - d.MODIS
    rows.append([label, len(d), round(d.Landsat.mean(), 2), round(d.MODIS.mean(), 2), round(diff.mean(), 2),
                 round(float(np.sqrt((diff ** 2).mean())), 2), round(float(diff.abs().mean()), 2),
                 round(reg.rvalue, 3), round(reg.rvalue ** 2, 3), round(reg.slope, 3), round(reg.intercept, 2)])
    print("  samples: %d | r = %.3f | bias (Landsat - MODIS) = %.2f C | RMSE = %.2f C" % (len(d), reg.rvalue, diff.mean(), np.sqrt((diff ** 2).mean())))
    fig_data[label] = (d, reg)

t = pd.DataFrame(rows, columns=["Period", "N_blocks", "Landsat_mean_C", "MODIS_mean_C", "Bias_Landsat_minus_MODIS_C",
                                "RMSE_C", "MAE_C", "Pearson_r", "R2", "Slope", "Intercept"])
t.to_csv(STATS + "Table_MODIS_validation.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, (label, (d, reg)) in zip(axes, fig_data.items()):
    ax.scatter(d.MODIS, d.Landsat, s=6, alpha=0.35, color="grey")
    lo, hi = min(d.MODIS.min(), d.Landsat.min()), max(d.MODIS.max(), d.Landsat.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1 line")
    xs = np.array([d.MODIS.min(), d.MODIS.max()])
    ax.plot(xs, reg.slope * xs + reg.intercept, color="#d7191c", lw=2, label="Regression")
    ax.text(0.04, 0.95, "r = %.2f\nR$^2$ = %.2f" % (reg.rvalue, reg.rvalue ** 2), transform=ax.transAxes, va="top", fontweight="bold")
    ax.set_xlabel("MODIS LST, 1 km (deg C)")
    ax.set_ylabel("Landsat LST aggregated to ~1 km (deg C)")
    ax.set_title(label, fontweight="bold")
    ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(CHARTS + "Chart11_Landsat_vs_MODIS.png", dpi=200)
plt.close(fig)
print("DONE")
print(t.to_string(index=False))

