import os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

ROOT = "DIP_Twin_Cities_Project"
RAW = ROOT + "/02_Raw_Data/"
OUT = ROOT + "/03_Preprocessed_Images/Enhanced_Images/"
MAPS = ROOT + "/08_Final_Maps/"
STATS = ROOT + "/07_Statistics/Tables/"
for d in (OUT, MAPS, STATS):
    os.makedirs(d, exist_ok=True)

# zoom window over the Islamabad-Rawalpindi urban core (UTM 43N metres): xmin, ymin, xmax, ymax
BOUNDS = (308000, 3712000, 332000, 3736000)

def win(path, bands):
    with rasterio.open(path) as f:
        w = from_bounds(*BOUNDS, transform=f.transform).round_offsets().round_lengths()
        return f.read(bands, window=w).astype("float32")

print("Loading window...")
s2 = win(RAW + "S2_pre_30m.tif", [1, 2, 3, 4, 5, 6])   # blue green red nir swir1 NDVI
valid = np.all(np.isfinite(s2), axis=0) & (s2[2] != 0) & (s2[3] != 0)
s2 = np.where(valid[None], s2, 0).astype('float32')

def stretch(b):
    lo, hi = np.percentile(b[valid], [2, 98])
    return np.clip((b - lo) / (hi - lo), 0, 1) * valid

def equalize(b):
    x = np.clip(b / np.percentile(b[valid], 99.5), 0, 1)
    v = (x * 255).astype(np.uint8)
    h = np.bincount(v[valid], minlength=256).astype(float)
    cdf = h.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    return cdf[v] * valid

def lin(b):
    return np.clip(b / 0.3, 0, 1) * valid

print("Enhancement 1: contrast stretch + histogram equalization...")
rgb_o = np.dstack([lin(s2[i]) for i in (2, 1, 0)])
rgb_s = np.dstack([stretch(s2[i]) for i in (2, 1, 0)])
rgb_e = np.dstack([equalize(s2[i]) for i in (2, 1, 0)])
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for j, (im, t) in enumerate([(rgb_o, "Original (linear 0-0.3)"), (rgb_s, "Contrast stretch (2-98%)"), (rgb_e, "Histogram equalization")]):
    axes[0, j].imshow(im)
    axes[0, j].set_title(t, fontweight="bold")
    axes[0, j].axis("off")
for j, arr in enumerate([lin(s2[2]), stretch(s2[2]), equalize(s2[2])]):
    axes[1, j].hist(arr[valid], bins=64, color="grey", edgecolor="black", linewidth=0.2)
    axes[1, j].set_title("Histogram of red band")
    axes[1, j].set_xlabel("Display value")
    axes[1, j].set_ylabel("Pixels")
fig.tight_layout()
fig.savefig(OUT + "Enh1_contrast_equalization.png", dpi=200)
plt.close(fig)

print("Enhancement 2: false-colour composites...")
fc1 = np.dstack([stretch(s2[i]) for i in (3, 2, 1)])
fc2 = np.dstack([stretch(s2[i]) for i in (4, 3, 2)])
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, (im, t) in zip(axes, [(rgb_s, "True colour (R-G-B)"), (fc1, "False colour NIR-R-G (vegetation = red)"), (fc2, "False colour SWIR-NIR-R (urban / vegetation / soil)")]):
    ax.imshow(im)
    ax.set_title(t, fontweight="bold", fontsize=10)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT + "Enh2_false_colour.png", dpi=200)
plt.close(fig)

print("Enhancement 3: median and Gaussian filters...")
lst = win(RAW + "LST_pre_30m.tif", [1])[0]
lv = (lst != 0) & np.isfinite(lst)
ndvi = s2[5]
nv = valid & np.isfinite(ndvi)
def rough(a):
    return float(np.nanstd(np.diff(a, axis=1)))
rows = []
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for r, (name, arr, m, cmap) in enumerate([("LST (deg C)", lst, lv, "RdYlBu_r"), ("NDVI", ndvi, nv, "RdYlGn")]):
    fill = np.where(m, arr, np.mean(arr[m]))
    med = np.where(m, ndi.median_filter(fill, size=3), np.nan)
    gau = np.where(m, ndi.gaussian_filter(fill, sigma=1), np.nan)
    org = np.where(m, arr, np.nan)
    lo, hi = np.percentile(arr[m], [1, 99])
    for c, (im, t) in enumerate([(org, "Original"), (med, "Median filter 3x3"), (gau, "Gaussian filter, sigma = 1")]):
        h = axes[r, c].imshow(im, cmap=cmap, vmin=lo, vmax=hi)
        axes[r, c].set_title(name + " - " + t, fontweight="bold", fontsize=10)
        axes[r, c].axis("off")
        rows.append([name, t, round(rough(im), 4)])
fig.tight_layout()
fig.savefig(OUT + "Enh3_filters.png", dpi=200)
plt.close(fig)
pd.DataFrame(rows, columns=["Layer", "Version", "Roughness_std_of_neighbour_difference"]).to_csv(STATS + "Table_filter_noise_reduction.csv", index=False)

print("Threshold segmentation + morphological opening...")
lstn = np.where(lv, lst, np.nan)
thr = np.nanpercentile(lstn, 90)
hot = lv & (lst >= thr)
opened = ndi.binary_opening(hot, structure=np.ones((3, 3)))
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
axes[0].imshow(lstn, cmap="RdYlBu_r")
axes[0].set_title("LST (pre-monsoon)", fontweight="bold")
axes[1].imshow(hot, cmap="gray_r")
axes[1].set_title("Threshold: LST >= 90th percentile (%.1f C)" % thr, fontweight="bold", fontsize=10)
axes[2].imshow(opened, cmap="gray_r")
axes[2].set_title("After morphological opening (3x3)", fontweight="bold", fontsize=10)
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT + "Enh4_threshold_morphology.png", dpi=200)
plt.close(fig)
print("  hotspot pixels before opening: %d | after: %d" % (hot.sum(), opened.sum()))

print("Image differencing (full extent)...")
def full(path, band):
    with rasterio.open(path) as f:
        return f.read(band).astype("float32"), (f.bounds.left, f.bounds.right, f.bounds.bottom, f.bounds.top)
lp, ext = full(RAW + "LST_pre_30m.tif", 1)
lq, _ = full(RAW + "LST_post_30m.tif", 1)
rp, _ = full(RAW + "S2_pre_30m.tif", 3)
rq, _ = full(RAW + "S2_post_30m.tif", 3)
np_, _ = full(RAW + "S2_pre_30m.tif", 6)
nq, _ = full(RAW + "S2_post_30m.tif", 6)
vl = (lp != 0) & (lq != 0) & np.isfinite(lp) & np.isfinite(lq)
vn = (rp != 0) & (rq != 0) & np.isfinite(np_) & np.isfinite(nq)
dlst = np.where(vl, lq - lp, np.nan)
dndvi = np.where(vn, nq - np_, np.nan)
for arr, name, cmap, lo, hi, lab, fn in [(dlst, "Change in LST (post minus pre)", "RdYlBu_r", -20, 0, "Delta LST (deg C)", "Map7_dLST.png"),
                                         (dndvi, "Change in NDVI (post minus pre)", "RdYlGn", -0.3, 0.6, "Delta NDVI", "Map8_dNDVI.png")]:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, extent=ext)
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Easting (m), UTM 43N")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain")
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label=lab)
    fig.tight_layout()
    fig.savefig(MAPS + fn, dpi=200)
    plt.close(fig)
print("  mean dLST: %.2f C | mean dNDVI: %.3f | share of pixels with NDVI increase: %.1f%%" % (np.nanmean(dlst), np.nanmean(dndvi), 100.0 * np.nanmean(dndvi > 0)))
pd.DataFrame({"Metric": ["mean_dLST_C", "std_dLST_C", "mean_dNDVI", "std_dNDVI", "pct_pixels_NDVI_increase"],
              "Value": [np.nanmean(dlst), np.nanstd(dlst), np.nanmean(dndvi), np.nanstd(dndvi), 100.0 * np.nanmean(dndvi > 0)]}).round(3).to_csv(STATS + "Table_seasonal_change.csv", index=False)

print("DONE")
print(pd.DataFrame(rows, columns=["Layer", "Version", "Roughness"]).to_string(index=False))

