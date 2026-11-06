import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr, linregress

ROOT = "DIP_Twin_Cities_Project"
RAW = ROOT + "/02_Raw_Data/"
MAPS = ROOT + "/08_Final_Maps/"
CHARTS = ROOT + "/09_Charts/"
STATS = ROOT + "/07_Statistics/Tables/"
for d in (MAPS, CHARTS, STATS):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({"font.size": 10})
rng = np.random.default_rng(42)

def load(season):
    with rasterio.open(RAW + "S2_" + season + "_30m.tif") as s2:
        a = s2.read().astype("float32")
        extent = (s2.bounds.left, s2.bounds.right, s2.bounds.bottom, s2.bounds.top)
    with rasterio.open(RAW + "LST_" + season + "_30m.tif") as l:
        lst = l.read(1).astype("float32")
    names = ["blue", "green", "red", "nir", "swir1", "NDVI", "NDBI", "NDWI"]
    d = {n: a[i] for i, n in enumerate(names)}
    valid = (d["red"] != 0) & (d["nir"] != 0) & np.isfinite(d["NDVI"])
    for k in d:
        d[k] = np.where(valid, d[k], np.nan)
    d["LST"] = np.where((lst != 0) & np.isfinite(lst), lst, np.nan)
    return d, extent

print("Loading data...")
data = {}
extent = None
for s in ("pre", "post"):
    data[s], extent = load(s)

TITLE = {"pre": "Pre-monsoon (May-June 2025)", "post": "Post-monsoon (Sep-Oct 2025)"}

def decorate(ax, extent):
    left, right, bottom, top = extent
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("Easting (m), UTM 43N")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain")
    ax.tick_params(labelsize=7)
    ax.grid(color="grey", lw=0.3, alpha=0.5)
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.85), xycoords="axes fraction",
                ha="center", fontsize=14, fontweight="bold",
                arrowprops=dict(facecolor="black", width=4, headwidth=12))
    x0 = left + 0.05 * (right - left)
    y0 = bottom + 0.05 * (top - bottom)
    ax.plot([x0, x0 + 20000], [y0, y0], color="black", lw=4)
    ax.text(x0 + 10000, y0 + 0.015 * (top - bottom), "20 km", ha="center", fontsize=8)

def continuous_map(arr, cmap, vmin, vmax, title, cbar_label, fname):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
    decorate(ax, extent)
    ax.set_title(title, fontsize=13, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(MAPS + fname, dpi=200)
    plt.close(fig)

def classed_map(arr, breaks, labels, colors, title, legend_title, fname):
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(breaks, cmap.N)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
    decorate(ax, extent)
    ax.set_title(title, fontsize=13, fontweight="bold")
    handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=handles, title=legend_title, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(MAPS + fname, dpi=200)
    plt.close(fig)

print("Map 1: LST...")
lst_classes = {
    "pre":  ([-50, 35, 40, 45, 50, 100], ["< 35", "35-40", "40-45", "45-50", "> 50"]),
    "post": ([-50, 25, 30, 35, 40, 100], ["< 25", "25-30", "30-35", "35-40", "> 40"]),
}
lst_colors = ["#a6d96a", "#ffffbf", "#fdae61", "#d73027", "#7f0000"]
for s in ("pre", "post"):
    br, lb = lst_classes[s]
    classed_map(data[s]["LST"], br, lb, lst_colors,
                "Land Surface Temperature - " + TITLE[s], "LST (deg C)", "Map1_LST_" + s + ".png")

print("Map 2: NDVI...")
ndvi_cmap = ListedColormap(["#8c2d04", "#d9a441", "#efe9a8", "#a6d96a", "#1a9850", "#00441b"])
for s in ("pre", "post"):
    continuous_map(data[s]["NDVI"], ndvi_cmap, -0.1, 0.8,
                   "NDVI - " + TITLE[s], "NDVI", "Map2_NDVI_" + s + ".png")

print("Map 3: NDBI...")
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, s in zip(axes, ("pre", "post")):
    im = ax.imshow(data[s]["NDBI"], cmap="RdYlBu_r", vmin=-0.4, vmax=0.3, extent=extent)
    decorate(ax, extent)
    ax.set_title("NDBI - " + TITLE[s], fontweight="bold")
fig.colorbar(im, ax=axes, shrink=0.6, label="NDBI")
fig.savefig(MAPS + "Map3_NDBI_pre_vs_post.png", dpi=200)
plt.close(fig)

print("Map 4: NDWI...")
ndwi_breaks = [-1, -0.3, -0.1, 0.0, 1]
ndwi_labels = ["Very low", "Low", "Moderate", "High (likely water)"]
ndwi_colors = ["#e0f3f8", "#abd9e9", "#4575b4", "#08306b"]
for s in ("pre", "post"):
    classed_map(data[s]["NDWI"], ndwi_breaks, ndwi_labels, ndwi_colors,
                "NDWI (water content) - " + TITLE[s], "NDWI class", "Map4_NDWI_" + s + ".png")

print("Sampling pixels for charts...")
def sample_pixels(s, n=60000):
    d = data[s]
    ok = np.isfinite(d["LST"]) & np.isfinite(d["NDVI"]) & np.isfinite(d["NDBI"]) & np.isfinite(d["NDWI"])
    idx = np.flatnonzero(ok.ravel())
    pick = rng.choice(idx, size=min(n, idx.size), replace=False)
    return {k: d[k].ravel()[pick] for k in ("LST", "NDVI", "NDBI", "NDWI", "blue", "green", "red", "nir", "swir1")}

samples = {s: sample_pixels(s) for s in ("pre", "post")}

print("Chart 1: histograms...")
sns.set_style("whitegrid")
for s in ("pre", "post"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (key, col) in zip(axes, [("LST", "#6bb9d6"), ("NDVI", "#7fd57f"), ("NDBI", "#e87474")]):
        sns.histplot(samples[s][key], bins=60, kde=True, color=col, ax=ax, edgecolor="black", linewidth=0.3)
        ax.set_title("Distribution of " + key + " - " + TITLE[s], fontsize=10)
        ax.set_xlabel("Temperature (deg C)" if key == "LST" else key)
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(CHARTS + "Chart1_histograms_" + s + ".png", dpi=200)
    plt.close(fig)

print("Chart 2: NDVI vs LST...")
stats_rows = []
for s in ("pre", "post"):
    x, y = samples[s]["NDVI"], samples[s]["LST"]
    reg = linregress(x, y)
    idx = rng.choice(x.size, size=min(15000, x.size), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[idx], y[idx], s=4, color="grey", alpha=0.4)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, reg.slope * xs + reg.intercept, color="black", lw=2)
    ax.text(0.97, 0.95, "LST = %.2f x NDVI + %.2f\nR$^2$ = %.2f" % (reg.slope, reg.intercept, reg.rvalue ** 2),
            transform=ax.transAxes, ha="right", va="top", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8))
    ax.set_xlabel("NDVI")
    ax.set_ylabel("LST (deg C)")
    ax.set_title("NDVI vs LST - " + TITLE[s])
    fig.tight_layout()
    fig.savefig(CHARTS + "Chart2_NDVI_vs_LST_" + s + ".png", dpi=200)
    plt.close(fig)
    for nm in ("NDVI", "NDBI", "NDWI"):
        pr = pearsonr(samples[s][nm], samples[s]["LST"])[0]
        sr = spearmanr(samples[s][nm], samples[s]["LST"])[0]
        stats_rows.append([s, nm, round(float(pr), 3), round(float(sr), 3), round(float(linregress(samples[s][nm], samples[s]["LST"]).rvalue ** 2), 3)])

print("Chart 4: correlation heatmaps...")
for s in ("pre", "post"):
    df = pd.DataFrame(samples[s])[["LST", "NDVI", "NDBI", "NDWI", "blue", "green", "red", "nir", "swir1"]]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Correlation matrix - " + TITLE[s])
    fig.tight_layout()
    fig.savefig(CHARTS + "Chart4_correlation_" + s + ".png", dpi=200)
    plt.close(fig)

print("Chart 5: seasonal box plots...")
frames = []
for s in ("pre", "post"):
    for k in ("LST", "NDVI", "NDBI"):
        frames.append(pd.DataFrame({"Season": "Pre-monsoon" if s == "pre" else "Post-monsoon",
                                    "Variable": k, "Value": samples[s][k][:20000]}))
bp = pd.concat(frames)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, k in zip(axes, ("LST", "NDVI", "NDBI")):
    sns.boxplot(data=bp[bp.Variable == k], x="Season", y="Value", hue="Season",
                palette=["#e6ab5a", "#7fb8d6"], legend=False, ax=ax)
    ax.set_title(k + " by season")
    ax.set_ylabel("deg C" if k == "LST" else k)
fig.tight_layout()
fig.savefig(CHARTS + "Chart5_seasonal_boxplots.png", dpi=200)
plt.close(fig)

print("Tables...")
rows3 = []
for s in ("pre", "post"):
    for k in ("LST", "NDVI", "NDBI", "NDWI"):
        a = data[s][k]
        rows3.append([s, k, np.nanmean(a), np.nanmin(a), np.nanmax(a), np.nanstd(a)])
pd.DataFrame(rows3, columns=["Season", "Layer", "Mean", "Min", "Max", "StdDev"]).round(3).to_csv(STATS + "Table3_4_statistics.csv", index=False)
corr = pd.DataFrame(stats_rows, columns=["Season", "Index", "Pearson_vs_LST", "Spearman_vs_LST", "R2"])
corr.to_csv(STATS + "Table7_correlation.csv", index=False)

print("DONE")
print(pd.DataFrame(rows3, columns=["Season", "Layer", "Mean", "Min", "Max", "StdDev"]).round(3).to_string(index=False))
print(corr.to_string(index=False))
