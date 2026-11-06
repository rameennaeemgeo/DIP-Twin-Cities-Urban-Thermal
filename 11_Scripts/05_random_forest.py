import os, time, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, precision_recall_fscore_support

ROOT = "DIP_Twin_Cities_Project"
RAW = ROOT + "/02_Raw_Data/"
CLS = ROOT + "/06_Classification/"
MAPS = ROOT + "/08_Final_Maps/"
CHARTS = ROOT + "/09_Charts/"
STATS = ROOT + "/07_Statistics/Tables/"
for d in (CLS + "Classified_Maps/", CLS + "Accuracy_Assessment/", CLS + "Training_Samples/", MAPS, CHARTS, STATS):
    os.makedirs(d, exist_ok=True)

rng = np.random.default_rng(42)
NAMES = {1: "Built-up", 2: "Vegetation", 3: "Agriculture", 4: "Water", 5: "Bare land"}
COLORS = {1: "#d7191c", 2: "#1a9641", 3: "#fdd835", 4: "#2b83ba", 5: "#bdbdbd"}
t0 = time.time()

print("Loading reference layers...")
with rasterio.open(RAW + "REF_30m.tif") as src:
    ref = src.read()
    profile = src.profile
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
wc, dw = ref[0], ref[1]
H, W = wc.shape

print("Loading Sentinel-2 features...")
band_names = ["blue", "green", "red", "nir", "swir1", "NDVI", "NDBI", "NDWI"]
feat_names = []
X = np.empty((16, H, W), dtype="float32")
valid = np.ones((H, W), dtype=bool)
k = 0
for s in ("pre", "post"):
    with rasterio.open(RAW + "S2_" + s + "_30m.tif") as f:
        a = f.read().astype("float32")
    valid &= (a[2] != 0) & (a[3] != 0) & np.all(np.isfinite(a), axis=0)
    X[k:k + 8] = a
    k += 8
    feat_names += [b + "_" + s for b in band_names]
    del a
Xf = X.reshape(16, -1)
print("Valid pixels:", int(valid.sum()))

agree = valid & (wc == dw) & (wc > 0)
print("WorldCover and Dynamic World agree on %.1f%% of valid pixels" % (100.0 * agree.sum() / valid.sum()))
labels = np.where(agree, wc, 0).astype("uint8")
lab_f = labels.ravel()
for c in NAMES:
    print("  agreed pixels, %s: %d" % (NAMES[c], int((lab_f == c).sum())))

# spatial block split (100 x 100 pixels = 3 km blocks) to limit spatial autocorrelation
B = 100
block_id = ((np.arange(H) // B)[:, None] * 1000 + (np.arange(W) // B)[None, :]).astype("int32")
ids = np.unique(block_id)
train_ids = rng.choice(ids, size=int(0.7 * len(ids)), replace=False)
is_train = np.isin(block_id, train_ids).ravel()

def draw(mask_f, n):
    idx_all, y_all = [], []
    for c in NAMES:
        idx = np.flatnonzero((lab_f == c) & mask_f)
        if idx.size == 0:
            print("  WARNING: no samples for", NAMES[c])
            continue
        pick = rng.choice(idx, size=min(n, idx.size), replace=False)
        idx_all.append(pick)
        y_all.append(np.full(pick.size, c, dtype="uint8"))
    idx_all = np.concatenate(idx_all)
    return idx_all, np.concatenate(y_all)

tr_idx, y_train = draw(is_train, 6000)
te_idx, y_test = draw(~is_train, 2000)
X_train, X_test = Xf[:, tr_idx].T, Xf[:, te_idx].T
print("Training samples:", len(y_train), "| Test samples:", len(y_test))

pd.DataFrame({"row": tr_idx // W, "col": tr_idx % W, "class": y_train}).to_csv(CLS + "Training_Samples/training_samples.csv", index=False)
pd.DataFrame({"row": te_idx // W, "col": te_idx % W, "class": y_test}).to_csv(CLS + "Training_Samples/test_samples.csv", index=False)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

print("Accuracy assessment on held-out spatial blocks...")
y_pred = rf.predict(X_test)
classes = [c for c in NAMES if (y_test == c).any() or (y_train == c).any()]
cm = confusion_matrix(y_test, y_pred, labels=classes)
oa = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
p, r, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
acc = pd.DataFrame({"Class": [NAMES[c] for c in classes], "Producers_accuracy": r.round(3),
                    "Users_accuracy": p.round(3), "F1": f1.round(3), "Test_samples": sup})
acc.loc[len(acc)] = ["Overall accuracy", round(oa, 3), None, None, int(sup.sum())]
acc.loc[len(acc)] = ["Kappa", round(kappa, 3), None, None, None]
acc.to_csv(STATS + "Table6_accuracy.csv", index=False)
acc.to_csv(CLS + "Accuracy_Assessment/Table6_accuracy.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(pd.DataFrame(cm, index=[NAMES[c] for c in classes], columns=[NAMES[c] for c in classes]),
            annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Reference")
ax.set_title("Confusion matrix (held-out spatial blocks)")
fig.tight_layout()
fig.savefig(CLS + "Accuracy_Assessment/Confusion_matrix.png", dpi=200)
plt.close(fig)

print("Classifying the full map (this takes a few minutes)...")
vidx = np.flatnonzero(valid.ravel())
pred_full = np.zeros(H * W, dtype="uint8")
CH = 400000
for i in range(0, len(vidx), CH):
    sel = vidx[i:i + CH]
    pred_full[sel] = rf.predict(Xf[:, sel].T)
    if (i // CH) % 5 == 0:
        print("  %d%%" % (100 * i // len(vidx)))
pred = pred_full.reshape(H, W)

profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
with rasterio.open(CLS + "Classified_Maps/LULC_RF_30m.tif", "w", **profile) as dst:
    dst.write(pred, 1)

print("Map 5...")
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

cmap = ListedColormap([COLORS[c] for c in NAMES])
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
arr = np.where(pred == 0, np.nan, pred.astype("float32"))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(arr, cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
decorate(ax)
ax.set_title("Land-cover classification (Random Forest)", fontsize=13, fontweight="bold")
ax.legend(handles=[Patch(facecolor=COLORS[c], label=NAMES[c]) for c in NAMES], title="Land cover",
          loc="lower right", fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(MAPS + "Map5_Land_cover_RF.png", dpi=200)
plt.close(fig)

print("Chart 3: land-cover area...")
counts = np.bincount(pred.ravel(), minlength=6)[1:6]
area_km2 = counts * 900.0 / 1e6
pct = 100.0 * counts / counts.sum()
t5 = pd.DataFrame({"Class": list(NAMES.values()), "Area_km2": area_km2.round(1), "Percent": pct.round(1)})
t5.to_csv(STATS + "Table5_landcover_area.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(t5["Class"], t5["Percent"], color=[COLORS[c] for c in NAMES], edgecolor="black")
for b, v in zip(bars, t5["Percent"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center", fontweight="bold")
ax.set_ylabel("Area (%)")
ax.set_title("Land-cover area, Islamabad-Rawalpindi", fontweight="bold")
fig.tight_layout()
fig.savefig(CHARTS + "Chart3_landcover_area.png", dpi=200)
plt.close(fig)

print("Chart 6: feature importance...")
imp = pd.DataFrame({"Feature": feat_names, "Importance": rf.feature_importances_}).sort_values("Importance")
imp.sort_values("Importance", ascending=False).round(4).to_csv(STATS + "Feature_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp["Feature"], imp["Importance"], color="#007ab8")
for i, v in enumerate(imp["Importance"]):
    ax.text(v + 0.002, i, "%.3f" % v, va="center", fontsize=8)
ax.set_xlabel("Feature importance")
ax.set_title("Feature importances (Random Forest)", fontweight="bold")
fig.tight_layout()
fig.savefig(CHARTS + "Chart6_feature_importance.png", dpi=200)
plt.close(fig)

print("LST by land cover...")
rows = []
for s in ("pre", "post"):
    with rasterio.open(RAW + "LST_" + s + "_30m.tif") as f:
        lst = f.read(1).astype("float32").ravel()
    ok = (lst != 0) & np.isfinite(lst) & (pred_full > 0)
    for c in NAMES:
        idx = np.flatnonzero(ok & (pred_full == c))
        if idx.size == 0:
            continue
        pick = rng.choice(idx, size=min(20000, idx.size), replace=False)
        rows.append(pd.DataFrame({"Season": "Pre-monsoon" if s == "pre" else "Post-monsoon",
                                  "Class": NAMES[c], "LST": lst[pick]}))
d = pd.concat(rows)
d.groupby(["Season", "Class"])["LST"].agg(["mean", "std", "min", "max"]).round(2).to_csv(STATS + "Table_LST_by_landcover.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, s in zip(axes, ("Pre-monsoon", "Post-monsoon")):
    sns.boxplot(data=d[d.Season == s], x="Class", y="LST", hue="Class", order=list(NAMES.values()),
                palette={NAMES[c]: COLORS[c] for c in NAMES}, legend=False, ax=ax)
    ax.set_title("LST by land cover - " + s)
    ax.set_ylabel("Temperature (deg C)")
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(CHARTS + "Chart7_LST_by_landcover.png", dpi=200)
plt.close(fig)

print("DONE in %.1f min" % ((time.time() - t0) / 60))
print(acc.to_string(index=False))
print(t5.to_string(index=False))
