"""Create seasonal LST maps, histograms, and summary statistics."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "02_Raw_Data"
OUTPUT_DIR = Path(__file__).resolve().parent
SEASONS = ("pre", "post")


def summarize(source_path, season):
    with rasterio.open(source_path) as source:
        values = source.read(1).astype("float32")
        bounds = source.bounds
        valid = np.isfinite(values) & (values != 0)
        values = np.where(valid, values, np.nan)

    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("No valid LST pixels found in %s" % source_path)

    lower, upper = np.percentile(finite, [2, 98])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    image = axes[0].imshow(
        np.ma.masked_invalid(values),
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap="RdYlBu_r",
        vmin=lower,
        vmax=upper,
    )
    axes[0].set(title="LST - %s-monsoon" % season,
                xlabel="Easting (m)", ylabel="Northing (m)")
    fig.colorbar(image, ax=axes[0], label="Temperature (deg C)", shrink=0.82)
    axes[1].hist(finite, bins=40, color="#c65d3b", edgecolor="white")
    axes[1].set(title="LST distribution - %s-monsoon" % season,
                xlabel="Temperature (deg C)", ylabel="Pixel count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / ("LST_%s_map_histogram.png" % season), dpi=180)
    plt.close(fig)

    return {
        "season": season,
        "valid_pixels": int(finite.size),
        "minimum_c": float(np.min(finite)),
        "p05_c": float(np.percentile(finite, 5)),
        "mean_c": float(np.mean(finite)),
        "median_c": float(np.median(finite)),
        "p95_c": float(np.percentile(finite, 95)),
        "maximum_c": float(np.max(finite)),
        "std_c": float(np.std(finite)),
    }


def main():
    rows = []
    for season in SEASONS:
        source = INPUT_DIR / ("LST_%s_30m.tif" % season)
        if not source.is_file():
            raise FileNotFoundError("Expected downloaded Landsat LST composite: %s" % source)
        rows.append(summarize(source, season))

    with (OUTPUT_DIR / "LST_statistics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()