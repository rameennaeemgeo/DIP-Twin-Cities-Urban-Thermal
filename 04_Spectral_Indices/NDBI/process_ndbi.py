"""Calculate, visualize, and summarize NDBI from the seasonal S2 stacks."""

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
NODATA = -9999.0


def calculate(source_path, season):
    output_path = OUTPUT_DIR / ("NDBI_%s_30m.tif" % season)
    figure_path = OUTPUT_DIR / ("NDBI_%s.png" % season)
    with rasterio.open(source_path) as source:
        nir = source.read(4).astype("float32")
        swir1 = source.read(5).astype("float32")
        denominator = swir1 + nir
        valid = (np.isfinite(nir) & np.isfinite(swir1) & (nir != 0) & (swir1 != 0)
                 & (np.abs(denominator) > 1e-6))
        values = np.full(nir.shape, np.nan, dtype="float32")
        np.divide(swir1 - nir, denominator, out=values, where=valid)

        profile = source.profile.copy()
        bounds = source.bounds
        profile.update(count=1, dtype="float32", nodata=NODATA, compress="deflate")
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(np.where(np.isfinite(values), values, NODATA), 1)

    shown = np.ma.masked_invalid(values)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(shown, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                      origin="upper", cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set(title="NDBI - %s-monsoon" % season, xlabel="Easting (m)", ylabel="Northing (m)")
    fig.colorbar(image, ax=ax, label="NDBI", shrink=0.8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    finite = values[np.isfinite(values)]
    stats = {
        "season": season,
        "valid_pixels": int(finite.size),
        "minimum": float(np.min(finite)) if finite.size else float("nan"),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "maximum": float(np.max(finite)) if finite.size else float("nan"),
    }
    return stats


def main():
    rows = []
    for season in SEASONS:
        source = INPUT_DIR / ("S2_%s_30m.tif" % season)
        if not source.is_file():
            raise FileNotFoundError("Expected downloaded seasonal composite: %s" % source)
        rows.append(calculate(source, season))

    with (OUTPUT_DIR / "NDBI_statistics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()