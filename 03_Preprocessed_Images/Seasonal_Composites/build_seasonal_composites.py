"""Build and export the project Sentinel-2 seasonal feature composites."""

import sys
from pathlib import Path

import ee


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLOUD_MASK_DIR = PROJECT_ROOT / "03_Preprocessed_Images" / "Cloud_Masked"
sys.path.insert(0, str(CLOUD_MASK_DIR))

from cloud_masking import scale_sentinel2_optical


EE_PROJECT = "ai-geo-navigators-task-2"
CRS = "EPSG:32643"
SEASONS = {
    "pre": ("2025-05-01", "2025-07-01"),
    "post": ("2025-09-01", "2025-11-01"),
}


def study_area():
    districts = (ee.FeatureCollection("FAO/GAUL/2015/level2")
                 .filter(ee.Filter.eq("ADM0_NAME", "Pakistan")))
    islamabad = districts.filter(ee.Filter.stringContains("ADM2_NAME", "Islamabad"))
    rawalpindi = districts.filter(ee.Filter.stringContains("ADM2_NAME", "Rawalpindi"))
    return islamabad.merge(rawalpindi).geometry()


def seasonal_composite(start, end, aoi):
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                  .map(scale_sentinel2_optical))

    optical = collection.median().clip(aoi)
    optical = optical.setDefaultProjection(ee.Projection(CRS).atScale(10))
    optical = (optical.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
               .reproject(crs=CRS, scale=30))

    ndvi = optical.normalizedDifference(["nir", "red"]).rename("NDVI")
    ndbi = optical.normalizedDifference(["swir1", "nir"]).rename("NDBI")
    ndwi = optical.normalizedDifference(["green", "nir"]).rename("NDWI")
    return optical.addBands([ndvi, ndbi, ndwi]).float(), collection.size()


def main():
    ee.Initialize(project=EE_PROJECT)
    aoi = study_area()
    region = aoi.bounds()

    for season, (start, end) in SEASONS.items():
        image, scene_count = seasonal_composite(start, end, aoi)
        print("Season %s: %s usable Sentinel-2 scenes" %
              (season, scene_count.getInfo()))
        name = "S2_%s_30m" % season
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=name,
            folder="DIP_Twin_Cities",
            fileNamePrefix=name,
            region=region,
            scale=30,
            crs=CRS,
            maxPixels=int(1e10),
            fileFormat="GeoTIFF",
        )
        task.start()
        print("Started export:", name)


if __name__ == "__main__":
    main()