"""Calculate and export seasonal Landsat Collection 2 LST composites."""

import sys
from pathlib import Path

import ee


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLOUD_MASK_DIR = PROJECT_ROOT / "03_Preprocessed_Images" / "Cloud_Masked"
sys.path.insert(0, str(CLOUD_MASK_DIR))

from cloud_masking import landsat_lst_celsius


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


def landsat_collection(start, end, aoi):
    landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    return (landsat8.merge(landsat9)
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", 40)))


def main():
    ee.Initialize(project=EE_PROJECT)
    aoi = study_area()
    region = aoi.bounds()

    for season, (start, end) in SEASONS.items():
        collection = landsat_collection(start, end, aoi).map(landsat_lst_celsius)
        image = collection.median().clip(aoi).reproject(crs=CRS, scale=30).float()
        print("Season %s: %s usable Landsat scenes" %
              (season, collection.size().getInfo()))
        name = "LST_%s_30m" % season
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