import warnings
warnings.filterwarnings("ignore")
import ee

ee.Initialize(project="ai-geo-navigators-task-2")

CRS = "EPSG:32643"
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
isb = gaul.filter(ee.Filter.stringContains("ADM2_NAME", "Islamabad"))
rwp = gaul.filter(ee.Filter.stringContains("ADM2_NAME", "Rawalpindi"))
aoi = isb.merge(rwp).geometry()

# Classes: 1 Built-up, 2 Vegetation, 3 Agriculture, 4 Water, 5 Bare land
wc_img = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
wc = wc_img.remap([50, 10, 20, 30, 40, 80, 60], [1, 2, 2, 2, 3, 4, 5]).rename("WC")
wc = wc.setDefaultProjection(wc_img.projection())
wc = wc.reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1024).reproject(crs=CRS, scale=30)

dw_img = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi)
          .filterDate("2025-05-01", "2025-11-01").select("label").mode())
dw = dw_img.remap([6, 1, 2, 5, 4, 0, 7], [1, 2, 2, 2, 3, 4, 5]).rename("DW")
dw = dw.setDefaultProjection(ee.Projection(CRS).atScale(10))
dw = dw.reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1024).reproject(crs=CRS, scale=30)

ref = ee.Image.cat([wc, dw]).clip(aoi).unmask(0).toByte()

task = ee.batch.Export.image.toDrive(
    image=ref, description="REF_30m", folder="DIP_Twin_Cities",
    fileNamePrefix="REF_30m", region=aoi.bounds(), scale=30, crs=CRS,
    maxPixels=int(1e10), fileFormat="GeoTIFF")
task.start()
print("Started: REF_30m")
