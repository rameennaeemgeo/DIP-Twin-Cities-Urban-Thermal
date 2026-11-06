import warnings
warnings.filterwarnings("ignore")
import ee

ee.Initialize(project="ai-geo-navigators-task-2")

CRS = "EPSG:32643"
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
isb = gaul.filter(ee.Filter.stringContains("ADM2_NAME", "Islamabad"))
rwp = gaul.filter(ee.Filter.stringContains("ADM2_NAME", "Rawalpindi"))
aoi = isb.merge(rwp).geometry()

seasons = {"pre": ("2025-05-01", "2025-07-01"), "post": ("2025-09-01", "2025-11-01")}

def get_landsat(s):
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    return l8.merge(l9).filterBounds(aoi).filterDate(*s)

def get_s2(s):
    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(*s)

def prep_landsat(img):
    qa, rad = img.select("QA_PIXEL"), img.select("QA_RADSAT")
    mask = (qa.bitwiseAnd(1 << 0).eq(0).And(qa.bitwiseAnd(1 << 1).eq(0))
            .And(qa.bitwiseAnd(1 << 2).eq(0)).And(qa.bitwiseAnd(1 << 3).eq(0))
            .And(qa.bitwiseAnd(1 << 4).eq(0)).And(rad.eq(0)))
    lst = (img.select("ST_B10").multiply(0.00341802).add(149.0)
           .subtract(273.15).rename("LST"))
    return lst.updateMask(mask)

def prep_s2(img):
    scl = img.select("SCL")
    bad = (scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3)).Or(scl.eq(8))
           .Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11)))
    bands = (img.select(["B2", "B3", "B4", "B8", "B11"]).divide(10000)
             .rename(["blue", "green", "red", "nir", "swir1"]))
    return bands.updateMask(bad.Not())

def s2_composite(s):
    col = get_s2(s).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(prep_s2)
    comp = col.median().clip(aoi).setDefaultProjection(ee.Projection(CRS).atScale(10))
    comp = (comp.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
            .reproject(crs=CRS, scale=30))
    ndvi = comp.normalizedDifference(["nir", "red"]).rename("NDVI")
    ndbi = comp.normalizedDifference(["swir1", "nir"]).rename("NDBI")
    ndwi = comp.normalizedDifference(["green", "nir"]).rename("NDWI")
    return comp.addBands([ndvi, ndbi, ndwi]).float()

def lst_composite(s):
    col = get_landsat(s).filter(ee.Filter.lt("CLOUD_COVER", 40)).map(prep_landsat)
    return col.median().clip(aoi).reproject(crs=CRS, scale=30).float()

for k, s in seasons.items():
    print(k, "Landsat scenes:", get_landsat(s).filter(ee.Filter.lt("CLOUD_COVER", 40)).size().getInfo(),
          "| Sentinel-2 scenes:", get_s2(s).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).size().getInfo())

region = aoi.bounds()
jobs = {}
for k, s in seasons.items():
    jobs["S2_%s_30m" % k] = s2_composite(s)
    jobs["LST_%s_30m" % k] = lst_composite(s)

for name, image in jobs.items():
    task = ee.batch.Export.image.toDrive(
        image=image, description=name, folder="DIP_Twin_Cities",
        fileNamePrefix=name, region=region, scale=30, crs=CRS,
        maxPixels=int(1e10), fileFormat="GeoTIFF")
    task.start()
    print("Started:", name)
