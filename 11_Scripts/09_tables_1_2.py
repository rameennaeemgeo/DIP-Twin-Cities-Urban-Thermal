import warnings
warnings.filterwarnings("ignore")
import os
import ee
import pandas as pd

ee.Initialize(project="ai-geo-navigators-task-2")
STATS = "DIP_Twin_Cities_Project/07_Statistics/Tables/"
os.makedirs(STATS, exist_ok=True)

gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
aoi = gaul.filter(ee.Filter.Or(ee.Filter.stringContains("ADM2_NAME", "Islamabad"),
                               ee.Filter.stringContains("ADM2_NAME", "Rawalpindi"))).geometry()

seasons = {"Pre-monsoon (May-June 2025)": ("2025-05-01", "2025-07-01"),
           "Post-monsoon (Sep-Oct 2025)": ("2025-09-01", "2025-11-01")}

def landsat(s):
    return (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(aoi).filterDate(*s))

def s2(s):
    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(*s)

rows = []
for name, s in seasons.items():
    for sensor, col, cloud_key, thr in (("Landsat 8/9 L2", landsat(s), "CLOUD_COVER", 40),
                                       ("Sentinel-2 SR", s2(s), "CLOUDY_PIXEL_PERCENTAGE", 40)):
        all_n = col.size().getInfo()
        sel = col.filter(ee.Filter.lt(cloud_key, thr))
        sel_n = sel.size().getInfo()
        if sensor.startswith("Landsat"):
            dates = sorted(set(sel.aggregate_array("DATE_ACQUIRED").getInfo()))
            l8 = sel.filter(ee.Filter.eq("SPACECRAFT_ID", "LANDSAT_8")).size().getInfo()
            l9 = sel.filter(ee.Filter.eq("SPACECRAFT_ID", "LANDSAT_9")).size().getInfo()
            extra = "Landsat 8: %d, Landsat 9: %d" % (l8, l9)
        else:
            dates = sorted(set(d[:10] for d in sel.aggregate_array("system:index").getInfo()
                               for d in [d.split("_")[0][:8]] for d in ["%s-%s-%s" % (d[:4], d[4:6], d[6:8])]))
            extra = ""
        rows.append([name, sensor, all_n, "cloud < %d%%" % thr, sel_n, len(dates), dates[0], dates[-1], extra, ", ".join(dates)])
        print(name, "|", sensor, "| all:", all_n, "| selected:", sel_n, "| unique dates:", len(dates), "|", extra)

t2 = pd.DataFrame(rows, columns=["Period", "Sensor", "Scenes_found", "Filter", "Scenes_used", "Unique_dates",
                                 "First_date", "Last_date", "Split", "Dates"])
t2.to_csv(STATS + "Table2_image_acquisition.csv", index=False)

t1 = pd.DataFrame([
    ["Landsat 8 C2 L2", "LANDSAT/LC08/C02/T1_L2", "Surface temperature (ST_B10), surface reflectance", "30 m", "May-Oct 2025"],
    ["Landsat 9 C2 L2", "LANDSAT/LC09/C02/T1_L2", "Surface temperature (ST_B10), surface reflectance", "30 m", "May-Oct 2025"],
    ["Sentinel-2 SR Harmonized", "COPERNICUS/S2_SR_HARMONIZED", "NDVI, NDBI, NDWI, spectral features", "10-20 m (aggregated to 30 m)", "May-Oct 2025"],
    ["ESA WorldCover v200", "ESA/WorldCover/v200", "Reference labels for classification", "10 m (aggregated to 30 m)", "2021"],
    ["Dynamic World V1", "GOOGLE/DYNAMICWORLD/V1", "Reference labels for classification", "10 m (aggregated to 30 m)", "May-Oct 2025"],
    ["FAO GAUL 2015 level 2", "FAO/GAUL/2015/level2", "District boundaries (study area)", "Vector", "2015"],
], columns=["Dataset", "Earth_Engine_ID", "Purpose", "Resolution", "Period"])
t1.to_csv(STATS + "Table1_datasets.csv", index=False)
print("DONE")
print(t1.to_string(index=False))
