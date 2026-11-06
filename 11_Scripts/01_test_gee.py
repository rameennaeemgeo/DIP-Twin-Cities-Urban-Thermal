import ee

ee.Initialize(project="ai-geo-navigators-task-2")

gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
isb = gaul.filter(ee.Filter.stringContains("ADM2_NAME", "Islamabad"))
rwp = gaul.filter(ee.Filter.stringContains("ADM2_NAME", "Rawalpindi"))
aoi = isb.merge(rwp).geometry()

print("Islamabad km2:", isb.geometry().area().divide(1e6).getInfo())
print("Rawalpindi km2:", rwp.geometry().area().divide(1e6).getInfo())
print("Combined km2:", aoi.area().divide(1e6).getInfo())
