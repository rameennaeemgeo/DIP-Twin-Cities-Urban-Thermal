"""Cloud and shadow masks plus sensor scaling for the project composites."""


LANDSAT_OPTICAL_BANDS = ("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6")
OUTPUT_OPTICAL_NAMES = ("blue", "green", "red", "nir", "swir1")
SENTINEL_OPTICAL_BANDS = ("B2", "B3", "B4", "B8", "B11")


def mask_landsat_c2(image):
    """Mask fill, dilated cloud, cirrus, cloud, shadow, snow, and saturation."""
    qa = image.select("QA_PIXEL")
    cloud_bits = sum(1 << bit for bit in (0, 1, 2, 3, 4, 5))
    clear = qa.bitwiseAnd(cloud_bits).eq(0)
    unsaturated = image.select("QA_RADSAT").eq(0)
    return image.updateMask(clear.And(unsaturated))


def mask_sentinel2_scl(image):
    """Mask no-data, saturated, cloud-shadow, cloud, and snow SCL classes."""
    scl = image.select("SCL")
    invalid = scl.eq(0)
    for class_id in (1, 3, 8, 9, 10, 11):
        invalid = invalid.Or(scl.eq(class_id))
    return image.updateMask(invalid.Not())


def scale_sentinel2_optical(image):
    """Return masked Sentinel-2 SR bands as unit-scale reflectance."""
    masked = mask_sentinel2_scl(image)
    return (masked.select(list(SENTINEL_OPTICAL_BANDS)).divide(10000)
            .rename(list(OUTPUT_OPTICAL_NAMES)))


def scale_landsat_optical(image):
    """Return masked Landsat Collection 2 SR bands as surface reflectance."""
    masked = mask_landsat_c2(image)
    return (masked.select(list(LANDSAT_OPTICAL_BANDS)).multiply(0.0000275)
            .add(-0.2).rename(list(OUTPUT_OPTICAL_NAMES)))


def landsat_lst_celsius(image):
    """Return cloud-masked Landsat Collection 2 ST_B10 in degrees Celsius."""
    kelvin = mask_landsat_c2(image).select("ST_B10").multiply(0.00341802).add(149.0)
    return kelvin.subtract(273.15).rename("LST")