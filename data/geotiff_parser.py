"""
SatQuery AI — GeoTIFF Metadata & Geospatial Coordinate Reference Parser

Extracts true geospatial metadata (CRS, spatial bounds, ground sample distance resolution,
transform matrix, band count) from GeoTIFF imagery using rasterio.
"""

import io
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("GeoTIFFParser")


def extract_geotiff_metadata(img_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Inspects image bytes for GeoTIFF tags and extracts genuine spatial metadata.
    Returns metadata dict if GeoTIFF, or None if standard JPEG/PNG.
    """
    if not img_bytes or len(img_bytes) < 4:
        return None

    # Check for TIFF magic headers (Little-endian 'II\x2a\x00' or Big-endian 'MM\x00\x2a')
    is_tiff = (img_bytes[:4] in (b'II*\x00', b'MM\x00*'))
    if not is_tiff:
        return None

    try:
        import rasterio
        with rasterio.open(io.BytesIO(img_bytes)) as dataset:
            crs_str = dataset.crs.to_string() if dataset.crs else None
            bounds = dataset.bounds
            transform = dataset.transform
            gsd_x = abs(transform[0]) if transform else None
            gsd_y = abs(transform[4]) if transform else None
            
            meta = {
                "is_geotiff": True,
                "driver": dataset.driver,
                "crs": crs_str,
                "bounds": {
                    "left": float(bounds.left),
                    "bottom": float(bounds.bottom),
                    "right": float(bounds.right),
                    "top": float(bounds.top)
                } if bounds else None,
                "resolution_gsd_m": (round(float(gsd_x), 3), round(float(gsd_y), 3)) if gsd_x else None,
                "width": dataset.width,
                "height": dataset.height,
                "bands": dataset.count
            }
            logger.info(f"Successfully extracted GeoTIFF spatial metadata: CRS={crs_str}, bands={dataset.count}")
            return meta
    except Exception as e:
        logger.warning(f"Could not parse GeoTIFF metadata: {e}")
        return None


def calculate_utm_crs(lat: Optional[float], lon: Optional[float]) -> str:
    """
    Dynamically computes the correct WGS 84 / UTM EPSG code for any location on Earth.
    """
    if lat is None or lon is None:
        return "EPSG:4326 (WGS 84 Geographic)"
    try:
        lat = float(lat)
        lon = float(lon)
        zone = int((lon + 180) / 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        hemisphere = "N" if lat >= 0 else "S"
        return f"EPSG:{epsg} (WGS 84 / UTM Zone {zone}{hemisphere})"
    except Exception:
        return "EPSG:4326 (WGS 84 Geographic)"
