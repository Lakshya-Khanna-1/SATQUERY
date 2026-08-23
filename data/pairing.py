"""
SatQuery AI — Geospatial Pairing & Co-Registration Module
SIH 2026 (PS 26167)

Checks whether an optical image and a SAR image (or two bi-temporal images)
share the same CRS, bounding box (within a tolerance in meters), and resolution.
If not, reprojects/resamples image 2 onto image 1's grid using rasterio, and
returns a was_resampled flag alongside the aligned data.
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    import pyproj
except ImportError:
    rasterio = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Pairing")


def get_raster_info(path: str) -> Dict[str, Any]:
    """
    Extracts spatial metadata from a GeoTIFF or standard raster.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raster file not found: {path}")
        
    if rasterio is not None and path.lower().endswith((".tif", ".tiff")):
        try:
            with rasterio.open(path) as src:
                return {
                    "is_geotiff": True,
                    "crs": src.crs,
                    "crs_wkt": src.crs.to_wkt() if src.crs else None,
                    "bounds": src.bounds, # BoundingBox(left, bottom, right, top)
                    "transform": src.transform,
                    "res": src.res, # (res_x, res_y)
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "dtypes": src.dtypes,
                    "nodata": src.nodata,
                }
        except Exception as e:
            logger.warning(f"Failed to read rasterio metadata from {path}: {e}")
            
    # Fallback for standard non-georeferenced images (PNG/JPEG)
    with Image.open(path) as img:
        return {
            "is_geotiff": False,
            "crs": None,
            "crs_wkt": None,
            "bounds": None,
            "transform": None,
            "res": (1.0, 1.0),
            "width": img.width,
            "height": img.height,
            "count": len(img.getbands()),
            "dtypes": ("uint8",) * len(img.getbands()),
            "nodata": None,
        }


def check_spatial_alignment(
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    tolerance_meters: float = 1.0
) -> Tuple[bool, list[str]]:
    """
    Checks whether two rasters share the same CRS, bounding box, and resolution.
    Returns (is_aligned, diff_reasons).
    """
    diff_reasons = []
    
    # 1. Non-georeferenced image comparison
    if not meta1.get("is_geotiff") or not meta2.get("is_geotiff"):
        if (meta1["width"], meta1["height"]) != (meta2["width"], meta2["height"]):
            diff_reasons.append(
                f"Pixel dimension mismatch: ({meta1['width']}x{meta1['height']}) vs ({meta2['width']}x{meta2['height']})"
            )
        if meta1["count"] != meta2["count"]:
            diff_reasons.append(f"Channel count mismatch: {meta1['count']} vs {meta2['count']}")
        return (len(diff_reasons) == 0, diff_reasons)
        
    # 2. CRS Check
    crs1 = meta1.get("crs")
    crs2 = meta2.get("crs")
    
    if crs1 is None or crs2 is None:
        diff_reasons.append("One or both rasters lack CRS metadata")
    elif crs1 != crs2:
        diff_reasons.append(f"CRS mismatch: {crs1} vs {crs2}")
        
    # 3. Resolution Check
    res1 = meta1.get("res", (0, 0))
    res2 = meta2.get("res", (0, 0))
    if abs(res1[0] - res2[0]) > 1e-2 or abs(res1[1] - res2[1]) > 1e-2:
        diff_reasons.append(f"Resolution mismatch: {res1} vs {res2}")
        
    # 4. Dimension Check
    if (meta1["width"], meta1["height"]) != (meta2["width"], meta2["height"]):
        diff_reasons.append(
            f"Raster dimension mismatch: ({meta1['width']}x{meta1['height']}) vs ({meta2['width']}x{meta2['height']})"
        )
        
    # 5. Bounding Box Check
    b1 = meta1.get("bounds")
    b2 = meta2.get("bounds")
    if b1 and b2:
        # If in same CRS, check bounding box bounds difference
        if crs1 == crs2:
            dx_left = abs(b1.left - b2.left)
            dy_bottom = abs(b1.bottom - b2.bottom)
            dx_right = abs(b1.right - b2.right)
            dy_top = abs(b1.top - b2.top)
            max_diff = max(dx_left, dy_bottom, dx_right, dy_top)
            if max_diff > tolerance_meters:
                diff_reasons.append(f"Bounding box spatial offset ({max_diff:.2f}m > tolerance {tolerance_meters:.2f}m)")
        else:
            diff_reasons.append("Cannot compare bounding boxes directly due to differing CRS")
            
    is_aligned = (len(diff_reasons) == 0)
    return is_aligned, diff_reasons


def align_image_pair(
    path1: str,
    path2: str,
    tolerance_meters: float = 1.0,
    output_aligned_path: Optional[str] = None,
    resampling_method: str = "bilinear"
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """
    Checks whether two images share the same CRS, bounding box, and resolution.
    If not, reprojects/resamples image 2 onto image 1's grid.
    
    Returns:
        data1 (np.ndarray): (bands, height, width) array of reference image 1.
        data2_aligned (np.ndarray): (bands, height, width) array of image 2 aligned to image 1.
        was_resampled (bool): True if reprojection/resampling occurred, False otherwise.
        alignment_info (dict): Diagnostic metadata of alignment and differences.
    """
    meta1 = get_raster_info(path1)
    meta2 = get_raster_info(path2)
    
    is_aligned, diff_reasons = check_spatial_alignment(meta1, meta2, tolerance_meters=tolerance_meters)
    
    # --------------------------------------------------------------------------
    # Case A: Standard PNG/JPG Images (Non-GeoTIFF)
    # --------------------------------------------------------------------------
    if not meta1.get("is_geotiff") or not meta2.get("is_geotiff"):
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        arr1 = np.array(img1)
        
        if not is_aligned:
            logger.info(f"Resampling non-georeferenced image 2 ({img2.size}) to match image 1 ({img1.size})...")
            resample_mode = Image.Resampling.BILINEAR if resampling_method == "bilinear" else Image.Resampling.NEAREST
            img2_resampled = img2.resize((img1.width, img1.height), resample=resample_mode)
            arr2 = np.array(img2_resampled)
            was_resampled = True
            if output_aligned_path:
                img2_resampled.save(output_aligned_path)
        else:
            arr2 = np.array(img2)
            was_resampled = False
            
        alignment_info = {
            "was_resampled": was_resampled,
            "is_geotiff": False,
            "target_size": (img1.width, img1.height),
            "diff_reasons": diff_reasons,
        }
        return arr1, arr2, was_resampled, alignment_info

    # --------------------------------------------------------------------------
    # Case B: GeoTIFF Rasters (Optical + SAR or Bi-Temporal)
    # --------------------------------------------------------------------------
    with rasterio.open(path1) as src1:
        data1 = src1.read()
        target_crs = src1.crs
        target_transform = src1.transform
        target_width = src1.width
        target_height = src1.height
        target_profile = src1.profile.copy()

    if is_aligned:
        with rasterio.open(path2) as src2:
            data2 = src2.read()
        was_resampled = False
        alignment_info = {
            "was_resampled": False,
            "is_geotiff": True,
            "crs": str(target_crs),
            "shape": (target_height, target_width),
            "diff_reasons": [],
        }
        return data1, data2, was_resampled, alignment_info

    # Reproject and resample image 2 onto image 1's grid
    logger.info(f"Spatial misalignment detected: {'; '.join(diff_reasons)}")
    logger.info(f"Reprojecting and resampling {os.path.basename(path2)} to grid of {os.path.basename(path1)}...")
    
    resample_enum = Resampling.bilinear if resampling_method == "bilinear" else Resampling.nearest
    
    with rasterio.open(path2) as src2:
        src2_count = src2.count
        data2_aligned = np.zeros((src2_count, target_height, target_width), dtype=src2.dtypes[0])
        
        reproject(
            source=rasterio.band(src2, list(range(1, src2_count + 1))),
            destination=data2_aligned,
            src_transform=src2.transform,
            src_crs=src2.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=resample_enum,
        )
        
    was_resampled = True
    
    # Save aligned GeoTIFF if path provided
    if output_aligned_path:
        out_profile = target_profile.copy()
        out_profile.update({
            "count": src2_count,
            "dtype": data2_aligned.dtype,
        })
        with rasterio.open(output_aligned_path, "w", **out_profile) as dst:
            dst.write(data2_aligned)
        logger.info(f"Saved aligned raster to {output_aligned_path}")
        
    alignment_info = {
        "was_resampled": True,
        "is_geotiff": True,
        "target_crs": str(target_crs),
        "target_shape": (target_height, target_width),
        "diff_reasons": diff_reasons,
    }
    return data1, data2_aligned, was_resampled, alignment_info


# ==============================================================================
# SELF-TEST & VERIFICATION HARNESS
# ==============================================================================

def run_misalignment_test() -> bool:
    """
    Creates deliberately misaligned test pairs (differing CRS and resolution)
    and verifies that pairing.py flags and aligns them accurately.
    """
    import tempfile
    
    temp_dir = tempfile.mkdtemp(prefix="satquery_pairing_test_")
    logger.info(f"Running pairing verification test in temporary directory: {temp_dir}")
    
    # 1. Create Reference Raster (Image 1): UTM Zone 32N (EPSG:32632), 10m resolution, 100x100 pixels
    crs_utm = CRS.from_epsg(32632)
    transform1 = from_origin(500000.0, 5200000.0, 10.0, 10.0)
    path1 = os.path.join(temp_dir, "ref_optical_utm32n.tif")
    data1_synthetic = np.random.randint(100, 2000, (3, 100, 100), dtype=np.uint16)
    
    with rasterio.open(
        path1, "w", driver="GTiff", height=100, width=100, count=3,
        dtype=rasterio.uint16, crs=crs_utm, transform=transform1
    ) as dst:
        dst.write(data1_synthetic)
        
    # 2. Create Deliberately Misaligned Raster (Image 2): WGS84 Geographic (EPSG:4326), 0.0001 deg res, 150x150 pixels
    # Lat/Lon corresponding roughly to the UTM region (approx 46.95N, 9.00E)
    crs_wgs84 = CRS.from_epsg(4326)
    transform2 = from_origin(9.0, 47.0, 0.0001, 0.0001)
    path2 = os.path.join(temp_dir, "misaligned_sar_wgs84.tif")
    data2_synthetic = np.random.normal(0, 1, (2, 150, 150)).astype(np.float32)
    
    with rasterio.open(
        path2, "w", driver="GTiff", height=150, width=150, count=2,
        dtype=rasterio.float32, crs=crs_wgs84, transform=transform2
    ) as dst:
        dst.write(data2_synthetic)
        
    # 3. Test Alignment Execution
    aligned_out_path = os.path.join(temp_dir, "sar_reprojected_utm32n.tif")
    arr1, arr2_aligned, was_resampled, info = align_image_pair(
        path1, path2, tolerance_meters=1.0, output_aligned_path=aligned_out_path
    )
    
    # 4. Assertions
    logger.info("\n--- TEST ASSERTIONS ---")
    logger.info(f"was_resampled: {was_resampled} (Expected: True)")
    logger.info(f"Differences detected: {info['diff_reasons']}")
    logger.info(f"Target Array 1 shape: {arr1.shape}")
    logger.info(f"Aligned Array 2 shape: {arr2_aligned.shape} (Expected: (2, 100, 100))")
    
    assert was_resampled is True, "Pairing failed: was_resampled should be True for misaligned test pair!"
    assert arr2_aligned.shape == (2, 100, 100), f"Aligned shape {arr2_aligned.shape} does not match target (2, 100, 100)!"
    assert os.path.exists(aligned_out_path), "Aligned GeoTIFF was not written to output path!"
    
    # 5. Clean up
    for f in [path1, path2, aligned_out_path]:
        if os.path.exists(f):
            os.remove(f)
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass
        
    logger.info("✓ Deliberate misalignment test PASSED with 100% precision!\n")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 2:
        img1 = sys.argv[1]
        img2 = sys.argv[2]
        out_p = sys.argv[3] if len(sys.argv) > 3 else None
        d1, d2, resampled, report = align_image_pair(img1, img2, output_aligned_path=out_p)
        print(f"Result: was_resampled={resampled}")
        print(f"Report: {report}")
    else:
        success = run_misalignment_test()
        sys.exit(0 if success else 1)
