from __future__ import annotations
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import geopandas as gpd
import glob
from tqdm import tqdm
import logging
import re


def compute_shape_metrics(gdf: gpd.GeoDataFrame, area_crs: str) -> gpd.GeoDataFrame:
    proj = gdf.to_crs(area_crs)
    out = gdf.copy()
    if len(out) == 0:
        return out
    # area_m2
    out['area_m2'] = proj.geometry.area
    # area_ha
    out['area_ha'] = out['area_m2'] / 10000.0
    # perimeter_m
    out['perimeter_m'] = proj.geometry.length

    A = out['area_m2']
    P = out['perimeter_m']
    A_pos = A.where(A > 0, np.nan)
    P_pos = P.where(P > 0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out['compactness'] = (4.0 * math.pi * A) / (P ** 2)
        out['shape_index'] = P / (2.0 * np.sqrt(math.pi * A))
        out['interior_edge'] = P / A
        out["fractal_dim"]   = 2.0 * np.log(P_pos) / np.log(A_pos)

    return out

# Each tile's polygons will receive a unique polygon_id based on this multiplier.
# For example, with MULTIPLIER = 1,000,000:
# tile_id = 824621, polygon_index = 5 → polygon_id = 824621000005
MULTIPLIER = 1_000_000

def attribute_shape_metrics(in_path: str,
                            out_path: str,
                            area_crs: str) -> tuple[str, None, bool]:
    gdf = gpd.read_file(in_path)
    out = compute_shape_metrics(gdf, area_crs)

    # Extract tile_id from filename like "tile840216_2018-06_polygons.geojson"
    match = re.search(r"tile(\d{6})_", Path(in_path).stem)
    if match:
        tile_id = int(match.group(1))
    else:
        raise ValueError(f"Could not extract tile_id from filename: {in_path}")

    out['tile_id'] = tile_id
    out['polygon_index'] = np.arange(len(out))
    out['polygon_id'] = tile_id * MULTIPLIER + out['polygon_index']
    out['polygon_id'] = out['polygon_id'].astype('int64')
    out.to_file(out_path, driver="GeoJSON")
    return (str(out_path), None, True)



def compute_shape_metrics_parallel(
    merged_polygon_dir: str, 
    attributed_merged_polygon_dir: str, 
    area_crs: str, num_workers: int
) -> None:
    if isinstance(merged_polygon_dir, Path):
        merged_polygon_dir = str(merged_polygon_dir)
    if isinstance(attributed_merged_polygon_dir, Path):
        attributed_merged_polygon_dir = str(attributed_merged_polygon_dir)
    input_geojsons = glob.glob(merged_polygon_dir + "/*.geojson")
    output_geojsons = []
    failures = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_path = {}
        for in_path in input_geojsons:
            out_path = attributed_merged_polygon_dir + "/" + in_path.split("/")[-1]
            future = executor.submit(attribute_shape_metrics, in_path, out_path, area_crs)
            future_to_path[future] = (in_path, out_path)

        # Collect results
        with tqdm(total=len(input_geojsons), desc="Computing shape metrics") as pbar:
            for future in as_completed(future_to_path):
                in_path, expected_out_path = future_to_path[future]
                try:
                    out_path, err, flag = future.result()
                    if flag:
                        output_geojsons.append(out_path)
                    else:
                        print(f"Error at: {in_path}, {err}")
                        failures.append((out_path, err))
                except Exception as exc:
                    print(f"Exception at: {in_path}, {exc}")
                    failures.append((expected_out_path, str(exc)))
                pbar.update(1)

    logging.info(f"success: {len(output_geojsons)}/{len(input_geojsons)}, "
                 f"failures: {len(failures)}/{len(input_geojsons)}")
