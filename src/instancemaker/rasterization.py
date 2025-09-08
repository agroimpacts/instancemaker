#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tile-wise multi-band COG rasterization (with chunked processing).
Processes large tile sets in manageable batches to avoid memory errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from tqdm import tqdm

FIRST_BAND = "tile"
CHUNK_COUNT = 10


def process_one_tile(
    polygons_path: str,
    tile_row: Dict,
    out_dir: str,
    res_deg: float,
    nodata: float,
    all_touched: bool,
    attrs: List[str],
) -> str:
    try:
        tile_id = tile_row[FIRST_BAND]
        geom = tile_row["geometry"]
        xmin, ymin, xmax, ymax = geom.bounds
        width = int(np.ceil((xmax - xmin) / res_deg))
        height = int(np.ceil((ymax - ymin) / res_deg))
        if width <= 0 or height <= 0:
            return f"[skip] tile {tile_id} zero-size"

        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        tmp_tif = out_dir_p / f"tile_{tile_id}.tmp.tif"
        cog_tif = out_dir_p / f"tile_{tile_id}_{1 + len(attrs)}bands_cog.tif"

        if cog_tif.exists():
            return str(cog_tif)

        band_names = [FIRST_BAND] + attrs

        profile = dict(
            driver="GTiff",
            height=height,
            width=width,
            count=len(band_names),
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=nodata,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="DEFLATE",
            predictor=2,
            BIGTIFF="IF_SAFER",
        )

        gdf = gpd.read_parquet(polygons_path)
        if str(gdf.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        sel = gdf[gdf.intersects(geom)]

        with rasterio.open(tmp_tif, "w", **profile) as dst:
            dst.write(
                np.full((height, width), float(tile_id), dtype="float32"),
                1,
            )
            dst.set_band_description(1, FIRST_BAND)

            for bidx, col in enumerate(attrs, start=2):
                band = np.full((height, width), nodata, dtype="float32")
                if not sel.empty:
                    values = sel[col].to_numpy()
                    shapes = [
                        (g, float(v) if np.isfinite(v) else nodata)
                        for g, v in zip(sel.geometry, values)
                    ]
                    band = rasterize(
                        shapes,
                        out_shape=(height, width),
                        transform=transform,
                        all_touched=all_touched,
                        dtype="float32",
                    )
                dst.write(band, bidx)
                dst.set_band_description(bidx, col)

            dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")

        rio_copy(
            tmp_tif,
            cog_tif,
            driver="COG",
            compress="DEFLATE",
            predictor=2,
            blocksize=512,
            bigtiff="IF_SAFER",
            overview_resampling="NEAREST",
            copy_src_overviews=True,
        )

        Path(tmp_tif).unlink(missing_ok=True)
        return str(cog_tif)

    except Exception as e:
        return f"[error] tile {tile_row.get(FIRST_BAND)} -> {e}"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polygons", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ids-file")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--res", type=float, default=0.000025)
    ap.add_argument("--nodata", type=float, default=0.0)
    ap.add_argument("--all-touched", action="store_true")
    ap.add_argument("--attributes")
    ap.add_argument("--auto-attrs", action="store_true")
    ap.add_argument("--attr-exclude")
    return ap.parse_args()


def run_rasterization(
    polygons_path,
    tiles_path,
    out_dir,
    workers,
    res_deg,
    nodata,
    all_touched,
    attributes,
):
    tiles = gpd.read_file(tiles_path)
    if str(tiles.crs).upper() != "EPSG:4326":
        tiles = tiles.to_crs("EPSG:4326")

    if FIRST_BAND not in tiles.columns:
        raise SystemExit(f"tiles vector must contain a '{FIRST_BAND}' field.")
    if tiles.empty:
        raise SystemExit("No tiles to process.")

    gdf_head = gpd.read_parquet(polygons_path)
    if str(gdf_head.crs).upper() != "EPSG:4326":
        gdf_head = gdf_head.to_crs("EPSG:4326")

    attrs = attributes
    print(f"[info] Final band order: [{FIRST_BAND}] + {attrs}")

    all_rows: List[Dict] = [
        {FIRST_BAND: r[FIRST_BAND], "geometry": r.geometry}
        for _, r in tiles.iterrows()
    ]
    chunk_size = int(np.ceil(len(all_rows) / CHUNK_COUNT))
    n_chunks = len(all_rows) // chunk_size + int(len(all_rows) % chunk_size != 0)

    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = all_rows[start:end]

        print(f"[chunk {i + 1}] Processing {len(chunk)} tiles...")
        results = []

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    process_one_tile,
                    polygons_path,
                    row,
                    out_dir,
                    res_deg,
                    nodata,
                    all_touched,
                    attrs,
                )
                for row in chunk
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Chunk {i + 1}"):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append(f"[error] {e}")

        ok = [r for r in results if r and r.endswith(".tif")]
        errs = [r for r in results if not (r and r.endswith(".tif"))]

        print(
            f"[chunk {i + 1}] Done. {len(ok)} COGs written. Errors: {len(errs)}"
        )
        for e in errs:
            print(e)


def main():
    args = parse_args()
    run_rasterization(
        polygons_path=args.polygons,
        tiles_path=args.tiles,
        out_dir=args.out_dir,
        workers=args.workers,
        res_deg=args.res,
        nodata=args.nodata,
        all_touched=args.all_touched,
        attributes=args.attributes,
    )


if __name__ == "__main__":
    main()





