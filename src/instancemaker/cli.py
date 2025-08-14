#!/usr/bin/env python3
import yaml
import click
from pathlib import Path
import numpy as np
import pandas as pd
from instancemaker import MakeInstances, MergeInstances, compute_shape_metrics_parallel
from instancemaker.utils import *
import concurrent.futures
from functools import partial
import logging

# configure logging
def configure_logging(log_file=None, level=logging.INFO):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    if log_file:
        log_path = Path(log_file)
        log_dir = log_path.parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    root_logger.setLevel(level)

@click.group()
def cli():
    """instancemaker CLI with subcommands for makefields and mergefields."""
    pass

def get_config_value(cli_value, config_dict, key):
    return cli_value if cli_value is not None \
        else config_dict.get(key, None)

def process_tile(row, mki, config):
    logging.info(f"Processing tile {row['tile']} ({row['year']},{row['date']})")
    pred_filename = Path(config['pred_dir']) / row['prediction']
    polygon_filename = Path(config['polygon_dir']) / get_filename(
        config['polygon_filename_template'], row['tile'], row['year'], 
        row['date']
    )
    if config['do_labelling']:
        if config['labeled_filename_template']:
            labeled_filename = Path(config['label_dir']) / get_filename(
                config['labeled_filename_template'], row['tile'], row['year'], 
                row['date']
            )
        else:
            labeled_filename = None
        labeled_prediction = mki.make_labeled_prediction(
            prediction_filename=pred_filename, 
            labeled_filename=labeled_filename, 
            threshold=config['threshold']
        )
        mki.polygonize(
            image=labeled_prediction,
            simplify=config['simplify'],
            labeled=True,
            polygon_filename=polygon_filename
        )
    else:
        mki.polygonize(
            image=pred_filename,
            simplify=config['simplify'],
            labeled=False,
            polygon_filename=polygon_filename
        )
    logging.info(f"Finished processing tile {row['tile']}")

@cli.command()
@click.option('--config', default='configs/example-config.yaml', type=str, 
              help='Path to config file.')
@click.option('--do-labelling', default=None, type=bool, 
              help='Whether to label score maps before polygonizing or not.')
@click.option('--label-dir', default=None, type=str, help='Label directory.')
@click.option('--polygon-dir', default=None, type=str, 
              help='Segmentation directory.')
@click.option('--catalog-file', default=None, type=str, 
              help='CSV catalog file with tiles ID, dates, pred & image names.')
@click.option('--pred-dir', default=None, type=str, 
              help='Directory holding prediction tiles.')
@click.option('--threshold', default=None, type=int, 
              help='Threshold value to apply in hardening a score map.')
@click.option('--erosion-iterations', default=None, type=int, 
              help='Erosion iterations.')
@click.option('--dilation-iterations', default=None, type=int, 
              help='Dilation iterations.')
@click.option('--simplify', default=None, type=float, 
              help='Tolerance value to apply in simplifying polygons.')
@click.option('--polygon-filename-template', default=None, type=str,
              help='Polygon filename template.')
@click.option('--prediction-filename-template', default=None, type=str,
              help='Prediction filename template.')
@click.option('--labeled-filename-template', default=None, type=str,
              help='Labeled filename template.')
@click.option('--num-workers', default=1, type=int, 
              help='Number of parallel workers.')
@click.option('--log-file', default=None, type=str, help='Log file path.')

def makeinstances(config, do_labelling, label_dir, polygon_dir, 
                  catalog_file, pred_dir, threshold, erosion_iterations, 
                  dilation_iterations, simplify, polygon_filename_template, 
                  prediction_filename_template, labeled_filename_template,
                  num_workers, log_file):
    """Run MakeInstances methods (score map labeling and polygonization)."""

    # Load config file
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)

    # Determine log file: CLI arg takes precedence, else config, else None
    log_file = get_config_value(log_file, config_data, 'log_file')
    configure_logging(log_file=log_file)
    logging.info("Running makeinstances command")
    
    polygon_dir = get_config_value(polygon_dir, config_data, 'polygon_dir')
    catalog_file = get_config_value(catalog_file, config_data, 
                                   'catalog_file')
    do_labelling = get_config_value(do_labelling, config_data, 
                                   'do_labelling')
    label_dir = get_config_value(label_dir, config_data, 'label_dir')
    pred_dir = get_config_value(pred_dir, config_data, 'pred_dir')
    kernel = np.array(config_data.get('kernel', None))
    threshold = get_config_value(threshold, config_data, 'threshold')
    erosion_iterations = get_config_value(erosion_iterations, config_data, 
                                          'erosion_iterations')
    dilation_iterations = get_config_value(dilation_iterations, config_data, 
                                           'dilation_iterations')
    polygon_filename_template = get_config_value(
        polygon_filename_template, config_data, 'polygon_filename_template'
    )
    prediction_filename_template = get_config_value(
        prediction_filename_template, config_data, 
        'prediction_filename_template'
    )
    labeled_filename_template = get_config_value(
        labeled_filename_template, config_data, 
        'labeled_filename_template'
    )
    simplify = get_config_value(simplify, config_data, 'simplify')
    
    check_dir([label_dir, polygon_dir])
    
    # Prepare catalog
    catalog = pd.read_csv(
        catalog_file, 
        dtype={"tile": int, "year": str, "date": str, "prediction": str, 
               "image": str}
    )
    
    # Prepare config dict for passing to workers
    config_dict = {
        'pred_dir': pred_dir,
        'polygon_dir': polygon_dir,
        'polygon_filename_template': polygon_filename_template,
        'do_labelling': do_labelling,
        'labeled_filename_template': labeled_filename_template,
        'label_dir': label_dir,
        'threshold': threshold,
        'simplify': simplify,
    }

    mki = MakeInstances(
        kernel=kernel,
        threshold=threshold,
        erosion_iterations=erosion_iterations,
        dilation_iterations=dilation_iterations
    )

    rows = [row for _, row in catalog.iterrows()]
    if num_workers == 1:
        # Serial execution
        logging.info("Processing tiles in serial")
        for row in rows:
            process_tile(row, mki, config_dict)
    else:
        # Parallel execution
        logging.info("Running in parallel with %d workers", num_workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
            as executor:
            executor.map(partial(process_tile, mki=mki, config=config_dict), 
                         rows)
    logging.info("Completed makeinstances command")

@cli.command()
@click.option('--config', default='configs/example-config.yaml', type=str, 
              help='Path to config file.')
@click.option('--tile-geojson', default=None, type=str, 
              help='Name of tile geojson file.')
@click.option('--polygon-dir', default=None, type=str, 
              help='Segmentation directory.')
@click.option('--merged-polygon-dir', default=None, type=str, 
              help='Output directory for writing merged polygon files.')
@click.option('--catalog-file', default=None, type=str, 
              help='Catalog file.')
@click.option('--num-workers', default=None, type=int, 
              help='Number of workers for parallelized processes.')
@click.option('--merged-parquet-file', default=None, type=str, 
              help='Output geoparquet filename.')
@click.option('--log-file', default=None, type=str, help='Log file path.')

def mergeinstances(config, tile_geojson, polygon_dir, merged_polygon_dir, 
                catalog_file, num_workers, merged_parquet_file, log_file):
    """Run MergeInstances methods (polygon merging across tile boundaries)."""
    # Load config file
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)

    # Determine log file: CLI arg takes precedence, else config, else None
    log_file = get_config_value(log_file, config_data, 'log_file')
    configure_logging(log_file=log_file)
    logging.info("Running mergeinstances command")
    
    tile_geojson = get_config_value(tile_geojson, config_data, 'tile_geojson')
    polygon_dir = get_config_value(polygon_dir, config_data, 'polygon_dir')
    merged_polygon_dir = get_config_value(merged_polygon_dir, config_data, 
                                          'merged_polygon_dir')
    catalog_file = get_config_value(catalog_file, config_data, 'catalog_file')
    num_workers = get_config_value(num_workers, config_data, 'num_workers')
    merged_parquet_file = get_config_value(merged_parquet_file, config_data, 
                                           'merged_parquet_file')
    # Prepare catalog
    catalog = pd.read_csv(
        catalog_file, 
        dtype={"tile": int, "year": str, "date": str, "prediction": str, 
               "image": str}
    )
    catalog = catalog.astype(
        {"tile": "int", "year": "str", "date": "str", 
         "prediction": "str", "image": "str"}
    )
    mrgi = MergeInstances(
        config=config_data,
        tile_path=tile_geojson, 
        polygon_dir=polygon_dir, 
        merged_polygon_dir=merged_polygon_dir,
        catalog=catalog
    )
    mrgi.create_tile_group()
    mrgi.merge_tiles(num_workers=num_workers) 
    mrgi.compare_stats()
    
    if merged_parquet_file:
        convert_geojson_to_geoparquet(mrgi.merged_polygon_dir,  
                                      merged_parquet_file)
    logging.info("Completed mergeinstances command")

@cli.command()

@click.option('--config', default='configs/example-config.yaml', type=str, 
              help='Path to config file.')
@click.option('--merged-polygon-dir', default=None, type=str, 
              help='Directory for reading merged polygon files.')
@click.option('--attributed-merged-polygon-dir', default=None, type=str, 
              help='Output directory for writing attributed merged polygon files.')
@click.option('--num-workers', default=None, type=int, 
              help='Number of workers for parallelized processes.')
@click.option('--attributed-merged-parquet-file', default=None, type=str, 
              help='Output attributed geoparquet filename.')
@click.option('--area-crs', default='ESRI:102022', type=str, help='CRS for projection.')
@click.option('--log-file', default=None, type=str, help='Log file path.')

def computeinstances(config, merged_polygon_dir, attributed_merged_polygon_dir, 
                     num_workers, attributed_merged_parquet_file, area_crs, log_file):
    """Run ComputeInstances methods (compute shape metrics on merged polygons)."""
    # Load config file
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)

    # Determine log file: CLI arg takes precedence, else config, else None
    log_file = get_config_value(log_file, config_data, 'log_file')
    configure_logging(log_file=log_file)
    logging.info("Running computeinstances command")

    merged_polygon_dir = get_config_value(merged_polygon_dir, config_data, 
                                          'merged_polygon_dir')
    attributed_merged_polygon_dir = get_config_value(attributed_merged_polygon_dir, config_data, 
                                                     'attributed_merged_polygon_dir')
    num_workers = get_config_value(num_workers, config_data, 'num_workers')
    area_crs = get_config_value(area_crs, config_data, 'area_crs')

    attributed_merged_parquet_file = get_config_value(attributed_merged_parquet_file, config_data, 
                                                      'attributed_merged_parquet_file')

    check_dir([attributed_merged_polygon_dir])

    # compute shape metrics
    compute_shape_metrics_parallel(merged_polygon_dir, 
                                   attributed_merged_polygon_dir, 
                                   area_crs, 
                                   num_workers)


    if attributed_merged_parquet_file:
        convert_geojson_to_geoparquet(attributed_merged_polygon_dir,  
                                      attributed_merged_parquet_file)
    logging.info("Completed computeinstances command")

if __name__ == '__main__':
    cli()
