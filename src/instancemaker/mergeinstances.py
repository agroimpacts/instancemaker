import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, Point
from tqdm import tqdm
import glob
from itertools import islice
import concurrent.futures
import matplotlib.patches as patches
import shutil
import numpy as np
from instancemaker.utils import get_filename, analyze_filename_mismatch
import logging

class MergeInstances:
    """
    MergeInstances Class
    ===================

    This class manages the cross-boundary merging of polygon tiles. It groups
    polygon files into non-adjacent groups based on their spatial relationships,
    and merges polygons across tile boundaries.

    Attributes
    ----------
    tile_path : str
        Path to the tile GeoJSON file.
    polygon_dir : str
        Directory containing the input polygon files.
    merged_polygon_dir : str
        Directory to store merged polygon files.
    tiles : GeoDataFrame
        GeoDataFrame of tiles with row/col info.
    tile_selection : DataFrame
        DataFrame of selected tiles for processing.
    """

    def __init__(self, config, tile_path, polygon_dir, merged_polygon_dir, 
                 catalog):
        """
        Initialize MergeInstances.

        Parameters
        ----------
        tile_path : str
            Path to the tile GeoJSON file.
        polygon_dir : str
            Directory containing the input polygon files.
        merged_polygon_dir : str
            Directory to store merged polygon files.
        selected_tile_csv_path : str, optional
            Path to CSV of selected tiles.
        """
        logging.info("Initializing MergeInstances")
        self.config = config
        self.tile_path = tile_path
        self.polygon_dir = polygon_dir
        self.merged_polygon_dir = merged_polygon_dir
        
        logging.info(f"Loading tiles from: {tile_path}")
        try:
            self.tiles = gpd.read_file(self.tile_path)
            logging.info(f"Loaded {len(self.tiles)} tiles")
        except Exception as e:
            logging.error(f"Failed to load tiles from {tile_path}: {e}")
            raise
        
        # Check if tiles were loaded successfully
        if self.tiles.empty:
            raise ValueError(f"No tiles found in {tile_path}")
        
        # Remove tile_col and tile_row columns if they exist
        if 'tile_col' in self.tiles.columns:
            self.tiles = self.tiles.drop('tile_col', axis=1)
        if 'tile_row' in self.tiles.columns:
            self.tiles = self.tiles.drop('tile_row', axis=1)
        
        logging.debug("Adding tile row/column information")
        self.tiles = add_tile_col_row_from_geometry(self.tiles)
        logging.debug("Tile row/column information added")

        self.tile_selection = catalog
        logging.debug("Merging catalog with tile information")
        self.tile_selection = self.tile_selection.merge(self.tiles, on='tile', 
                                                        how='left')
        
        # Check if merge was successful
        if self.tile_selection.empty:
            raise ValueError("No matching tiles found between catalog and tile file")
        
        logging.info(f"Initialized with {len(self.tile_selection)} selected tiles")
    
    def create_tile_group(self):
        """
        Group tiles into non-adjacent groups based on row and column indices. 
        This grouping is used to parallelize processing and avoid adjacent tile 
        conflicts.

        Returns
        -------
        None
        """
        logging.info("Creating tile groups")
        # Filter tile DataFrame to just existing tiles
        logging.debug("Creating tile mapping")
        
        # Check if tile_selection has required columns
        required_columns = ['tile_row', 'tile_col', 'tile']
        missing_columns = [col for col in required_columns if col not in self.tile_selection.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in tile_selection: {missing_columns}")
        
        # Check for invalid tile_row or tile_col values
        invalid_tiles = self.tile_selection[
            self.tile_selection['tile_row'].isna() | 
            self.tile_selection['tile_col'].isna() |
            self.tile_selection['tile_row'].isnull() | 
            self.tile_selection['tile_col'].isnull()
        ]
        if not invalid_tiles.empty:
            invalid_tile_ids = invalid_tiles['tile'].tolist()
            raise ValueError(f"Found tiles with invalid row/col values: {invalid_tile_ids}")
        
        self.tile_mapping = {}
        for _, row in self.tile_selection.iterrows():
            tile_key = (row['tile_row'], row['tile_col'])
            tile_geometry = self.tiles[self.tiles['tile'] == row['tile']]
            
            # Get the first (and should be only) geometry for this tile
            geometry_row = tile_geometry.iloc[0]
            self.tile_mapping[tile_key] = {
                'tilename': row['tile'],
                'geometry': geometry_row['geometry']
            }
        
        logging.debug(f"Created tile mapping for {len(self.tile_mapping)} tiles")

        # Group tiles using non-adjacent grouping rule
        logging.debug("Applying non-adjacent grouping rule")
        tile_positions = set(self.tile_mapping.keys())
        self.color_groups = {}
        for row, col in tile_positions:
            color = (row % 3, col % 2)
            self.color_groups.setdefault(color, []).append((row, col))
                    
        # Print how many tiles are in each group and list the tiles
        logging.info("Tile groups created:")
        for color_key, tile_list in self.color_groups.items():
            logging.info(f"Group {color_key}: {len(tile_list)} tiles")
            logging.debug(f"Tiles in group {color_key}: {tile_list}")
        
        logging.info("Tile grouping completed")

    def reset_tile_selection(self, selected_tile_csv_path=None):
        """
        Reset the tile selection from a CSV file.

        Parameters
        ----------
        selected_tile_csv_path : str
            Path to CSV of selected tiles.

        Returns
        -------
        None
        """
        logging.info(f"Resetting tile selection from: {selected_tile_csv_path}")
        try:
            self.tile_selection = pd.read_csv(selected_tile_csv_path)
            logging.info("Tile selection reset successfully")
        except Exception as e:
            logging.error(f"Failed to reset tile selection: {e}")
            raise

    def visualize_and_save_tile_groups(self, outname=None):
        """
        Visualize the tile groups and optionally save the plot as an image.

        Parameters
        ----------
        outname : str, optional
            If provided, saves the plot to this filename.

        Returns
        -------
        None
        """
        logging.info("Visualizing tile groups")
        if not hasattr(self, 'color_groups'):
            logging.debug("Creating tile groups for visualization")
            self.create_tile_group()
        
        group_colors = {group: f"C{i}" 
                        for i, group in enumerate(self.color_groups.keys())}
        
        fig, ax = plt.subplots(figsize=(4, 4))

        for group, tiles_ in self.color_groups.items():
            color = group_colors[group]
            for row, col in tiles_:
                rect = patches.Rectangle((col, row), 1, 1, facecolor=color, 
                                         edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
        ax.set_xlim(min(self.tile_selection.tile_col)-1, 
                    max(self.tile_selection.tile_col)+2)
        ax.set_ylim(min(self.tile_selection.tile_row)-1, 
                    max(self.tile_selection.tile_row)+2)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("Tile Grid Colored by Non-Adjacent Groups")
        plt.xlabel("Tile Column")
        plt.ylabel("Tile Row")
        if outname:
            logging.info(f"Saving tile group visualization to: {outname}.png")
            plt.savefig(f"{outname}.png")
        else:
            logging.debug("Displaying tile group visualization")
            plt.show()

    def merge_tiles(self, num_workers=8):
        """
        Merge polygons from tiles in parallel, grouped by non-adjacent groups.

        Parameters
        ----------
        num_workers : int, optional
            Number of parallel workers to use (default is 8).

        Returns
        -------
        None
        """
        logging.info(f"Starting tile merging with {num_workers} workers")
        
        # Clone polygon files
        logging.info(f"Cloning polygon files from {self.polygon_dir} to {self.merged_polygon_dir}")
        try:
            os.makedirs(self.merged_polygon_dir, exist_ok=True)
            logging.debug(f"Created merged polygon directory: {self.merged_polygon_dir}")
            
            geojson_files = [f for f in os.listdir(self.polygon_dir) if f.endswith(".geojson")]
            logging.info(f"Found {len(geojson_files)} GeoJSON files to clone")
            
            for fname in geojson_files:
                src = os.path.join(self.polygon_dir, fname)
                dst = os.path.join(self.merged_polygon_dir, fname)
                shutil.copy2(src, dst)
                logging.debug(f"Cloned: {fname}")
            
            logging.info("Polygon file cloning completed")
        except Exception as e:
            logging.error(f"Failed to clone polygon files: {e}")
            raise

        # Run in parallel with batching
        logging.info("Starting parallel tile processing")
        results = {}
        # controls how many tiles to process at the same time in each batch
        # Loop through each color group
        for color, tiles in self.color_groups.items():
            logging.info(f"Processing color group: {color} with {len(tiles)} tiles")
            total_tiles = len(tiles)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)\
                as executor:
                # Launch all tasks for this color group concurrently.
                future_to_tile = {
                    executor.submit(self.process_tile_at, (row, col)):(row, col) 
                    for row, col in tiles
                }
                with tqdm(total=total_tiles, desc=f"Processing group {color}") \
                    as pbar:
                    for future in \
                        concurrent.futures.as_completed(future_to_tile):
                        tile = future_to_tile[future]
                        try:
                            results[tile] = future.result()
                            logging.debug(f"Completed processing tile {tile}")
                        except Exception as exc:
                            logging.error(f"Tile {tile} generated an exception: {exc}")
                        pbar.update(1)

        logging.info("Tile merging completed") 
        
    def process_tile_at(self, target_tile):
        """
        Process merging for a single tile by combining polygons from adjacent 
        tiles, assigning polygons to the correct tile based on spatial 
        relationships.

        Parameters
        ----------
        target_tile : tuple
            (row, col) index of the target tile.

        Returns
        -------
        bool
            True if processing was successful, False otherwise.
        """
        logging.debug(f"Processing tile at position: {target_tile}")
        
        # Find adjacent tiles
        adjacent_tiles = []
        for i in [-1, 0, 1]:
            for j in [0, 1]:
                neighbor = (target_tile[0] + i, target_tile[1] + j)
                if neighbor in self.tile_mapping:
                    adjacent_tiles.append(self.tile_mapping[neighbor])
        
        logging.debug(f"Found {len(adjacent_tiles)} adjacent tiles")

        # Retrieve polygons from adjacent tiles
        retrieved_polygons = []
        for tile_info in adjacent_tiles:
            tilename = tile_info['tilename']
            row = self.tile_selection[self.tile_selection['tile']\
                                      .isin([tilename])].iloc[0]
            filename = get_filename(self.config['polygon_filename_template'], 
                                    row['tile'], row['year'], row['date'])
            filepath = os.path.join(self.merged_polygon_dir, filename)
            
            if os.path.exists(filepath):
                # Check if filename matches the expected pattern
                expected_filename = get_filename(self.config['polygon_filename_template'], 
                                               row['tile'], row['year'], row['date'])
                if filename != expected_filename:
                    error_msg = analyze_filename_mismatch(filename, self.config['polygon_filename_template'], 
                                                        row['tile'], row['year'], row['date'])
                    raise ValueError(error_msg)
                
                try:
                    gdf = gpd.read_file(filepath)
                    retrieved_polygons.append(gdf)
                    logging.debug(f"Loaded {len(gdf)} polygons from tile {tilename}")
                except Exception as e:
                    logging.warning(f"Failed to load polygons from {filepath}: {e}")
            else:
                logging.warning(f"Tile {tilename} not found in {filepath}")

        if not retrieved_polygons:
            logging.warning(f"No polygons found for tile {target_tile}")
            return False

        # Combine all polygons
        logging.debug("Combining polygons from adjacent tiles")
        all_combine = pd.concat(retrieved_polygons, 
                                ignore_index=True).unary_union
        polygons = list(all_combine.geoms)
        logging.debug(f"Combined into {len(polygons)} polygons")

        # Assign polygons to tiles
        tile_polygons = {tile['tilename']: [] for tile in adjacent_tiles}
        tile_geometries = {tile['tilename']: tile['geometry'] 
                           for tile in adjacent_tiles}
        assigned_count = 0
        
        logging.debug("Assigning polygons to tiles")
        for polygon in polygons:
            left_point = find_leftmost_coordinate(polygon)
            is_assigned = False
            for tilename, tile_geom in tile_geometries.items():
                if tile_geom.contains(left_point) or \
                    tile_geom.touches(left_point):
                    tile_polygons[tilename].append(polygon)
                    assigned_count += 1
                    is_assigned = True
                    break
            if not is_assigned: # Find tile closest to unassigned point
                min_distance = float('inf')
                closest_tile = None
                for tilename, tile_geom in tile_geometries.items():
                    distance = tile_geom.distance(left_point)
                    if distance < min_distance:
                        min_distance = distance
                        closest_tile = tilename
                tile_polygons[closest_tile].append(polygon)
                assigned_count += 1
        
        logging.debug(f"Assigned {assigned_count} polygons to tiles")
                
        # Save output per tile
        logging.debug("Saving merged polygons to files")
        for tilename, geoms in tile_polygons.items():
            if geoms:  # Only save if there are polygons
                out_gdf = gpd.GeoDataFrame({'geometry': geoms}, crs="EPSG:4326")

                row = self.tile_selection[self.tile_selection['tile']\
                                          .isin([tilename])].iloc[0]
                filename = get_filename(self.config['polygon_filename_template'], 
                                        row['tile'], row['year'], row['date'])
                filepath = os.path.join(self.merged_polygon_dir, filename)
                
                try:
                    out_gdf.to_file(filepath, driver="GeoJSON")
                    logging.debug(f"Saved {len(geoms)} polygons to {filename}")
                except Exception as e:
                    logging.error(f"Failed to save polygons for tile {tilename}: {e}")

        logging.debug(f"Completed processing tile {target_tile}")
        return True
    
    def compare_stats(self):
        """
        Print statistics comparing the number of polygons before and after 
        merging.

        Returns
        -------
        None
        """
        logging.info("Comparing statistics before and after merging")
        
        try:
            original_files = glob.glob(os.path.join(self.polygon_dir, "*.geojson"))
            logging.debug(f"Found {len(original_files)} original GeoJSON files")
            
            # Validate original file names against template pattern
            for fp in original_files:
                filename = os.path.basename(fp)
                # Extract tile, year, date from filename for validation
                # This is a simplified check - you might want more robust parsing
                if not filename.endswith('.geojson'):
                    raise ValueError(f"Original file '{filename}' does not have expected .geojson extension")
            
            original_polygons = pd.concat(
                [gpd.read_file(fp) for fp in original_files], ignore_index=True
            )
            logging.info(f"Number of polygons before merging: {len(original_polygons)}")

            merged_files = glob.glob(
                os.path.join(self.merged_polygon_dir, "*.geojson")
            )
            logging.debug(f"Found {len(merged_files)} merged GeoJSON files")
            
            # Validate merged file names against template pattern
            for fp in merged_files:
                filename = os.path.basename(fp)
                # Check if filename matches the expected pattern
                # Extract tile, year, date from catalog for comparison
                for _, row in self.tile_selection.iterrows():
                    expected_filename = get_filename(self.config['polygon_filename_template'], 
                                                   row['tile'], row['year'], row['date'])
                    if filename == expected_filename:
                        break
                else:
                    # If no matching pattern found, raise error
                    error_msg = f"Merged file '{filename}' does not match any expected pattern from config template '{self.config['polygon_filename_template']}'\n"
                    error_msg += f"Expected patterns for available tiles:\n"
                    for _, row in self.tile_selection.iterrows():
                        expected = get_filename(self.config['polygon_filename_template'], row['tile'], row['year'], row['date'])
                        error_msg += f"  - {expected}\n"
                    raise ValueError(error_msg)
            
            merged_polygons = pd.concat([gpd.read_file(fp) for fp in merged_files], 
                                        ignore_index=True)
            logging.info(f"Number of polygons after merging: {len(merged_polygons)}")
            
            reduction = len(original_polygons) - len(merged_polygons)
            reduction_percent = (reduction / len(original_polygons)) * 100 if len(original_polygons) > 0 else 0
            logging.info(f"Polygon reduction: {reduction} ({reduction_percent:.1f}%)")
            
        except Exception as e:
            logging.error(f"Failed to compare statistics: {e}")
            raise

def find_leftmost_coordinate(polygon):
    """
    Find the leftmost coordinate of a polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The input polygon.

    Returns
    -------
    shapely.geometry.Point
        The leftmost point of the polygon.
    """
    x, y = polygon.exterior.coords.xy
    coords = list(zip(x, y))
    min_x = float('inf')
    point = None
    for coord in coords:
        if coord[0] < min_x:
            min_x = coord[0]
            point = coord
    return Point(point)

import numpy as np

def add_tile_col_row_from_geometry(tiles):
    """
    Add row and column indices to a GeoDataFrame of tiles based on geometry.

    Parameters
    ----------
    tiles : geopandas.GeoDataFrame
        GeoDataFrame containing tile geometries.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with added 'tile_row' and 'tile_col' columns.
    """
    logging.debug("Adding tile row/column information from geometry")
    centroids = tiles.geometry.centroid
    xs = centroids.x
    ys = centroids.y

    left_col = min(xs)
    top_row = max(ys)
    right_col = max(xs)
    bottom_row = min(ys)
    width = (tiles.geometry.area ** 0.5).mean()
    tile_col = np.arange(left_col, right_col, width)
    tile_row = np.arange(top_row, bottom_row, -width)
    centroid_rowcol = []
    # for i in range(len(tile_col)):
    #     for j in range(len(tile_row)):
    #         centroid_rowcol.append(((tile_row[j], tile_col[i]), i,j))
    for i in range(len(tile_row)):
        for j in range(len(tile_col)):
            centroid_rowcol.append(((tile_row[i], tile_col[j]), i, j))

    centroid_rowcol = pd.DataFrame(centroid_rowcol, 
                                   columns=['centroid', 'tile_row', 'tile_col'])
    centroid_rowcol['geometry'] = centroid_rowcol['centroid']\
        .apply(lambda x: Point(x[1], x[0]))
    centroid_rowcol = gpd.GeoDataFrame(centroid_rowcol, geometry='geometry')
    centroid_rowcol.drop(columns=['centroid'], inplace=True)

    tiles = gpd.sjoin_nearest(
        tiles,
        centroid_rowcol[['tile_row', 'tile_col', 'geometry']],
        how='left',
        distance_col='dist' # optional: gives distance to the matched point
    )

    tiles = tiles.drop(columns=['index_right', 'dist'])
    logging.debug("Tile row/column information added successfully")
    return tiles