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
from instancemaker.utils import get_filename

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
        self.config = config
        self.tile_path = tile_path
        self.polygon_dir = polygon_dir
        self.merged_polygon_dir = merged_polygon_dir
        self.tiles = gpd.read_file(self.tile_path)
        self.tiles = add_tile_col_row_from_geometry(self.tiles)

        self.tile_selection = catalog
        self.tile_selection = self.tile_selection.merge(self.tiles, on='tile', 
                                                        how='left')
    
    def create_tile_group(self):
        """
        Group tiles into non-adjacent groups based on row and column indices. 
        This grouping is used to parallelize processing and avoid adjacent tile 
        conflicts.

        Returns
        -------
        None
        """
        print("Creating tile group")
        # Filter tile DataFrame to just existing tiles
        self.tile_mapping = {
            (row['tile_row'], row['tile_col']): 
            {
                'tilename': row['tile'],
                'geometry': self.tiles[self.tiles['tile'] == row['tile']]\
                    .iloc[0]['geometry']
            }
            for _, row in self.tile_selection.iterrows()
        }

        # Group tiles using non-adjacent grouping rule
        tile_positions = set(self.tile_mapping.keys())
        self.color_groups = {}
        for row, col in tile_positions:
            color = (row % 3, col % 2)
            self.color_groups.setdefault(color, []).append((row, col))
                    
        # Print how many tiles are in each group and list the tiles
        for color_key, tile_list in self.color_groups.items():
            print(f"Group {color_key}: {len(tile_list)} tiles")
            print("Tiles:", tile_list)
        
        print("Grouping tiles done")

        print()

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
        self.tile_selection = pd.read_csv(selected_tile_csv_path)

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
        if not hasattr(self, 'color_groups'):
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
            plt.savefig(f"{outname}.png")
        else:
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
        print("Cloning polygon files from {} to {}"\
              .format(self.polygon_dir, self.merged_polygon_dir))
        os.makedirs(self.merged_polygon_dir, exist_ok=True)
        for fname in os.listdir(self.polygon_dir):
            if fname.endswith(".geojson"):
                src = os.path.join(self.polygon_dir, fname)
                dst = os.path.join(self.merged_polygon_dir, fname)
                shutil.copy2(src, dst)
        print("Cloning polygon files done")
        print()

        print("Merging tiles with {} workers".format(num_workers))
        # Run in parallel with batching
        results = {}
        # controls how many tiles to process at the same time in each batch
        # Loop through each color group
        for color, tiles in self.color_groups.items():
            print(f"Processing color group: {color} with {len(tiles)} tiles.")
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
                        except Exception as exc:
                            print(f"Tile {tile} generated an exception: {exc}")
                        pbar.update(1)

        print("Merging tiles done") 
        print()
        
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
        adjacent_tiles = []
        for i in [-1, 0, 1]:
            for j in [0, 1]:
                neighbor = (target_tile[0] + i, target_tile[1] + j)
                if neighbor in self.tile_mapping:
                    adjacent_tiles.append(self.tile_mapping[neighbor])

        retrieved_polygons = []
        for tile_info in adjacent_tiles:
            tilename = tile_info['tilename']
            row = self.tile_selection[self.tile_selection['tile']\
                                      .isin([tilename])].iloc[0]
            filename = get_filename(self.config['polygon_filename_template'], 
                                    row['tile'], row['year'], row['date'])
            filepath = os.path.join(self.merged_polygon_dir, filename)
            if os.path.exists(filepath):
                retrieved_polygons.append(gpd.read_file(filepath))
            else:
                print(f"Tile {tilename} not found in {filepath}")

        if not retrieved_polygons:
            return False

        all_combine = pd.concat(retrieved_polygons, 
                                ignore_index=True).unary_union

        polygons = list(all_combine.geoms)

        tile_polygons = {tile['tilename']: [] for tile in adjacent_tiles}
        tile_geometries = {tile['tilename']: tile['geometry'] 
                           for tile in adjacent_tiles}
        assigned_count = 0
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
                
        #  Save output per tile
        for tilename, geoms in tile_polygons.items():
            out_gdf = gpd.GeoDataFrame({'geometry': geoms}, crs="EPSG:4326")

            row = self.tile_selection[self.tile_selection['tile']\
                                      .isin([tilename])].iloc[0]
            filename = get_filename(self.config['polygon_filename_template'], 
                                    row['tile'], row['year'], row['date'])
            filepath = os.path.join(self.merged_polygon_dir, filename)
            out_gdf.to_file(filepath, driver="GeoJSON")

        return True
    
    def compare_stats(self):
        """
        Print statistics comparing the number of polygons before and after 
        merging.

        Returns
        -------
        None
        """
        original_files = glob.glob(os.path.join(self.polygon_dir, "*.geojson"))
        original_polygons = pd.concat(
            [gpd.read_file(fp) for fp in original_files], ignore_index=True
        )

        merged_files = glob.glob(
            os.path.join(self.merged_polygon_dir, "*.geojson")
        )
        merged_polygons = pd.concat([gpd.read_file(fp) for fp in merged_files], 
                                    ignore_index=True)

        print(f"Number of polygons before merging: {len(original_polygons)}")
        print(f"Number of polygons after merging: {len(merged_polygons)}")

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
    return tiles