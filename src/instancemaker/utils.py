import os
import re
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import geopandas as gpd
import leafmap.leafmap as leafmap
import pandas as pd
import glob

def convert_geojson_to_geoparquet(input_dir, output_file):
    """
    Convert all GeoJSON files in a directory to a single GeoParquet file.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing GeoJSON files
    output_file : str
        Path for the output GeoParquet file
    """
    # Get all GeoJSON files
    geojson_files = glob.glob(f"{input_dir}/*.geojson")
    
    # Read and combine all GeoJSON files
    gdfs = []
    for file in geojson_files:
        gdf = gpd.read_file(file)
        # Add source file information
        # gdf['source_file'] = Path(file).stem
        gdfs.append(gdf)
    
    # Combine all GeoDataFrames
    combined_gdf = pd.concat(gdfs, ignore_index=True)
    
    # Save as GeoParquet
    combined_gdf.to_parquet(output_file, index=False)
    
    print(f"Converted {len(geojson_files)} files to {output_file}")
    print(f"Total features: {len(combined_gdf)}")

def get_filename(name_template, tile, year, date, suffix=None):
    """
    Get image file name from template.

    Parameters
    ----------
    name_template : str
        File name template.
    tile : int
        Tile number.
    year : int
        Year of the image.
    date : str
        Date of image. This can be just a month, a month-day, or other formats.
    suffix : str, optional
        Suffix to add to the file name.

    Returns
    -------
    str
        File name.
    """
    file_name = re.sub('<tile>', f'{tile}', name_template)
    file_name = re.sub('<year>', f'{year}', file_name)
    file_name = re.sub('<date>', f'{date}', file_name)

    if suffix:
        fname = os.path.join(suffix, file_name)
    
    return file_name

def check_dir(dirs):
    """
    Check if output directory exists, and create it if it does not.

    Parameters
    ----------
    dirs : list or str
        List of directories or a single directory to check.

    Returns
    -------
    None
    """

    if not isinstance(dirs, list):
        dirs = [dirs]

    for dir in dirs:
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

def get_date(row):
    """
    Extract the date from the file name.

    Parameters
    ----------
    row : str
        File name.

    Returns
    -------
    str
        Date of the image.
    """
    result = re.search(
        r".*tile\d{6}_(\w{4}-\w{2})?_?(\w{4}-\w{2})_.*_cog\.tif", row
    )

    if result.group(1):
        date = f'{result.group(1)}_{result.group(2)}'
    else:
        date = result.group(2)
    
    return date

def quick_plot(image, figsize=(4, 3), title='Title', cmap=plt.cm.gray):
    """
    Plot an image quickly.

    Parameters
    ----------
    image : ndarray
        Image to plot.
    figsize : tuple, optional
        Size of the figure (default is (4, 3)).
    title : str, optional
        Title of the plot (default is 'Title').
    cmap : Colormap, optional
        Colormap to use (default is plt.cm.gray).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    plt.show()

def min_max(file, bands, clip=None):
    """
    Calculate the minimum/maximum values for specified bands in a raster file.

    Parameters
    ----------
    file : str
        Path to the raster file.
    bands : tuple
        Tuple of band indices to process.
    clip : float, optional
        Percentile for clipping (default is None).

    Returns
    -------
    tuple
        Two lists containing the minimum and maximum values for each band.
    """
    values = rxr.open_rasterio(file).isel(band=list(bands)).values
    mins = []
    maxs = []
    
    for i in range(values.shape[0]):
        if clip:
            mins.append(np.nanpercentile(values[i], clip))
            maxs.append(np.nanpercentile(values[i], 100-clip))
        else: 
            mins.append(np.nanmin(values[i]))
            maxs.append(np.nanmax(values[i]))
           
    return mins, maxs

def view_instance(instance, image, bands, stretch=True, clip=1.5, width=12, 
                  height=5, zoom=14): 
    """
    Display a comparison of selected polygons and its corresponding image.

    Parameters
    ----------
    instance : str or geopandas.GeoDataFrame
        Path to the polygon (instance) file or a GeoDataFrame object.
    image : str or xarray.DataArray
        Path to the image file or an xarray.DataArray object.
    bands : list
        List of band indices (in rgb order, base 0) to use for the image.
    stretch : bool, optional
        Whether to apply a stretch to the image (default is True).
    clip : float, optional
        Percentile for clipping (default is 1.5).
    width : int, optional
        Width of the plot (default is 12).
    height : int, optional
        Height of the plot (default is 5).
    zoom : int, optional
        Zoom level for the map (default is 14).

    Returns
    -------
    leafmap.Map
        A Leafmap viewer or matplotlib-based plots.
    """
    # Load instance as GeoDataFrame if it's a file path
    if isinstance(instance, str):
        gdf = gpd.read_file(instance)
    elif isinstance(instance, gpd.GeoDataFrame):
        gdf = instance
    else:
        raise ValueError("Instance must be a file path or a GeoDataFrame.")

    # Load image as xarray.DataArray if it's a file path
    if isinstance(image, str):
        r = rxr.open_rasterio(image)
    elif isinstance(image, rxr.DataArray):
        r = image
    else:
        raise ValueError("Image must be a file path or an xarray.DataArray.")

    # Calculate bounds and center
    bb = np.array(r.rio.bounds())
    yx = [np.mean(bb[[1, 3]]), np.mean(bb[[0, 2]])]

    # Initialize Leafmap
    m = leafmap.Map(zoom=zoom, center=yx)
    m.add_basemap("SATELLITE")

    # Prepare bands for visualization
    rbands = list((np.array(bands) + 1).astype(int))
    if stretch:
        mins, maxs = min_max(image if isinstance(image, str) 
                             else image.rio.to_raster(), bands, clip)
        m.add_raster(image if isinstance(image, str) 
                     else image.rio.to_raster(), indexes=rbands, 
                     layer_name='Tile', zoom_to_layer=False, vmin=mins, 
                     vmax=maxs)
    else: 
        m.add_raster(image if isinstance(image, str) 
                     else image.rio.to_raster(), bands=rbands, 
                     layer_name='Tile', zoom_to_layer=False)
    
    # Add GeoDataFrame to the map
    m.add_gdf(gdf, layer_name="Polygons", zoom_to_layer=False)
    
    return m