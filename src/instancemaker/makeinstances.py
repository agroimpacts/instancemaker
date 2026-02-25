import numpy as np
import xarray as xr
import rioxarray as rxr
from scipy import ndimage as ndi
from skimage.morphology import erosion, dilation, remove_small_objects
import geopandas as gpd
from shapely.geometry import shape
from rasterio import features

class MakeInstances:
    """
    MakeInstances Class
    ====================

    This class provides various methods for image processing, including 
    erosion, dilation, segmentation, and labeling. It can be used with 
    raw imagery to segment an image, followed by masking with 
    model predictions. Alternatively, it can be used to simply convert 
    semantic predictions to polygons. 

    Attributes
    ----------
    kernel : ndarray
        The structuring element used for morphological operations.
    threshold : int
        The threshold value for labeling predictions.
    erosion_iterations : int
        The number of iterations for the erosion operation.
    dilation_iterations : int
        The number of iterations for the dilation operation.

    Methods
    -------
    __init__(kernel, threshold=50, erosion_iterations=4, dilation_iterations=3)
        Initializes the MakeInstances with the given parameters.
    make_labeled_prediction(prediction_filename, labeled_filename)
        Labels the prediction and saves the labeled prediction to a file.
    label_prediction(prediction, threshold, kernel, erosion_iterations, 
                     dilation_iterations, min_size=21)
        Labels individual objects in CNN score maps after performing erosion 
        and dilation.
    polygonize(image, simplify, labeled=False, erode_dilate=True)
        Converts raster data into vector polygons.
    crop_image(tile_path, prediction_path)
        Crops the image to match the prediction.
    multi_erosion(image, kernel, iterations)
        Performs multiple erosion operations on an image.
    multi_dilation(image, kernel, iterations)
        Performs multiple dilation operations on an image.
    mask_segment(segmentation, labeled_prediction)
        Masks out instances using the score map.
    """

    def __init__(self, kernel, threshold=None, erosion_iterations=None, 
                 dilation_iterations=None, threshold_type="score", class_id=1):
        """
        Initialize the MakeInstances with the given parameters.

        Parameters
        ----------
        kernel : ndarray
            Numpy array providing erosion and dilation kernel.
        threshold : int, optional
            Classification threshold (default is 50).
        erosion_iterations : int, optional
            Number of erosion iterations (default is 4).
        dilation_iterations : int, optional
            Number of dilation iterations (default is 3).
        """
        self.threshold_type = threshold_type
        self.class_id = class_id
        
        self.kernel = np.array(kernel)
        self.threshold = threshold
        self.erosion_iterations = erosion_iterations
        self.dilation_iterations = dilation_iterations

    def make_labeled_prediction(self, prediction_filename,labeled_filename=None, 
                                threshold=None, erode_dilate=True, min_size=21):
        """
        Label prediction and optionally save the labeled prediction to a file 
        or return it as a rioxarray object.
        
        Parameters
        ----------
        prediction_filename : str
            Path to the prediction image.
        labeled_filename : str, optional
            Path to save the labeled prediction. If None, returns the labeled 
            prediction as a rioxarray object.
        threshold : int, optional
            Threshold to override the instance default threshold.
        erode_dilate : bool, optional
            If True, applies erosion and dilation (default is True).
        min_size : int, optional
            Minimum size of objects to retain (default is 0).
        
        Returns
        -------
        xarray.DataArray or None
            Labeled prediction as a rioxarray object if `labeled_filename` is 
            None. Otherwise, returns None.
        """
        try:
            prediction_image = rxr.open_rasterio(
                prediction_filename, masked=True
            )
            prediction = prediction_image.squeeze().values  # Extract 2D array

            # Use the provided threshold if specified, otherwise use the instance's
            # threshold
            threshold = threshold if threshold is not None else self.threshold

            # Validate threshold
            if threshold is None:
                raise ValueError(
                    f"Threshold must be provided or set during initialization. "
                    f"Problematic file: {prediction_filename}"
                )

            # Labeling prediction
            labeled_prediction = self.label_prediction(
                prediction, threshold, self.kernel, self.erosion_iterations, 
                self.dilation_iterations, min_size=min_size, 
                erode_dilate=erode_dilate
            )

            # Create labeled prediction as a DataArray
            labeled_da = xr.DataArray(
                labeled_prediction['labels'],
                dims=("y", "x"),
                coords={
                    "y": prediction_image.y,
                    "x": prediction_image.x,
                })
            
            # Save to disk if labeled_filename is provided
            if labeled_filename:
                labeled_da.rio.to_raster(
                    labeled_filename, driver="COG", compress="DEFLATE"
                )
            
            return labeled_da

        except Exception as e:
            raise e
        
    def label_prediction(self, prediction, threshold, kernel, 
                         erosion_iterations, dilation_iterations, 
                         min_size, erode_dilate=True):
        """
        Label individual objects in CNN score maps with optional erosion and 
        dilation.

        Parameters
        ----------
        prediction : ndarray 
            Score map read into a numpy array. 
        threshold : int 
            Classification threshold (e.g., 50).
        kernel : ndarray
            Numpy array providing erosion and dilation kernel.
        erosion_iterations : int
            Number of erosion iterations.
        dilation_iterations : int
            Number of dilation iterations.
        min_size : int, optional
            Minimum size of objects to retain (default is 0).
        erode_dilate : bool, optional
            If True, applies erosion and dilation (default is True).

        Returns
        -------
        dict
            Dictionary with labeled predictions and the number of segments. 
        """

        # classify score map
        if self.threshold_type == "class":
            classified_image = (prediction == self.class_id)
        else:
            classified_image = prediction > threshold

        if erode_dilate:
            # morphology operations
            eroded_image = self.multi_erosion(
                classified_image, kernel, erosion_iterations
            )
            processed_image = self.multi_dilation(
                eroded_image, kernel, dilation_iterations
            )
        else:
            processed_image = classified_image

        # Label
        labeled_prediction = ndi.label(processed_image)[0]

        # Filter
        if min_size > 0:
            labeled_prediction = remove_small_objects(
                labeled_prediction, min_size
            )

        # labeled_prediction = self.reshape_image(labeled_prediction)

        return {'labels': labeled_prediction,
                'count': len(np.unique(labeled_prediction))}
        
    def polygonize(self, image, simplify, labeled=False, erode_dilate=True, 
                   polygon_filename=None, threshold=None):
        """
        Convert raster data into vector polygons.

        Parameters
        ----------
        image : str or xarray.DataArray
            Input image to polygonize. Can be a file path or xarray.DataArray.
        simplify : float or None
            Tolerance for polygon simplification. If None, no simplification.
        labeled : bool, optional
            If True, treats the image as labeled data (default is False).
        erode_dilate : bool, optional
            If True, applies erosion and dilation (default is True).
        polygon_filename : str, optional
            Path to save the output filename.
        threshold : int, optional
            Threshold to override the instance's default threshold 
            (default is None).

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing the polygonized features.
        """
        if isinstance(image, str):
            image = rxr.open_rasterio(image, masked=True).squeeze()
    
        # Use provided threshold or default
        threshold = threshold if threshold is not None else self.threshold
    
        if labeled:
            shapes = features.shapes(
                image.values,
                transform=image.rio.transform()
            )
            polygons = [shape(geom) for geom, val in shapes if val > 0]
    
        else:
            # -------------------------------
            # CLASS MODE vs SCORE MODE
            # -------------------------------
            if self.threshold_type == "class":
                # Keep only selected class (e.g., interior class 1)
                mask = image.values == self.class_id
            else:
                # Original score threshold behavior
                mask = image.values > threshold
    
            # Morphological refinement (optional)
            if erode_dilate:
                mask = self.multi_erosion(
                    mask, self.kernel, self.erosion_iterations
                )
                mask = self.multi_dilation(
                    mask, self.kernel, self.dilation_iterations
                )
    
            shapes = features.shapes(
                mask.astype("uint8"),
                transform=image.rio.transform()
            )
            polygons = [shape(geom) for geom, val in shapes if val == 1]
    
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=image.rio.crs)
    
        if simplify:
            gdf["geometry"] = gdf["geometry"].simplify(
                tolerance=simplify,
                preserve_topology=True
            )
    
        if polygon_filename:
            gdf.to_file(polygon_filename, driver="GeoJSON")

        return gdf

    def crop_image(self, tile_path, prediction_path):
        """
        Crop an image to match the prediction.

        Parameters
        ----------
        tile_path : str
            Path to the tile image.
        prediction_path : str
            Path to the prediction image.

        Returns
        -------
        tuple
            Tuple containing cropped image and metadata for cropped image.
        """
        try:
            tile_image = rxr.open_rasterio(tile_path, masked=True)
            prediction_image = rxr.open_rasterio(prediction_path, masked=True)
        except Exception as e:
            return None, e

        # Unpack the bounds of the prediction image
        minx, miny, maxx, maxy = prediction_image.rio.bounds()
        cropped_image = tile_image.rio.clip_box(minx=minx, miny=miny, 
                                                maxx=maxx, maxy=maxy)

        return cropped_image, cropped_image.rio.to_dict()

    def multi_erosion(self, image, kernel, iterations):
        """
        Perform multiple erosion operations on an image.
        
        Parameters
        ----------
        image : ndarray
            Image to erode.
        kernel : ndarray
            Numpy array providing the erosion kernel.
        iterations : int
            Number of erosion iterations.
            
        Returns
        -------
        ndarray
            Eroded image.
        """
        for _ in range(iterations):
            image = erosion(image, kernel)
        return image

    def multi_dilation(self, image, kernel, iterations):
        """
        Perform multiple dilation operations on an image.
        
        Parameters
        ----------
        image : ndarray
            Image to dilate.
        kernel : ndarray
            Numpy array providing the dilation kernel.
        iterations : int
            Number of dilation iterations.
            
        Returns
        -------
        ndarray
            Dilated image.
        """
        for _ in range(iterations):
            image = dilation(image, kernel)
        return image

    def mask_segment(self, segmentation, labeled_prediction):
        """
        Mask out SNIC instances using the score map.
        
        Parameters
        ----------
        segmentation : ndarray
            Segmentation array.
        labeled_prediction : ndarray
            Labeled prediction array.

        Returns
        -------
        ndarray
            Masked segmentation array.
        """
        segmentation_arr = np.array(segmentation)
        masked_segment = np.where(labeled_prediction == 0, np.nan, 
                                  segmentation_arr)
        return masked_segment
