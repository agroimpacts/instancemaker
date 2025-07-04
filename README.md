# Overview

`instacemaker` converts score maps from a segmentation model into vectorized instances of a specific class. It was designed specifically for delineating agricultural fields, and is the final step in the following workflow:

1. Downloading and retiling Planet images. The methods for that are [mostly here](https://github.com/agroimpacts/maputil/tree/main/maputil), but are optimized for NICFI tiles that are no longer available, so now in need of some revision; 2. Making labels by manually delineating fields in the Planet images, using the [labeller](https://github.com/agroimpacts/labeller) platform. See [here](https://github.com/agroimpacts/lacunalabels) and [here](http://arxiv.org/abs/2412.18483) for the dataset and description of the process)
3. Training and finetuning a convolutional neural network (see [Khallaghi et al, 2025](https://www.mdpi.com/2072-4292/17/3/474) for a description of the model),  to recognize three classes (field boundary, field edge, non-field). 
4. Using the model to make predictions in 0.05&deg; tiles of the field interior class, which returns a score map. 
5. Converting those score maps into vectorized instances and then into unified, country-scale field boundary maps, which is the job of `instancemaker`. 

Although developed for this purpose, `instancemaker` should also be applicable to the outputs of other segmentation tasks.

The tools developed in the package are grouped into two classes: `MakeInstances` and `MergeInstances`. The former provides the methods for converting score maps into vectorized instances, including the optional but recommended intermediate step of labeling the score maps and performing erosion and dilation operations to separate narrowly joined instance, and the ability to filter out small objects. `MergeInstances` enables cross-boundary polygon merging (using an approach developed by [Wanjing Li](https://github.com/wanjing-1116)) and the combining the post-merging results into a single geoparquet file.  

Functions can be run from the command line, or from a notebook (see the [demo notebook](notebooks/demo.ipynb)).

## Installation
This was developed and tested using a virtual environment set up with `pyenv` as follows:

```bash
pyenv install 3.13.3 
pyenv virtualenv 3.13.3 instancemaker
pyenv activate instancemaker
python -m pip install --upgrade pip 
```

Then clone and install `instancemaker`:

```bash
git clone git@github.com:agroimpacts/instancemaker.git
cd instancemaker
pip install -e .
```

## Command Line Interface
This is the preferred way to use `instancemaker`, which is designed for large-scale map-making. The CLI provides logging and parallelization. 

From the command line, it is easiest to pass arguments through the configuration file, with the [`example-config.yaml`](configs/example-config.yaml) file as a template. You can also override parameters in the config file with commands.

```bash
% instancemaker --help
Usage: instancemaker [OPTIONS] COMMAND [ARGS]...

  instancemaker CLI with subcommands for makefields and mergefields.

Options:
  --help  Show this message and exit.

Commands:
  makeinstances   Run MakeInstances methods (score map labeling and...
  mergeinstances  Run MergeInstances methods (polygon merging across tile...
```

### `makeinstances`
For converting score maps into vectorized instances, with optional labeling and polygonization. 

```bash
%  instancemaker makeinstances --help
Usage: instancemaker makeinstances [OPTIONS]

  Run MakeInstances methods (score map labeling and polygonization).

Options:
  --config TEXT                   Path to config file.
  --do-labelling BOOLEAN          Whether to label score maps before
                                  polygonizing or not.
  --label-dir TEXT                Label directory.
  --polygon-dir TEXT              Segmentation directory.
  --catalog-file TEXT             CSV catalog file with tiles ID, dates, pred
                                  & image names.
  --pred-dir TEXT                 Directory holding prediction tiles.
  --threshold INTEGER             Threshold value to apply in hardening a
                                  score map.
  --erosion-iterations INTEGER    Erosion iterations.
  --dilation-iterations INTEGER   Dilation iterations.
  --simplify FLOAT                Tolerance value to apply in simplifying
                                  polygons.
  --polygon-filename-template TEXT
                                  Polygon filename template.
  --prediction-filename-template TEXT
                                  Prediction filename template.
  --labeled-filename-template TEXT
                                  Labeled filename template.
  --num-workers INTEGER           Number of parallel workers.
  --log-file TEXT                 Log file path.
  --help                          Show this message and exit.
```

A straightforward example using the example file and the provided demonstration data. Under the current settings in the yaml, this command will first label the score maps, using the specified probability thresholds to harden the images, then polygonize these with simplification, and save the results in the `polygon_dir` specified in the config file. 

```bash
instancemaker makeinstances
```

Or using your own config file
```bash
instancemaker makeinstances --config configs/my-config.yaml
```

### `mergeinstances`
This takes the outputs of `makeinstances` and does cross-boundary polygon merging. The final step (optional) combines the post merged outputs into a single geoparquet file.

```bash
% instancemaker mergeinstances --help
Usage: instancemaker mergeinstances [OPTIONS]

  Run MergeInstances methods (polygon merging across tile boundaries).

Options:
  --config TEXT               Path to config file.
  --tile-geojson TEXT         Name of tile geojson file.
  --polygon-dir TEXT          Segmentation directory.
  --merged-polygon-dir TEXT   Output directory for writing merged polygons.
  --catalog-file TEXT         Catalog file.
  --num-workers INTEGER       Number of workers for parallelized processes.
  --merged-parquet-file TEXT  Output geoparquet filename.
  --log-file TEXT             Log file path.
  --help                      Show this message and exit.
```

After creating a set of polygon files, you can merge polygons that are truncated on either side of boundary. 
```bash
instancemaker mergeinstances
```
