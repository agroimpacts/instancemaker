from setuptools import setup, find_packages

setup(
    name="instancemaker",
    version="0.2.0",
    description="For labeling and polygonizing semantic predictions of field " \
    "boundaries",
    author="Lyndon Estes, Nguyen Ha, and Wanjing Li",
    author_email="lestes@clarku.edu",
    url="https://github.com/agroimpacts/instancemaker",    
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy",
        "xarray",
        "rioxarray",
        "scipy",
        "shapely",
        "scikit-image",
        "geopandas",
        "matplotlib", 
        "pyyaml", 
        "leafmap",
        "localtileserver",
        "jupyterlab"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],      entry_points={
        'console_scripts': [
            'instancemaker = instancemaker.cli:cli',
        ],
    },
    python_requires=">=3.7",
)
