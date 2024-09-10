"""
Reads all the GRIBS downloaded from the ADS and generates a single NetCDF file for the selected region
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import xarray as xr

from cams_downscaling.utils import read_config


DATA_FOLDER = "/home/urbanaq/data/cams/2024"
region = "poland"

paths = [f for f in sorted(Path(DATA_FOLDER).glob("**/*.grib")) if (f.is_file() and not 'idx' in str(f))]

config = read_config('/home/urbanaq/cams_downscaling/config')
region_bbox = config["regions"][region]["bbox"]

def read_gribs(dim: str, drop_vars: list[str]) -> xr.Dataset:
    datasets = []
    for file in paths:
        print(f'Processing file {file}')
        dataset = xr.open_dataset(file, engine="cfgrib", drop_variables=drop_vars)
        mask_lon = (dataset.longitude >= region_bbox["min_lon"]) & (dataset.longitude <= region_bbox["max_lon"])
        mask_lat = (dataset.latitude >= region_bbox["min_lat"]) & (dataset.latitude <= region_bbox["max_lat"])
        dataset = dataset.where(mask_lon & mask_lat, drop=True)
        datasets.append(dataset)
    combined = xr.concat(datasets, dim)

    return combined

drop_vars = []

xds = read_gribs('time', drop_vars)
xds.to_netcdf(str(Path(DATA_FOLDER) / f"cams_{region}.nc"))