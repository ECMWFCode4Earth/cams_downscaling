""" 
Reads all the grib files downloaded from CDSAPI and saves the selected region in a single netcdf file.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import xarray as xr

from cams_downscaling.utils import read_config


region = "poland"

config = read_config('/home/urbanaq/cams_downscaling/config')
region_bbox = config["regions"][region]["bbox"]

DATA_FOLDER = Path("/home/urbanaq/data/era5_land/2024")

paths = [f for f in sorted(DATA_FOLDER.glob("**/*.grib")) if (f.is_file() and not 'idx' in str(f))]

def read_gribs(dim: str, drop_vars: list[str]=[], region_bbox: dict[str, float]={}) -> xr.Dataset:
    datasets = []
    for file in paths:
        print(f'Processing file {file}')
        dataset = xr.load_dataset(file, engine="cfgrib", drop_variables=drop_vars)

        if region_bbox:
            mask_lon = (dataset.longitude >= region_bbox["min_lon"]) & (dataset.longitude <= region_bbox["max_lon"])
            mask_lat = (dataset.latitude >= region_bbox["min_lat"]) & (dataset.latitude <= region_bbox["max_lat"])

            dataset = dataset.where(mask_lon & mask_lat, drop=True)
            
        datasets.append(dataset)
    combined = xr.concat(datasets, dim)

    return combined


xds = read_gribs('time', [], region_bbox)
xds.to_netcdf(str(DATA_FOLDER / f"era5_land_{region}.nc"))
