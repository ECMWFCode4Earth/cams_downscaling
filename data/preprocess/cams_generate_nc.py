""" 
Generate a single netCDF file from multiple netcdf files downloaded from CDS API (you must untar them first).

The data folder should only contain one variable
"""
from pathlib import Path

import xarray as xr


DATA_FOLDER = "/home/urbanaq/data/cams/no2"

def read_netcdfs(dim):
    paths = sorted(Path(DATA_FOLDER).glob("*.nc"))
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined

xds = read_netcdfs("time")
xds.to_netcdf(str(Path(DATA_FOLDER) / "data.nc"))
