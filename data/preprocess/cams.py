from pathlib import Path

import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy as cp

DATA_FOLDER = "/home/urbanaq/data/cams/no2"

def read_netcdfs(dim):
    paths = sorted(Path(DATA_FOLDER).glob("*"))
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined

xds = read_netcdfs("time")
xds.to_netcdf(str(Path(DATA_FOLDER) / "data.nc"))
