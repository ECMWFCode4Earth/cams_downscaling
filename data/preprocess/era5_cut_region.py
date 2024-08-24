""" 
This script reads the netcdf generated by `era5_generate_nc.py` and cuts the region of interest
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import xarray as xr

from cams_downscaling.utils import read_config


file = "/home/urbanaq/data/era5/era5.nc"
region = "poland"


config = read_config('/home/urbanaq/cams_downscaling/config')
region_bbox = config["regions"][region]["bbox"]

dataset = xr.load_dataset(file)

mask_lon = (dataset.longitude >= region_bbox["min_lon"]) & (dataset.longitude <= region_bbox["max_lon"])
mask_lat = (dataset.latitude >= region_bbox["min_lat"]) & (dataset.latitude <= region_bbox["max_lat"])

dataset = dataset.where(mask_lon & mask_lat, drop=True)
dataset.to_netcdf(str(Path("/home/urbanaq/data/era5") / f"era5_{region}.nc"))
