from pathlib import Path

import xarray as xr


REGIONS = {
    "IBERIA": {"min_lat": 35.7,
               "min_lon": -9.9,
               "max_lat": 44.1,
               "max_lon": 4.5}
}

file = "/home/urbanaq/data/era5/2023/era5.nc"
region = "IBERIA"
region_bbox = REGIONS[region]

dataset = xr.load_dataset(file)

mask_lon = (dataset.longitude >= region_bbox["min_lon"]) & (dataset.longitude <= region_bbox["max_lon"])
mask_lat = (dataset.latitude >= region_bbox["min_lat"]) & (dataset.latitude <= region_bbox["max_lat"])

dataset = dataset.where(mask_lon & mask_lat, drop=True)
dataset.to_netcdf(str(Path("/home/urbanaq/data/era5/2023") / f"era5_{region}.nc"))
