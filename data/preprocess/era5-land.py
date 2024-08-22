from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cp

REGIONS = {
    "IBERIA": {"min_lat": 35.7,
               "min_lon": -9.9,
               "max_lat": 44.1,
               "max_lon": 4.5}
}

DATA_FOLDER = "/home/urbanaq/data/era5_land"

paths = [f for f in sorted(Path(DATA_FOLDER).glob("**/*")) if (f.is_file() and not 'idx' in str(f))]

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

for region in REGIONS:
    xds = read_gribs('time', region_bbox=REGIONS[region])
    xds.to_netcdf(str(Path(DATA_FOLDER) / f"era5_land_{region}.nc"))
