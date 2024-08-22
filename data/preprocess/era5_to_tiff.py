import os

import numpy as np
import xarray as xr
from PIL import Image


region = "iberia"
file = f"/home/urbanaq/data/era5/2023/era5_{region.upper()}.nc"
output_folder = "/data1/data_prep/era5_old/{variable}/{region}"

variables = ["iews", "inss", "blh"]

xds = xr.open_dataset(file)


for v in variables:
    print(f"Processing variable {v}")
    var_folder = output_folder.format(variable=v, region=region)

    os.makedirs(var_folder, exist_ok=True)

    # Save coordinates as separate .tiff files
    Image.fromarray((xds['latitude'].values).astype(np.float32)).save(f"{var_folder}/lat.tiff")
    Image.fromarray((xds['longitude'].values).astype(np.float32)).save(f"{var_folder}/lon.tiff")

    # Save each time step as a separate tiff file
    for time in xds.time:
        data = xds[v].sel(time=time).values
        filename = time.values.astype(str).replace(":", "-").replace("T", "_").split(".")[0]
        Image.fromarray((data).astype(np.float32)).save(f"{var_folder}/{filename}.tiff")
