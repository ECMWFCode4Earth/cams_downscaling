import os

import numpy as np
import xarray as xr
from PIL import Image


region = "iberia"
file = f"/home/urbanaq/data/era5_land/era5_land_{region.upper()}.nc"
output_folder = "/data1/data_prep/era5_land/{variable}/{region}"

variables = ["u10", "v10", "t2m", "ssr", "tp", "d2m"]
variables = ["d2m"]

xds = xr.open_dataset(file)


for v in variables:
    print(f"Processing variable {v}")
    var_folder = output_folder.format(variable=v, region=region)

    os.makedirs(var_folder, exist_ok=True)

    # Save coordinates as separate .tiff files
    Image.fromarray((xds['latitude'].values).astype(np.float32)).save(f"{var_folder}/lat.tiff")
    Image.fromarray((xds['longitude'].values).astype(np.float32)).save(f"{var_folder}/lon.tiff")

    # Save each time step as a separate tiff file
    for start_date in xds.time:

        for i, step in enumerate(xds[v].sel(time=start_date.values).step):
            filename = (start_date + step).values.astype(str).replace(":", "-").replace("T", "_").split(".")[0]

            data = xds[v].sel(time=start_date, step=step).values
            if len(data) == 2:
                data = data[0]
            Image.fromarray((data).astype(np.float32)).save(f"{var_folder}/{filename}.tiff")
