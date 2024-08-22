import numpy as np
import xarray as xr
from PIL import Image

region = "IBERIA"
variable = "no2"
file = f"/home/urbanaq/data/cams/no2/no2_{region}.nc"
output_folder = f"/home/urbanaq/data/cams/no2/{region}"

xds = xr.open_dataset(file)
# Save coordinates as separate .tiff files
Image.fromarray((xds['lat'].values).astype(np.float32)).save(f"{output_folder}/lat.tiff")
Image.fromarray((xds['lon'].values).astype(np.float32)).save(f"{output_folder}/lon.tiff")

# Save each time step as a separate .tiff file
for time in xds.time:
    data = xds[variable].sel(time=time).values
    filename = time.values.astype(str).replace(":", "-").replace("T", "_").split(".")[0]
    Image.fromarray((data).astype(np.float32)).save(f"{output_folder}/{filename}.tiff")
