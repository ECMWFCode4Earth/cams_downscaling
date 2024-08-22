import xarray as xr
import numpy as np
from scipy.interpolate import griddata


NEW_GRID_RESOLUTION = 0.001175  # Resolution in degrees for the new grid


REGIONS = {
    "IBERIA": {"min_lat": 35.7,
               "min_lon": -9.9,
               "max_lat": 44.1,
               "max_lon": 4.5
               },

    "BARCELONA": {
                "min_lat": 41.2,
                "min_lon": 1.9,
                "max_lat": 41.8,
                "max_lon": 2.3
                },
    "MADRID": {
                "min_lat": 40.2,
                "min_lon": -3.95,
                "max_lat": 40.6,
                "max_lon": -3.4
                },
    "LISBON": {
                "min_lat": 35.5,
                "min_lon": -9.5,
                "max_lat": 3.9,
                "max_lon": -8.9
                }
}

region = "BARCELONA"

# Step 1: Load the NetCDF file
input_file = '/home/urbanaq/data/cams/no2/no2_IBERIA.nc'
dataset = xr.open_dataset(input_file, chunks={"time": 1})
variable = "no2"

print("Data loaded successfully")

# Step 2: Define the new grid with 0.001° resolution
lat_new = np.arange(REGIONS[region]["min_lat"], REGIONS[region]["max_lat"], NEW_GRID_RESOLUTION)
lon_new = np.arange(REGIONS[region]["min_lon"], REGIONS[region]["max_lon"], NEW_GRID_RESOLUTION)

print("New grid defined")

dataset = dataset.interp(lat=lat_new, lon=lon_new)


print("Data interpolated successfully")

output_file = '/data1/resampled_data/iberia/cams_no2.nc'
dataset.to_netcdf(output_file)

print(f"Interpolated data saved to {output_file}")
