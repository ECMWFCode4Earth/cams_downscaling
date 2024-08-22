from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cp

DATA_FOLDER = "/home/urbanaq/data/era5/2023"

paths = [f for f in sorted(Path(DATA_FOLDER).glob("**/*")) if (f.is_file() and not 'idx' in str(f))]

def read_gribs(dim: str, drop_vars: list[str]) -> xr.Dataset:
    datasets = []
    for file in paths:
        print(f'Processing file {file}')
        dataset = xr.open_dataset(file, engine="cfgrib", drop_variables=drop_vars)
        datasets.append(dataset)
    combined = xr.concat(datasets, dim)

    return combined

drop_vars = [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'surface_net_solar_radiation', 'total_precipitation',
                'standard_deviation_of_orography'
            ]

xds = read_gribs('time', drop_vars)
xds.to_netcdf(str(Path(DATA_FOLDER) / "era5.nc"))