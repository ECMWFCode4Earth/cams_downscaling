from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ..datatypes import TimeseriesGridData

def load_era5_land(era5_land_path: Path, region: str, dates: list[datetime] = []) -> TimeseriesGridData:
    vars_era5_land = ['d2m', 'ssr', 't2m', 'tp', 'v10', 'u10']
    data_dict = {}

    lat = np.array(Image.open(str(era5_land_path / "lat" / f"{region}.tiff")))
    lon = np.array(Image.open(str(era5_land_path / "lon" / f"{region}.tiff")))

    # Load each variable
    for i, var in enumerate(vars_era5_land):
        path_var = era5_land_path / var / region

        # Load each time step from separate TIFF files
        data_arrays = []
        for time in dates:
            filename = time.strftime("%Y-%m-%d_%H-%M-%S")
            tiff_file = str(path_var / f"{filename}.tiff")
            
            data_tiff = Image.open(tiff_file)
            data = np.array(data_tiff)
            data_arrays.append(data)
        data_dict[var] = np.stack(data_arrays, axis=0)
        
    return TimeseriesGridData(dates, lat, lon, data_dict)