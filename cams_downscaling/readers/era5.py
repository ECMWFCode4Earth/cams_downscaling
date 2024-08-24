import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path

from ..datatypes import TimeseriesGridData


def load_era5(era5_path: Path, region: str, dates: list[datetime] = []) -> TimeseriesGridData:
    vars_era5 = ['blh', 'iews', 'inss']
    data_dict = {}

    lat = np.array(Image.open(str(era5_path / "lat" / f"{region}.tiff")))
    lon = np.array(Image.open(str(era5_path / "lon" / f"{region}.tiff")))

    # Load each variable
    for i, var in enumerate(vars_era5):
        path_var = era5_path / var / region

        # Load each time step from separate TIFF files
        data_arrays = []
        for time in dates:
            filename = time.strftime("%Y-%m-%d_%H-%M-%S")
            tiff_file = str(path_var / f"{filename}.tiff")
            
            data_tiff = Image.open(tiff_file)
            data = np.array(data_tiff)
            
            # Reorder data according to sorted latitudes and longitudes
            data_arrays.append(data)
        
        data_dict[var] = np.stack(data_arrays, axis=0)
        
    return TimeseriesGridData(dates, lat, lon, data_dict)
