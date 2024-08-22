import numpy as np
from PIL import Image
from datetime import datetime

from ..datatypes import TimeseriesGridData

def load_era5_land(era5_land_path: str, dates: list[datetime] = []) -> TimeseriesGridData:
    vars_era5_land = ['d2m', 'ssr', 't2m', 'tp', 'v10', 'u10']
    data_dict = {}

    lat = np.array(Image.open(f"{era5_land_path}/lat.tiff"))
    lon = np.array(Image.open(f"{era5_land_path}/lon.tiff"))

    # Load each variable
    for i, var in enumerate(vars_era5_land):
        path_var = f"{era5_land_path}/{var}/iberia"

        # Load each time step from separate TIFF files
        data_arrays = []
        for time in dates:
            filename = time.strftime("%Y-%m-%d_%H-%M-%S")
            tiff_file = f"{path_var}/{filename}.tiff"
            
            data_tiff = Image.open(tiff_file)
            data = np.array(data_tiff)
            data_arrays.append(data)
        data_dict[var] = np.stack(data_arrays, axis=0)
        
    return TimeseriesGridData(dates, lat, lon, data_dict)