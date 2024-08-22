import numpy as np
from PIL import Image
from datetime import datetime

from ..datatypes import TimeseriesGridData

def load_era5(era5_path: str, dates: list[datetime] = []) -> TimeseriesGridData:
    vars_era5 = ['blh', 'iews', 'inss']
    data_dict = {}

    lat = np.array(Image.open(f"{era5_path}/lat.tiff"))
    lon = np.array(Image.open(f"{era5_path}/lon.tiff"))

    # Load each variable
    for i, var in enumerate(vars_era5):
        path_var = f"{era5_path}/{var}/iberia"

        # Load each time step from separate TIFF files
        data_arrays = []
        for time in dates:
            filename = time.strftime("%Y-%m-%d_%H-%M-%S")
            tiff_file = f"{path_var}/{filename}.tiff"
            
            data_tiff = Image.open(tiff_file)
            data = np.array(data_tiff)
            
            # Reorder data according to sorted latitudes and longitudes
            data_arrays.append(data)
        
        data_dict[var] = np.stack(data_arrays, axis=0)
        
    return TimeseriesGridData(dates, lat, lon, data_dict)
