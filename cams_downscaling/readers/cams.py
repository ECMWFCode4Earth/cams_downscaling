from pathlib import Path 

import numpy as np
from PIL import Image

from ..datatypes import TimeseriesGridData


def load_cams(cams_path: Path, dates=[]):
    # Load coordinates from TIFF files
    lat = np.array(Image.open(f"{cams_path}/lat.tiff"))
    lon = np.array(Image.open(f"{cams_path}/lon.tiff"))

    # Load each time step from separate TIFF files
    data_arrays = []
    for time in dates:
        filename = time.strftime("%Y-%m-%d_%H-%M-%S")
        tiff_file = str(cams_path / f"{filename}.tiff")
        
        data_tiff = Image.open(tiff_file)
        data = np.array(data_tiff)
        data_arrays.append(data)

    return TimeseriesGridData(dates, lat, lon, {'no2': np.stack(data_arrays, axis=0)})