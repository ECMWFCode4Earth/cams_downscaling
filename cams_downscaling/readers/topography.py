import numpy as np
import rasterio
from rasterio.windows import from_bounds

from ..datatypes import GridData


def load_topography(topography_path, bbox):
    with rasterio.open(topography_path) as src:
        # Create a window from the bbox
        window = from_bounds(bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat'], src.transform)
        
        # Read the data within the window
        data = src.read(1, window=window)
        
        # Update the metadata to reflect the windowed read
        transform = src.window_transform(window)

        # Generate arrays of latitudes and longitudes
        rows, cols = np.indices(data.shape)
        lons, lats = rasterio.transform.xy(transform, rows, cols)
        lats, lons = np.array(lats), np.array(lons)

        # Sort latitudes and longitudes in increasing order and rearrange data accordingly
        lat_indices = np.argsort(lats[:, 0])
        lon_indices = np.argsort(lons[0, :])
        data = data[lat_indices, :][:, lon_indices]

        return GridData(
            lats[lat_indices, 0],
            lons[0, lon_indices],
            {'elevation': np.where(data == src.nodata, 0, data)})