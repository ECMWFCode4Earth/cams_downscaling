import numpy as np
import rasterio
from pyproj import Transformer

from ..datatypes import GridData

def load_height(height_path, bbox):
    with rasterio.open(height_path) as src:
        # Forward transformation from EPSG:4326 to the source CRS
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        min_lon, min_lat = transformer.transform(bbox["min_lon"], bbox["min_lat"])
        max_lon, max_lat = transformer.transform(bbox["max_lon"], bbox["max_lat"])

        # Create a window from the bbox
        window = src.window(min_lon, min_lat, max_lon, max_lat)

        # Read the data within the window
        data = src.read(1, window=window)
        
        # Update the metadata to reflect the windowed read
        transform = src.window_transform(window)

        # Generate arrays of pixel coordinates
        rows, cols = np.indices(data.shape)
        lons, lats = rasterio.transform.xy(transform, rows, cols)
        lats, lons = np.array(lats), np.array(lons)

        # Inverse transformation from the source CRS back to EPSG:4326
        inverse_transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lons, lats = inverse_transformer.transform(lons, lats)

        # Sort latitudes and longitudes in increasing order and rearrange data accordingly
        lat_indices = np.argsort(lats[:, 0])
        lon_indices = np.argsort(lons[0, :])
        data = data[lat_indices, :][:, lon_indices]

        return GridData(
            lats[lat_indices, 0],
            lons[0, lon_indices],
            {'height': np.where(data == src.nodata, 0, data)})