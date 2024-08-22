from itertools import repeat

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import KDTree
from scipy.stats import mode
import pandas as pd

from .utils import get_db_connection


class GridData:
    def __init__(self, lat, lon, values):
        self.lat = lat
        self.lon = lon
        self.values = values if isinstance(values, dict) else {'values': values}

    def __repr__(self):
        return f"""GridData(
    > Axes:
        - lat: {len(self.lat)} elements
        - lon: {len(self.lon)} elements
    > Data:
        """ + '\n    '.join([f"- {key}: {value.shape}" for key, value in self.values.items()]) + """
    > Memory: {:.2f} MB""".format(sum(value.nbytes for value in self.values.values()) / 1024**2) + '\n)'
    
    def __len__(self):
        return len(self.lat), len(self.lon)
    
    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Index should be a tuple of (latitudes, longitudes)")
        
        lat_key, lon_key = key

        # Convert key to indices
        lat_indices = self._get_indices(self.lat, lat_key)
        lon_indices = self._get_indices(self.lon, lon_key)

        # Extract the data using the computed indices
        subset_data = {key: value[np.ix_(lat_indices, lon_indices)] for key, value in self.values.items()}
        
        return GridData(self.lat[lat_indices], self.lon[lon_indices], subset_data)

    @staticmethod
    def _get_indices(axis, key):
        """Helper method to convert slicing and indexing to array indices."""
        if isinstance(key, slice):
            return np.arange(len(axis))[key]
        elif isinstance(key, int):
            return key
        elif hasattr(key, '__iter__'):
            return np.array([np.where(axis == k)[0][0] for k in key])
        else:
            return np.where(axis == key)[0]
        
    def interpolate(self, lat, lon, grid=True):
        new_values = {}
        for key, value in self.values.items():
            Z = np.nan_to_num(value)
            interpolator = RectBivariateSpline(self.lat, self.lon, Z, kx=1, ky=1)
            new_Z = interpolator(lat, lon, grid=grid)
            new_values[key] = new_Z
        if grid:
            return GridData(lat, lon, new_values)
        else:
            return PointData(np.stack([lat, lon], axis=1), new_values)
        
    def interpolate_discrete(self, lat, lon, grid=True):
        old_points = np.array(np.meshgrid(self.lat, self.lon)).T.reshape(-1, 2)

        new_values = {}
        for key, value in self.values.items():
            Z = np.nan_to_num(value).flatten()
            interpolator = KDTree(old_points)
            _, indices = interpolator.query(np.stack([lat, lon], axis=1), k=4)
            new_Z = mode(Z[indices], axis=1).mode
            new_values[key] = new_Z
        
        if grid:
            return GridData(lat, lon, new_values)
        else:
            return PointData(np.stack([lat, lon], axis=1), new_values)

    def to_frame(self):
        index = pd.MultiIndex.from_product([self.lat, self.lon], names=['lat', 'lon'])
        data = np.stack([value.flatten() for value in self.values.values()], axis=1)
        return pd.DataFrame(data, index=index, columns=list(self.values.keys()))


class TimeseriesGridData(GridData):
    def __init__(self, date, lat, lon, values):
        super().__init__(lat, lon, values)
        self.date = np.array(date)

    def __repr__(self):
        return f"""TimeseriesGridData(
    > Axes:
        - date: {len(self.date)} elements
        - lat: {len(self.lat)} elements
        - lon: {len(self.lon)} elements
    > Data:
        """ + '\n    '.join([f"- {key}: {value.shape}" for key, value in self.values.items()]) + """
    > Memory: {:.2f} MB""".format(sum(value.nbytes for value in self.values.values()) / 1024**2) + '\n)'

    def __len__(self):
        return len(self.date), len(self.lat), len(self.lon)
    
    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError("Index should be a tuple of (dates, latitudes, longitudes)")
        
        date_key, lat_key, lon_key = key

        # Convert key to indices
        date_indices = self._get_indices(self.date, date_key)
        lat_indices = self._get_indices(self.lat, lat_key)
        lon_indices = self._get_indices(self.lon, lon_key)

        # Extract the data using the computed indices
        subset_data = {key: value[np.ix_(date_indices, lat_indices, lon_indices)] for key, value in self.values.items()}
        
        return TimeseriesGridData(self.date[date_indices], self.lat[lat_indices], self.lon[lon_indices], subset_data)

    def interpolate(self, lat, lon, grid=True):
        new_values = {}
        for key, value in self.values.items():
            new_value_list = []
            for t in range(len(self.date)):
                Z = np.nan_to_num(value[t])
                interpolator = RectBivariateSpline(self.lat, self.lon, Z, kx=1, ky=1)
                new_Z = interpolator(lat, lon, grid=grid)
                new_value_list.append(new_Z)
            new_values[key] = np.stack(new_value_list, axis=0)
        if grid:
            return TimeseriesGridData(self.date, lat, lon, new_values)
        else:
            return TimeseriesPointData(self.date, np.stack([lat, lon], axis=1), new_values)

    def to_frame(self):
        index = pd.MultiIndex.from_product([self.date, self.lat, self.lon], names=['date', 'lat', 'lon'])
        data = np.stack([value.flatten() for value in self.values.values()], axis=1)
        return pd.DataFrame(data, index=index, columns=list(self.values.keys()))


class PointData:
    def __init__(self, points, values):
        self.points = np.array(points)
        self.values = values if isinstance(values, dict) else {'values': values}

    def __repr__(self):
        return f"""PointData(
    > Axes:
        - points: {len(self.points)} points
    > Data: 
        """ + '\n    '.join([f"- {key}: {value.shape}" for key, value in self.values.items()]) + """
    > Memory: {:.2f} MB""".format(sum(value.nbytes for value in self.values.values()) / 1024**2) + '\n)'

    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise IndexError("Index should be an integer")
        
        point_key = key

        # Extract the data using the computed indices
        subset_data = {key: value[point_key] for key, value in self.values.items()}
        
        return PointData(self.points[point_key], subset_data)
    
    def interpolate(self, points, grid=True):
        NotImplemented

    def to_frame(self):
        index = pd.MultiIndex.from_tuples(self.points.tolist(), names=['lat', 'lon'])
        data = np.stack([value.flatten() for value in self.values.values()], axis=1)
        return pd.DataFrame(data, index=index, columns=list(self.values.keys()))


class TimeseriesPointData(PointData):
    def __init__(self, date, points, values):
        super().__init__(points, values)
        self.date = np.array(date)

    def __repr__(self):
        return f"""TimeseriesPointData(
    > Axes:
        - date: {len(self.date)} elements
        - points: {len(self.points)} points
    > Data:
        """ + '\n    '.join([f"- {key}: {value.shape}" for key, value in self.values.items()]) + """
    > Memory: {:.2f} MB""".format(sum(value.nbytes for value in self.values.values()) / 1024**2) + '\n)'

    def __len__(self):
        return len(self.date), len(self.points)
    
    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Index should be a tuple of (dates, points)")
        
        date_key, point_key = key

        # Convert key to indices
        date_indices = self._get_indices(self.date, date_key)
        point_indices = self._get_indices(self.points, point_key)

        # Extract the data using the computed indices
        subset_data = {key: value[np.ix_(date_indices, point_indices)] for key, value in self.values.items()}
        
        return TimeseriesPointData(self.date[date_indices], self.points[point_indices], subset_data)

    def interpolate(self, points, grid=True):
        NotImplemented

    def to_frame(self):
        index = pd.MultiIndex.from_tuples(
            [(date, lat, lon) for date in self.date for lat, lon in self.points],
            names=['date', 'lat', 'lon'])
        data = np.stack([value.flatten() for value in self.values.values()], axis=1)
        return pd.DataFrame(data, index=index, columns=list(self.values.keys()))


class DatabaseTSPD(TimeseriesPointData):
    def __init__(self, stations, date, points, values):
        super().__init__(date, points, values)
        self.stations = stations

    def save_sql(self, model_version: int, drop_previous: bool = False):
        """Saves output data in the `results` database

        Args:
            model_version (int): A four-digit integer where the first digit 
                                 represents the pollutant type returned by 
                                 the model and the last two digits represent
                                 the version of the model for that pollutant.
        """
        if not (0 <= model_version <= 99999):
            raise ValueError("model_version must be a five-digit integer")
        
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Drop previous rows with that model version, if requested
        #if drop_previous:
        #    for station in self.stations.unique():
        #        cursor.execute(f"DELETE FROM {station} WHERE model_version={model_version}")

        # Insert data for each station
        data_query = """INSERT INTO {} (date, model_version, value) VALUES (%s, %s, %s)"""
        
        #print()
        #for i, station, date, value in zip(range(len(self.stations)), self.stations, self.date, self.values['values']):
        #    print(f"{i+1}/{len(self.stations)}", end="\r")
        #    cursor.execute(data_query.format(station), (date, model_version, value))

        total_stations = len(self.stations.unique())
        for i, station in enumerate(self.stations.unique()):
            print(f"{i+1}/{total_stations}", end="\r")
            cursor.execute(f"DELETE FROM {station} WHERE model_version={model_version}")
            values = [(date, model_version, value) 
                      for date, model_version, value in zip(self.date[self.stations==station], 
                                                            repeat(model_version), 
                                                            self.values['values'][self.stations==station])]
            cursor.executemany(data_query.format(station), values)
            conn.commit()
        print()

        cursor.close()
        conn.commit()
        conn.close()

        return

if __name__ == '__main__':
    import numpy as np
    from dataloader import load_corine

    bbox = {'min_lat': 35.7, 'min_lon': -9.9, 'max_lat': 44.1, 'max_lon': 4.5}

    corine_path = '/data1/data_prep/corine_land_cover/U2018_CLC2018_V2020_20u1_4326.tif'
    land_cover = load_corine(corine_path, bbox)

    lat = np.linspace(35.7, 44.1, 10)
    lon = np.linspace(-9.9, 4.5, 30)

    lat, lon = np.meshgrid(lat, lon)
    lat, lon = lat.flatten(), lon.flatten()

    print(land_cover.interpolate_discrete(lat, lon, grid=False).to_frame())
