import time
from pathlib import Path
import uuid
import shutil

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree
from joblib import load
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from decimal import Decimal
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from .datatypes import DatabaseTSPD
from .readers.topography import load_topography
from .readers.land_cover import load_corine
from .readers.population import load_pop
from .readers.build_height import load_height
from .readers.roads import load_osm
from .readers.cams import load_cams
from .readers.era5 import load_era5
from .readers.era5_land import load_era5_land
from .readers import eea
from .utils import read_config, get_db_connection

config = read_config('/home/urbanaq/cams_downscaling/config')
DATA_PATH = Path(config['paths']['stations'])

# VERSION NUMBERING:
# 1st number: 1 for time-based split, 2 for station-based split
split_method = -1
# 2nd - 3rd number: combination of datasets
# datasets_to_run = [str(i).zfill(2) for i in range(0,11)] # 2nd - 3rd number
datasets_to_run = ['00']
# 4th number: 0 for 2022 and 2023, 1 for 2022, 2 for 2023
year_code = 0
# 5th number: None for Iberia, 1 for Italy, 2 for Poland
region_code = None

pollutant_code = 1

# DON'T change it, just add new combinations as needed
datasets_combinations = {
    '00': ['topography','land_use'], 
    '01': ['topography','land_use','era5_land','era5'],
    '02': ['topography','land_use','population'],
    '03': ['topography','land_use','height'],
    '04': ['topography','land_use','era5_land','era5','population','height'],
    '05': [],
    '06': ['topography'],
    '07': ['land_use'],
    '08': ['topography','land_use', 'roads'], 
    '09': ['topography','land_use','era5_land','era5','population','height','roads'],
    '10': ['roads']
}

# OPTIONS FOR CLUSTER MAPPING
hours_to_map = [dt.datetime(2022, 1, 1, hour) for hour in range(0,24)] # monday 10th january 2022

# bbox = { "min_lat": 41.3135,
#          "min_lon": 1.9232,
#          "max_lat": 41.5612,
#          "max_lon": 2.222
#         } # Barcelona

cluster = 101
resolution = 1000 # Resolution of the map in meters

# AUTOMATIC GENERATION OF VERSIONS BASED ON ABOVE
model_versions = {
    f"{split_method}{combination}{year_code}{region_code if region_code else ''}": datasets_combinations[combination] for combination in datasets_to_run
}

if not region_code:
    region = 'iberia'
    countries = ["Spain", "Portugal"]
elif region_code == 1:
    region = 'italy'
    countries = ["Italy"]
elif region_code == 2:
    region = 'poland'
    countries = ["Poland"]
else:
    raise ValueError('Countries not defined for the region selected')

years = {
    0: [2022, 2023],
    1: [2022],
    2: [2023]
}[year_code]

variable = config['pollutants']['pollutants'][pollutant_code]

sources = {'topography': 'topography',
           'land_cover': 'corine',
           'population': 'pop',
           'height': 'height',
           'osm': 'osm',
           'cams': 'cams',
           'era5': 'era5',
           'era5_land': 'era5_land'}

def eea_stations(countries) -> tuple[pd.DataFrame, pd.DataFrame]:
    print('Loading stations and observations from EEA')

    stations = eea.load_stations(countries=countries,
                                 bbox=config['regions'][region]['bbox'],
                                 data_path=Path(config['paths']['stations']))

    stations, observations = eea.load_eea_data(stations=stations,
                                               pollutant=variable,
                                               years=years,
                                               countries=countries,
                                               data_path=Path(config['paths']['stations']))

    stations = eea.get_clusters(stations) # type: ignore
    for cluster in stations.cluster.unique():
        # Check how many rows are in each cluster and delete those with less than 2 rows
        if stations[stations.cluster == cluster].shape[0] < 2:
            stations = stations[stations.cluster != cluster]
    observations = observations.join(stations['cluster'], on='station', how='inner') # type: ignore
    observations = eea.fill_missing_values(observations, variable=variable)

    return stations, observations

def interpolate_points(x, variable, lats, lons):
    x = x.reset_index()[[
        'Latitude', 'Longitude'] +
        [variable]]
    
    # Separate points and observations
    all_points = x.values

    lats, lons = np.meshgrid(lats, lons)
    coords = np.vstack([lats.ravel(), lons.ravel()]).T

    tree = KDTree(all_points[:, :2])
    dist, idx = tree.query(coords)

    interpolated_results = np.hstack([
        coords,
        all_points[idx][:, -1:],
        dist[:, np.newaxis]])

    # Combine all results into a single DataFrame
    return pd.DataFrame(
        interpolated_results,
        columns=[
            'Latitude', 'Longitude'] +
            [f'{variable}_interp', 'dist'])

def save_results(dataset, X_test, y_pred, model_version):
    print(f'Saving results for model version {model_version}')
    stations=dataset.station
    date=dataset.time
    points=dataset[['Latitude', 'Longitude']]

    db = DatabaseTSPD(stations, date, points, values=y_pred)

    db.save_sql(model_version=int(model_version), drop_previous=True)

def lats_lons(global_bbox, resolution):
    # Get the bounding box for the cluster. Resolution is in meters
    resolution = resolution / 111000
    lats = np.arange(global_bbox["min_lat"], global_bbox["max_lat"], resolution)
    lons = np.arange(global_bbox["min_lon"], global_bbox["max_lon"], resolution)
    return lats, lons

def prepare_dataset(data, external_variables):
    data['hour'] = pd.to_datetime(data['time']).dt.hour
    data['day'] = pd.to_datetime(data['time']).dt.day
    data['month'] = pd.to_datetime(data['time']).dt.month
    data['year'] = pd.to_datetime(data['time']).dt.year

    features = [
        'cluster', 'Latitude', 'Longitude'] + external_variables + [
        'NO2_cams', 'NO2_interp', 'dist', 'hour', 'day', 'month', 'year']

    X = data[features]

    return X

def chunk_bbox(lat, lon, lat_chunk, lon_chunk):
    for i in range(0, len(lat), lat_chunk):
        for j in range(0, len(lon), lon_chunk):
            # Define the current box
            lat_box = lat[i:i+lat_chunk]
            lon_box = lon[j:j+lon_chunk]
            
            # Include incomplete boxes
            yield lat_box, lon_box

def find_cluster_bbox(cluster: int) -> tuple[float, float, float, float]:
    # from the database get data from table stations

    conn = get_db_connection()
    cursor = conn.cursor()

    query = f"""SELECT lat, lon FROM stations WHERE cluster = {cluster}"""
    cursor.execute(query)
    rows = cursor.fetchall()

    lats = [float(row[0]) if type(row[0])==Decimal else float(row[0].decode('utf-8')) for row in rows]
    lons = [float(row[1]) if type(row[1])==Decimal else float(row[1].decode('utf-8')) for row in rows]

    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    conn.close()
    return min_lat, max_lat, min_lon, max_lon

def get_stations_data(grid_lat, grid_lon, hour, model_version):
    conn = get_db_connection()
    cursor = conn.cursor()

    min_lat, max_lat = grid_lat[0], grid_lat[-1]
    min_lon, max_lon = grid_lon[0], grid_lon[-1]

    query = """
    SELECT station_id, lat, lon
    FROM stations
    WHERE lat BETWEEN %s AND %s
    AND lon BETWEEN %s AND %s
    """
    cursor.execute(query, (min_lat, max_lat, min_lon, max_lon))
    stations = cursor.fetchall()

    cursor.close()
    conn.close()

    data = []

    csv_stations = pd.read_csv(DATA_PATH / f"{hour.year}" / f"ES.csv")
    for station_id, lat, lon in stations:
        data_station = csv_stations[csv_stations['station'] == station_id.decode('utf-8')]
        if data_station.empty:
            continue
        value = data_station[variable].values[0]
        data.append([station_id.decode('utf-8'), lat.decode('utf-8'), lon, value])
        

    df = pd.DataFrame(data, columns=['station_id', 'lat', 'lon', 'value'])

    return df

def plot_map(y_inference, hour: dt.datetime, resolution, version, grid_lat, grid_lon):
    data = y_inference[pd.to_datetime(y_inference.time)==hour].drop(columns='time')
    data.set_index(['Latitude', 'Longitude'], inplace=True)

    min_lat, max_lat, min_lon, max_lon = find_cluster_bbox(cluster)
    bbox = {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon
    }
    extent = (min_lon, max_lon, min_lat, max_lat)

    # Define the extent
    # extent = (bbox["min_lon"], bbox["max_lon"], bbox["min_lat"], bbox["max_lat"])

    # Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution="10m")  # type: ignore

    # Set the extent of the map
    ax.set_extent(extent, crs=ccrs.PlateCarree()) # type: ignore

    # Plot the data
    im = ax.imshow(data.NO2_pred.values.reshape(len(grid_lat), len(grid_lon)), extent=extent, transform=ccrs.PlateCarree(), cmap=plt.cm.jet, origin='lower') # type: ignore

    # Add stations scatter to the plot
    stations = get_stations_data(grid_lat, grid_lon, hour, version)
    stations = stations.dropna()
    stations['lon'] = stations['lon'].astype(float)
    stations['lat'] = stations['lat'].astype(float)
    # sc = ax.scatter(list(stations['lon']), stations['lat'], color='red', s=100, edgecolor='black', transform=ccrs.PlateCarree())  # No color mapping for testing
    
    sc = ax.scatter(stations['lon'], stations['lat'], c=stations['value'], s=100, cmap=plt.cm.jet, edgecolor='black', transform=ccrs.PlateCarree(), vmin=data.NO2_pred.min(), vmax=data.NO2_pred.max())  # type: ignore

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
    cbar.set_label('NO2 (µg/m³)')

    # Set the title
    plt.title(f"Downscaled NO2 concentration in {config['places']['cluster_names'][cluster]}\n"
          + hour.strftime('%Y-%m-%d %H:%M:%S') + f'\n Model version: {version} - Grid spacing: {resolution} m')
    
    # Add lats and lons to the axis
    ax.set_xticks(grid_lon[::40], crs=ccrs.PlateCarree())
    ax.set_yticks(grid_lat[::20], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True, transform_precision=0.01)
    lat_formatter = LatitudeFormatter(transform_precision=0.01)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # Create the output path
    output_path = Path(config['paths']['maps'], config['places']['cluster_names'][cluster], version,
                       f"{version}_{config['places']['cluster_names'][cluster]}_{hour.strftime('%Y%m%d_%H')}_{resolution}m.png")

    # Ensure the output directory and its partents exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)

def main():

    start = time.time() 

    # Create tmp folder with results
    tmp_folder = Path("/tmp", str(uuid.uuid4()).replace('-', '_'))
    tmp_folder.mkdir(parents=True, exist_ok=False)
    print(f"Created temporary folder {tmp_folder}")

    stations, observations = eea_stations(countries)

    global_lat, global_lon = lats_lons(bbox, resolution)
    total_chunks = 0

    print(f"It took {(time.time() - start)/60} minutes to load stations and observations")
    start = time.time()
    print ("---- Starting inference ----")
    for chunk_i, (grid_lat, grid_lon) in enumerate(chunk_bbox(global_lat, global_lon, 1000, 1000)):
        print()
        print(f"Processing chunk {chunk_i}")
        print()

        total_chunks += 1

        min_lat, max_lat = grid_lat[0], grid_lat[-1]
        min_lon, max_lon = grid_lon[0], grid_lon[-1]

        # Load static datasets for this chunk
        topography = (
            load_topography(
                config['paths']['topography'],
                {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon})
            .interpolate(
                lat=grid_lat,
                lon=grid_lon).to_frame())

        land_use = (
            load_corine(
                config['paths']['land_cover'],
                {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon})
            .interpolate_discrete(
                lat=grid_lat,
                lon=grid_lon).to_frame())

        population = (
            load_pop(
                config['paths']['population'],
                {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon})
            .interpolate(
                lat=grid_lat,
                lon=grid_lon).to_frame())

        height = (
            load_height(
                config['paths']['height'],
                {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon})
            .interpolate(
                lat=grid_lat,
                lon=grid_lon).to_frame())

        roads = (
            load_osm(
                f"{config['paths']['osm']}/iberia.tif",
                {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon})
            .interpolate_discrete(
                lat=grid_lat,
                lon=grid_lon).to_frame())
        print("It took ", (time.time() - start)/60, " minutes to load all static datasets")
        start = time.time()

        for hour in hours_to_map:
            print(f"Processing hour {hour}")

            dataset = observations[observations['cluster']==cluster].loc[pd.IndexSlice[hour, :], :]
            # dataset = observations.loc[pd.IndexSlice[hour, :], :]

            dataset = dataset.groupby(['time', 'cluster']).apply(
                interpolate_points, # type: ignore
                variable=variable,
                lats=grid_lat,
                lons=grid_lon
            ).reset_index().drop(columns=['level_2']).set_index(['time', 'Latitude', 'Longitude']).sort_index() # type: ignore
            print("EEA interpolated")

            # Model Datasets
            cams = (
                load_cams(
                    Path(config['paths']['cams'], variable.lower(), region), 
                    dates=[hour])
                .interpolate(
                    lat=grid_lat,
                    lon=grid_lon,
                    grid=True).to_frame())
            cams.index = dataset.index
            cams.columns = [v.upper() for v in cams.columns]
            print("CAMS interpolated")

            print(f"It took {(time.time() - start)/60} minutes to load the time-based datasets")
            start = time.time()

            for version in model_versions.keys():
                print(f'Running model version {version}')
                dataset_version = dataset.copy()
                
                for source in model_versions[version]:
                    if source in ['era5', 'era5_land']:
                        dataset_version = dataset_version.join(locals()[source])
                    else:
                        dataset_version = dataset_version.join(locals()[source], on=['Latitude', 'Longitude'])

                dataset_version['NO2_cams'] = cams[variable]

                external_variables = [var for var in list(dataset_version.columns) if var not in ['cluster', 'NO2_interp', 'dist', 'NO2_cams']]

                model_path = Path(config['paths']['models'], f'{version}.joblib')
                model = load(model_path)

                X_inference = prepare_dataset(dataset_version.reset_index(), external_variables)
                y_inference = model.predict(X_inference)

                y_inference = pd.DataFrame(
                    y_inference,
                    index=dataset.index,
                    columns=['NO2_pred'])
                
                results_filename = f"{version}_{hour}_{chunk_i}.csv"
                y_inference.to_csv(tmp_folder / results_filename)
                
                print(f"It took {(time.time() - start)/60} minutes to do the inference for model version {version} at chunk {chunk_i} for hour {hour}")
                start = time.time()

                # Maps (està dins el bucle perquè només hi ha 1 chunck)
                all_dfs = []
                for chunk_i in range(total_chunks):
                    results_filename = f"{version}_{hour}_{chunk_i}.csv"
                    all_dfs.append(pd.read_csv(tmp_folder / results_filename))
                all_dfs = pd.concat(all_dfs)
                all_dfs = all_dfs.set_index(['time', 'Latitude', 'Longitude'])
                all_dfs.sort_index(inplace=True)
                all_dfs = all_dfs.reset_index()
                plot_map(all_dfs, hour, resolution, version, global_lat, global_lon)
    
    print("---- Inference finished! ----")

    # # Start plotting the maps
    # for hour in hours_to_map:
    #     for version in model_versions.keys():
    #         all_dfs = []
    #         for chunk_i in range(total_chunks):
    #             results_filename = f"{version}_{hour}_{chunk_i}.csv"
    #             all_dfs.append(pd.read_csv(tmp_folder / results_filename))
    #         all_dfs = pd.concat(all_dfs)
    #         all_dfs = all_dfs.set_index(['time', 'Latitude', 'Longitude'])
    #         all_dfs.sort_index(inplace=True)
    #         all_dfs = all_dfs.reset_index()
    #         plot_map(all_dfs, hour, resolution, version, global_lat, global_lon)

    # print('---- Plots created! ----')

    shutil.rmtree(str(tmp_folder))

if __name__ == '__main__':
    min_lat, max_lat, min_lon, max_lon = find_cluster_bbox(cluster)
    bbox = {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon
    }
    main()
