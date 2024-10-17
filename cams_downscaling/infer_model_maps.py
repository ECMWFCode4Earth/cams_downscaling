import time
from pathlib import Path

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


# VERSION NUMBERING:
# 1st number: 1 for time-based split, 2 for station-based split
split_method = 2
# 2nd - 3rd number: combination of datasets
datasets_to_run = ["09"] # 2nd - 3rd number
# 4th number: 0 for 2022 and 2023, 1 for 2022, 2 for 2023
year_code = 1
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
hours_to_map = [dt.datetime(2022, 1, 1, 0)] # Monday 8:00
# clusters_to_map = [int('10' + str(i)) for i in range(0, 15)] # Cluster 100 to 1014
clusters_to_map = [108] # Barcelona
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

config = read_config('/home/urbanaq/cams_downscaling/config')
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

def get_grid_points(cluster, resolution, hour):
    lats, lons = lats_lons(cluster, resolution)
    grid_points = pd.DataFrame(
        np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2),
        columns=['Latitude', 'Longitude'])
    grid_points['cluster'] = cluster

    # Create grid_points_time by adding a time column
    grid_points_time = grid_points.copy()
    grid_points_time['time'] = hour

    return grid_points, grid_points_time

def get_permutations(grid_point_time, n):
    print('Generating permutations')
    permutations = pd.get_dummies(
        grid_point_time.groupby(['time', 'cluster'], group_keys=False)
        .apply(lambda x: x.sample(n).assign(i=range(n)), include_groups=False)
        .get('i').astype(str).reindex(grid_point_time.index)
        .groupby(['time', 'station'], group_keys=False)
        .apply(lambda x: x.ffill().bfill())
    ).add_prefix('perm_')

    return permutations

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

def region_box_by_cluster(grid_point: pd.DataFrame) -> pd.DataFrame:
    region_box = grid_point.groupby('cluster').agg({'Latitude': ['min', 'max'], 'Longitude': ['min', 'max']})

    region_box['n'] = grid_point.cluster.value_counts()
    region_box['area (km²)'] = 111*(region_box['Latitude']['max'] - region_box['Latitude']['min']) * 111*(region_box['Longitude']['max'] - region_box['Longitude']['min'])
    region_box['area (hm²)'] = region_box['area (km²)'] * 100
    region_box['density (km²)'] = region_box['n'] / region_box['area (km²)']
    region_box['density (hm²)'] = region_box['n'] / region_box['area (hm²)']

    return region_box

def get_some_static_data(source):
    print(f'Loading {source} data')
    load_func = globals()[f'load_{sources[source]}']
    return lambda grid_point, region_box: pd.concat([
    load_func(
        config['paths'][source],
        {k: v for k, v in zip(['min_lat', 'max_lat', 'min_lon', 'max_lon'],
                              region_box.loc[c][['Latitude', 'Longitude']].values)})
    .interpolate(
        lat=grid_point[grid_point.cluster == c]['Latitude'],
        lon=grid_point[grid_point.cluster == c]['Longitude'],
        grid=False).to_frame().set_index(grid_point[grid_point.cluster == c].index)
    for c in grid_point.cluster.unique()])

def get_topography(region_box, grid_point):
    return get_some_static_data('topography')(grid_point, region_box)

def get_population(region_box, grid_point):
    return get_some_static_data('population')(grid_point, region_box)

def get_height(region_box, grid_point):
    return get_some_static_data('height')(grid_point, region_box)

def get_land_cover(region_box, grid_point):
    print(f'Loading land cover data')
    land_use = pd.concat([
    load_corine(
        config['paths']['land_cover'],
        {k: v for k, v in zip(['min_lat', 'max_lat', 'min_lon', 'max_lon'], region_box.loc[c][['Latitude', 'Longitude']].values)})
    .interpolate_discrete(
        lat=grid_point[grid_point.cluster == c]['Latitude'],
        lon=grid_point[grid_point.cluster == c]['Longitude'],
        grid=False).to_frame().set_index(grid_point[grid_point.cluster == c].index)
    for c in grid_point.cluster.unique()])

    return land_use

def get_roads(region_box, grid_point):
    print('Loading roads data')
    roads = pd.concat([
    load_osm(
        f"{config['paths']['osm']}/{region}.tif",
        {k: v for k, v in zip(['min_lat', 'max_lat', 'min_lon', 'max_lon'], region_box.loc[c][['Latitude', 'Longitude']].values)})
    .interpolate_discrete(
        lat=grid_point[grid_point.cluster == c]['Latitude'],
        lon=grid_point[grid_point.cluster == c]['Longitude'],
        grid=False).to_frame().set_index(grid_point[grid_point.cluster == c].index)
    for c in grid_point.cluster.unique()])

    return roads

def get_cams(grid_point_time, grid_point):
    print('Loading CAMS data')
    cams_path = config['paths']['cams']
    cams_path = Path(cams_path, variable.lower(), region)

    dates = grid_point_time.time.unique()
    cams = load_cams(cams_path, dates=dates)
    cams =  cams.interpolate(
        lat=grid_point.sort_index()['Latitude'],
        lon=grid_point.sort_index()['Longitude'],
        grid=False).to_frame()
    cams.index = grid_point_time.sort_index().index

    cams.columns = [v.upper() for v in cams.columns]

    return cams

def get_era5(grid_point_time, grid_point):
    print('Loading ERA5 data')
    dates = grid_point_time.time.unique()
    era5 = load_era5(Path(config['paths']['era5']), region, dates)
    era5 = era5.interpolate(
        lat=grid_point.sort_index()['Latitude'],
        lon=grid_point.sort_index()['Longitude'],
        grid=False).to_frame()
    era5.index = grid_point_time.sort_index().index

    return era5

def get_era5_land(grid_point_time, grid_point):
    print('Loading ERA5-Land data')
    dates = grid_point_time.time.unique()
    era5_land = load_era5_land(Path(config['paths']['era5_land']), region, dates)
    era5_land = era5_land.interpolate(
        lat=grid_point.sort_index()['Latitude'],
        lon=grid_point.sort_index()['Longitude'],
        grid=False).to_frame()
    era5_land.index = grid_point_time.sort_index().index

    return era5_land

def get_dataset(version,observations,
                                topography, land_use, population, height, roads,
                                cams, era5, era5_land):
    dataset = observations
    for source in model_versions[version]:
        dataset = dataset.join(locals()[source])
    
    return dataset

def get_test(dataset, permutations, cams, external_variables):

    pairs = dataset.reset_index()[['cluster', 'station']]

    test_split = pairs.drop_duplicates().groupby('cluster').apply(lambda x: x.sample(n = int(0.2 * x.shape[0] + 1), random_state=42))

    test_set = set(test_split.itertuples(index=False, name=None))

    test_dataset = dataset[[t in test_set for t in zip(dataset.cluster, dataset.index.get_level_values('station'))]].copy()

    test_dataset[f'{variable}_cams'] = cams[variable]
    test_dataset = test_dataset.join(permutations).groupby(['time', 'cluster']).apply(
        interpolate_points,
        variable=variable, external_variables=external_variables
    ).reset_index().drop(columns=['level_2'])

    return test_dataset

def get_train_test_old(dataset, permutations, cams, external_variables):
    dataset[f'{variable}_cams'] = cams[variable]
    dataset = dataset.join(permutations).groupby(['time', 'cluster']).apply(
        interpolate_points,
        variable=variable, external_variables=external_variables
    ).reset_index().drop(columns=['level_2'])

    pairs = dataset.reset_index()[['time', 'cluster']]
    pairs['time'] = pairs['time'].dt.date

    #train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_split, test_split = train_test_split(pairs.drop_duplicates(), test_size=0.1, random_state=42)

    # Copiat de la notebook
    train_set = set(train_split.itertuples(index=False, name=None))
    test_set = set(test_split.itertuples(index=False, name=None))

    train_dataset = dataset[[t in train_set for t in zip(dataset.time.dt.date, dataset.cluster)]].copy()
    test_dataset = dataset[[t in test_set for t in zip(dataset.time.dt.date, dataset.cluster)]].copy()

    return train_dataset, test_dataset

def get_model_test_sets(version, observations, permutations, topography, land_use, population, height, roads, cams, era5, era5_land):
    dataset = get_dataset(version,observations,
                                topography, land_use, population, height, roads,
                                cams, era5, era5_land)
    external_variables = dataset.columns[4:].tolist()

    test_dataset = get_test(dataset, permutations, cams, external_variables)
    X_test, y_test = prepare_dataset(test_dataset, external_variables)

    return test_dataset, X_test, y_test

def save_results(dataset, X_test, y_pred, model_version):
    print(f'Saving results for model version {model_version}')
    stations=dataset.station
    date=dataset.time
    points=dataset[['Latitude', 'Longitude']]

    db = DatabaseTSPD(stations, date, points, values=y_pred)

    db.save_sql(model_version=int(model_version), drop_previous=True)

def get_datasets(grid_point, grid_point_time):
    region_box = region_box_by_cluster(grid_point)

    # Static Datasets
    topography = get_topography(region_box, grid_point)
    land_use = get_land_cover(region_box, grid_point)
    population = get_population(region_box, grid_point)
    height = get_height(region_box, grid_point)
    roads = get_roads(region_box, grid_point)

    # CAMS Dataset
    cams = get_cams(grid_point_time, grid_point)

    # Meteo Datasets
    era5 = get_era5(grid_point_time, grid_point)
    era5_land = get_era5_land(grid_point_time, grid_point)

    return topography, land_use, population, height, roads, cams, era5, era5_land

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

def lats_lons(cluster, resolution):
    # Get the bounding box for the cluster. Resolution is in meters
    resolution = resolution / 111000
    min_lat, max_lat, min_lon, max_lon = find_cluster_bbox(cluster)
    lats = np.arange(min_lat, max_lat, resolution)
    lons = np.arange(min_lon, max_lon, resolution)
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

def plot_map(y_inference, cluster, hour: dt.datetime, resolution, version, grid_lat, grid_lon):
    data = y_inference.xs(hour, level='time').unstack().values

    # Define the extent
    min_lat, max_lat, min_lon, max_lon = find_cluster_bbox(cluster)
    extent = (min_lon, max_lon, min_lat, max_lat)

    # Create a meshgrid for the coordinates
    lon, lat = np.meshgrid(grid_lon, grid_lat)

    # Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution="10m")  # type: ignore

    # Set the extent of the map
    ax.set_extent(extent, crs=ccrs.PlateCarree()) # type: ignore

    # Plot the data
    im = ax.imshow(data, extent=extent, transform=ccrs.PlateCarree(), cmap=plt.cm.jet, origin='lower') # type: ignore

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
    cbar.set_label('NO2 (µg/m³)')

    # Set the title
    plt.title(f"Downscaled NO2 concentration in {config['places']['cluster_names'][cluster]}\n"
          + hour.strftime('%Y-%m-%d %H:%M:%S') + f'\n Model version: {version} - Resolution: {resolution} m')

    # Add lats and lons to the axis
    ax.set_xticks(grid_lon, crs=ccrs.PlateCarree())
    ax.set_yticks(grid_lat, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True, transform_precision=0.01)
    lat_formatter = LatitudeFormatter(transform_precision=0.01)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # Create the output path
    output_path = Path(config['paths']['maps'],
                       f"{version}_{config['places']['cluster_names'][cluster]}_{hour.strftime('%Y-%m-%d_%H-%M-%S')}_{resolution}m.png")

    # Ensure the output directory and its partents exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)

def main():

    start = time.time()  
    for cluster in clusters_to_map:
        for hour in hours_to_map:

            # Get the meshgrid for the cluster to plot
            # grid_point, grid_point_time = get_grid_points(cluster, resolution, hour)

            min_lat, max_lat, min_lon, max_lon = find_cluster_bbox(cluster)
            grid_lat, grid_lon = lats_lons(cluster, resolution)
            stations, observations = eea_stations(countries)

            dataset = observations[[c == cluster for c in observations.cluster]].copy()

            dataset = dataset.groupby(['time', 'cluster']).apply(
                interpolate_points, # type: ignore
                variable=variable,
                lats=grid_lat,
                lons=grid_lon
            ).reset_index().drop(columns=['level_2']).set_index(['time', 'Latitude', 'Longitude']).sort_index() # type: ignore

            # Model Datasets
            cams_path = config['paths']['cams']
            cams_path = Path(cams_path, variable.lower(), region)

            dates = observations.index.get_level_values('time').unique()

            cams = load_cams(cams_path, dates=dates)

            cams =  cams.interpolate(
                lat=grid_lat,
                lon=grid_lon,
                grid=True).to_frame()

            cams.index = dataset.index
            cams.columns = [v.upper() for v in cams.columns]

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

            era5 = (
                load_era5(
                    Path(config['paths']['era5']),
                    region,
                    dates) # type: ignore
                .interpolate(
                    lat=grid_lat,
                    lon=grid_lon).to_frame())
            era5.index = dataset.index

            era5_land = (
                load_era5_land(
                    Path(config['paths']['era5_land']),
                    region,
                    dates) # type: ignore
                .interpolate(
                    lat=grid_lat,
                    lon=grid_lon).to_frame())
            era5_land.index = dataset.index

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


            print(f"It took {(time.time() - start)/60} minutes to load all datasets and create permutations")
            start = time.time()

            print()
            print('---- Running models ----')
            print()

            start = time.time()

            for version in model_versions.keys():
                print(f'Running model version {version}')
                
                for source in model_versions[version]:
                    if source in ['era5', 'era5_land']:
                        dataset = dataset.join(locals()[source])
                    else:
                        dataset = dataset.join(locals()[source], on=['Latitude', 'Longitude'])

                dataset['NO2_cams'] = cams[variable]

                external_variables = [var for var in list(dataset.columns) if var not in ['cluster', 'NO2_interp', 'dist', 'NO2_cams']]

                model_path = Path(config['paths']['models'], f'{version}.joblib')
                model = load(model_path)

                X_inference = prepare_dataset(dataset.reset_index(), external_variables)
                y_inference = model.predict(X_inference)

                y_inference = pd.DataFrame(
                    y_inference,
                    index=dataset.index,
                    columns=['NO2_pred'])
                
                plot_map(y_inference, cluster, hour, resolution, version, grid_lat, grid_lon)

                start = time.time()

    print('---- Models finished ----')

if __name__ == '__main__':
    main()
