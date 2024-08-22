import os
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree

from .datatypes import DatabaseTSPD
from .readers.topography import load_topography
from .readers.land_cover import load_corine
from .readers.population import load_pop
from .readers.build_height import load_height
from .readers.roads import load_osm
from .readers.cams import load_cams
from .readers.era5 import load_era5
from .readers.era5_land import load_era5_land
from .utils import read_config, get_db_connection

# The 4th number of all versions is version_to_run needs to be the same
versions_to_run = ['1000', '1010', '1020', '1030', '1040', '1060', '1070', '1080', '1090', '1100']
#versions_to_run = ['1001', '1011', '1021', '1031', '1041', '1061', '1071', '1081', '1091', '1101']
#versions_to_run = ['1002', '1012', '1022', '1032', '1042', '1062', '1072', '1082', '1092', '1102']
#countries = ['Poland']
countries = ['Spain', 'Portugal']

#TODO: Gereralize region definition, depending on the countries
region = 'iberia'#'poland'

model_versions = {
    '1000': ['topography','land_use'], # 2022 and 2023
    '1010': ['topography','land_use','era5_land','era5'],
    '1020': ['topography','land_use','population'],
    '1030': ['topography','land_use','height'],
    '1040': ['topography','land_use','era5_land','era5','population','height'],
    '1050': [],
    '1060': ['topography'],
    '1070': ['land_use'],
    '1080': ['topography','land_use', 'roads'], # 2022 and 2023
    '1090': ['topography','land_use','era5_land','era5','population','height','roads'],
    '1100': ['roads'],

    '1001': ['topography','land_use'], # 2022
    '1011': ['topography','land_use','era5_land','era5'],
    '1021': ['topography','land_use','population'],
    '1031': ['topography','land_use','height'],
    '1041': ['topography','land_use','era5_land','era5','population','height'],
    '1051': [],
    '1061': ['topography'],
    '1071': ['land_use'],
    '1081': ['topography','land_use', 'roads'], # 2022
    '1091': ['topography','land_use','era5_land','era5','population','height','roads'],
    '1101': ['roads'],

    '1002': ['topography','land_use'], # 2022
    '1012': ['topography','land_use','era5_land','era5'],
    '1022': ['topography','land_use','population'],
    '1032': ['topography','land_use','height'],
    '1042': ['topography','land_use','era5_land','era5','population','height'],
    '1052': [],
    '1062': ['topography'],
    '1072': ['land_use'],
    '1082': ['topography','land_use', 'roads'], # 2022
    '1092': ['topography','land_use','era5_land','era5','population','height','roads'],
    '1102': ['roads'],
}

model_versions_years = {
    0: [2022, 2023],
    1: [2022],
    2: [2023]
}

country_codes = {
    'Spain': 'ES',
    'Portugal': 'PT',
    'Andorra': 'AD',
    'Italy': 'IT',
    'Poland': 'PL',
}

needed_sources = list({var for version in versions_to_run for var in model_versions[version]})

if False in [versions_to_run[i] in model_versions.keys() for i in range(len(versions_to_run))]:
    raise ValueError('Some versions are not defined in model_versions')

# Check that the last number of all versions is version_to_run is the same
if len(set([version[-1] for version in versions_to_run])) != 1:
    raise ValueError('The last number of all versions is version_to_run needs to be the same')

sources_vars = {'topography': ['elevation'],
                'land_use': ['land'],
                'population': ['population'],
                'height': ['height'],
                'roads': ['roads'],
                'era5': ['d2m','ssr','t2m','tp','v10','u10'],
                'era5_land': ['blh','iews','inss']}

config = read_config('/home/urbanaq/cams_downscaling/config')
variable = 'NO2'

new_resolution = 0.01

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

    bbox = config['regions'][region]['bbox']
    stations = pd.read_csv(
        config['paths']['stations'] + '/stations.csv',
        usecols=["Air Quality Station EoI Code", "Longitude", "Latitude", "Country"],
        index_col="Air Quality Station EoI Code"
    )

    # Filter stations by name. Only stations for selected countries
    stations.index.name = 'station'
    stations = stations.drop_duplicates()
    stations = stations[stations["Country"].isin(countries)].drop(columns=["Country"])
    stations = stations[(stations['Longitude'] >= bbox['min_lon']) &
                        (stations['Longitude'] <= bbox['max_lon']) &
                        (stations['Latitude'] >= bbox['min_lat']) &
                        (stations['Latitude'] <= bbox['max_lat'])]

    contries_data = []
    years = model_versions_years[int(versions_to_run[0][3])]
    for country in countries:
        country_code = country_codes[country]
        for year in years:
            year = str(year)
            if os.path.isdir(os.path.join(config['paths']['stations'], year)):
                if f'{country_code}.csv' not in os.listdir(os.path.join(config['paths']['stations'], year)):
                    continue
                country_data = pd.read_csv(
                    os.path.join(config['paths']['stations'], year, f'{country_code}.csv'),
                    usecols=['time', 'station', variable],
                    index_col=['time', 'station'],
                    parse_dates=['time'])
                contries_data.append(country_data)
        
    observations = pd.concat(contries_data)

    all_combinations = pd.MultiIndex.from_product([
        observations.index.get_level_values('time').unique(),
        observations.index.get_level_values('station').unique()],
        names=['time', 'station']).to_frame(index=False)
    
    observations = pd.merge(all_combinations, observations, on=['time', 'station'], how='left').set_index(['time', 'station'])

    # Join the station coordinates
    observations = observations.join(stations, how='left', on='station')
    # Drop observations where NO2 is NaN
    observations = observations.groupby('station').filter(lambda x: x[variable].notnull().any())
    # Drop observations where Latitude is NaN
    observations = observations.groupby('station').filter(lambda x: x['Latitude'].notnull().any())
    # Drop stations where NO2 is NaN
    stations = stations[stations.index.isin(observations.index.get_level_values('station').unique())]
    stations = get_clusters(stations)
    # complete_clusters_sql(stations)
    observations = observations.join(stations['cluster'], on='station', how='inner')
    observations = fill_missing_values(observations, variable=variable)

    return stations, observations

def get_clusters(stations: pd.DataFrame) -> pd.DataFrame:
    # Add the cluster column to the stations dataframe by connecting to the stations table of the SQL database
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT station_id, cluster FROM stations WHERE cluster IS NOT NULL"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    data = [(row[0].decode("utf-8"), int(row[1].decode("utf-8"))) for row in data]

    clusters = pd.DataFrame(data, columns=['station', 'cluster'])
    clusters = clusters.set_index('station')
    stations = stations.join(clusters, how='inner')

    return stations

def fill_missing_values(observations, variable):
    df = observations.copy()
    df['hour'] = df.index.get_level_values('time').hour
    df['dayofweek'] = df.index.get_level_values('time').dayofweek
    df['weekofyear'] = df.index.get_level_values('time').isocalendar().week.values
    df[f'last_observed_{variable}'] = df.groupby('station')[variable].shift(1).transform(lambda x: x.ffill())
    
    missing = df[df[variable].isnull()]
    df = df.dropna(subset=[variable])
    
    features = ['hour', 'dayofweek', 'weekofyear', 'Longitude', 'Latitude', f'last_observed_{variable}']
    target = variable
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
    model.fit(X_train, y_train)
        
    missing = missing.join(
        df.groupby(['station', 'hour', 'dayofweek'])[f'last_observed_{variable}'].mean(),
        on=['station', 'hour', 'dayofweek'],
        rsuffix='_mean')
    
    missing[f'last_observed_{variable}'] = missing[f'last_observed_{variable}'].fillna(missing[f'last_observed_{variable}_mean'])
    
    missing = missing.drop(columns=[f'last_observed_{variable}_mean'])
    
    # Predict the missing values
    missing[variable] = model.predict(missing[features])
    observations = pd.concat([df, missing])[[variable, 'Latitude', 'Longitude', 'cluster']]
    observations = observations.sort_index()
    
    return observations

def get_permutations(observations, n):
    print('Generating permutations')
    permutations = pd.get_dummies(
        observations.groupby([observations.index.get_level_values('time').date, 'cluster'], group_keys=False)
        .apply(lambda x: x.sample(n).assign(i=range(n)), include_groups=False)
        .get('i').astype(str).reindex(observations.index)
        .groupby([observations.index.get_level_values('time').date, 'station'], group_keys=False)
        .apply(lambda x: x.ffill().bfill())
    ).add_prefix('perm_')

    return permutations

def interpolate_points(x, variable, external_variables):
    perm = x.filter(like='perm_').reset_index(drop=True)
    station = x.index.get_level_values('station')
    x = x[[
        'Latitude', 'Longitude']
        + external_variables +
        [f'{variable}_cams', variable]].reset_index(drop=True)
    x.insert(2, 'station', station)

    # Separate points and observations
    mask = perm.values.T
    all_points = x.values
    
    interpolated_results = []

    for mask_row in mask:
        points = all_points[~mask_row]
        obs = all_points[mask_row]

        tree = KDTree(points[:, :2])
        dist, idx = tree.query(obs[:, :2])
        
        interpolated_results.append(np.hstack([
            obs,
            points[idx][:, -1:],
            dist[:, np.newaxis]]))
    
    # Combine all results into a single DataFrame
    return pd.DataFrame(
        np.vstack(interpolated_results),
        columns=[
            'Latitude', 'Longitude', 'station']
            + external_variables +
            [f'{variable}_cams',
            f'{variable}_obs', f'{variable}_interp', 'dist'])

def region_box_by_cluster(stations: pd.DataFrame) -> pd.DataFrame:
    region_box = stations.groupby('cluster').agg({'Latitude': ['min', 'max'], 'Longitude': ['min', 'max']})

    region_box['n'] = stations.cluster.value_counts()
    region_box['area (km²)'] = 111*(region_box['Latitude']['max'] - region_box['Latitude']['min']) * 111*(region_box['Longitude']['max'] - region_box['Longitude']['min'])
    region_box['area (hm²)'] = region_box['area (km²)'] * 100
    region_box['density (km²)'] = region_box['n'] / region_box['area (km²)']
    region_box['density (hm²)'] = region_box['n'] / region_box['area (hm²)']

    return region_box

def get_some_static_data(source):
    print(f'Loading {source} data')
    load_func = globals()[f'load_{sources[source]}']
    return lambda stations, region_box: pd.concat([
    load_func(
        config['paths'][source],
        {k: v for k, v in zip(['min_lat', 'max_lat', 'min_lon', 'max_lon'],
                              region_box.loc[c][['Latitude', 'Longitude']].values)})
    .interpolate(
        lat=stations[stations.cluster == c]['Latitude'],
        lon=stations[stations.cluster == c]['Longitude'],
        grid=False).to_frame().set_index(stations[stations.cluster == c].index)
    for c in stations.cluster.unique()])

def get_topography(region_box, stations):
    return get_some_static_data('topography')(stations, region_box)

def get_population(region_box, stations):
    return get_some_static_data('population')(stations, region_box)

def get_height(region_box, stations):
    return get_some_static_data('height')(stations, region_box)

def get_land_cover(region_box, stations):
    land_use = pd.concat([
    load_corine(
        config['paths']['land_cover'],
        {k: v for k, v in zip(['min_lat', 'max_lat', 'min_lon', 'max_lon'], region_box.loc[c][['Latitude', 'Longitude']].values)})
    .interpolate_discrete(
        lat=stations[stations.cluster == c]['Latitude'],
        lon=stations[stations.cluster == c]['Longitude'],
        grid=False).to_frame().set_index(stations[stations.cluster == c].index)
    for c in stations.cluster.unique()])

    return land_use

def get_roads(region_box, stations):
    print('Loading roads data')
    roads = pd.concat([
    load_osm(
        f"{config['paths']['osm']}/{region}.tif",
        {k: v for k, v in zip(['min_lat', 'max_lat', 'min_lon', 'max_lon'], region_box.loc[c][['Latitude', 'Longitude']].values)})
    .interpolate_discrete(
        lat=stations[stations.cluster == c]['Latitude'],
        lon=stations[stations.cluster == c]['Longitude'],
        grid=False).to_frame().set_index(stations[stations.cluster == c].index)
    for c in stations.cluster.unique()])

    return roads

def get_cams(observations, stations):
    print('Loading CAMS data')
    cams_path = config['paths']['cams']
    cams_path = os.path.join(cams_path, variable.lower(), region)

    dates = observations.index.get_level_values('time').unique()
    cams = load_cams(cams_path, dates=dates)
    cams =  cams.interpolate(
        lat=stations.sort_index()['Latitude'],
        lon=stations.sort_index()['Longitude'],
        grid=False).to_frame()
    cams.index = observations.sort_index().index

    cams.columns = [v.upper() for v in cams.columns]

    return cams

def get_era5(observations, stations):
    print('Loading ERA5 data')
    dates = observations.index.get_level_values('time').unique()
    era5 = load_era5(config['paths']['era5'], dates)
    era5 = era5.interpolate(
        lat=stations.sort_index()['Latitude'],
        lon=stations.sort_index()['Longitude'],
        grid=False).to_frame()
    era5.index = observations.sort_index().index

    return era5

def get_era5_land(observations, stations):
    print('Loading ERA5-Land data')
    dates = observations.index.get_level_values('time').unique()
    era5_land = load_era5_land(config['paths']['era5_land'], dates)
    era5_land = era5_land.interpolate(
        lat=stations.sort_index()['Latitude'],
        lon=stations.sort_index()['Longitude'],
        grid=False).to_frame()
    era5_land.index = observations.sort_index().index

    return era5_land

def get_dataset(version,observations,
                                topography, land_use, population, height, roads,
                                cams, era5, era5_land):
    dataset = observations
    for source in model_versions[version]:
        dataset = dataset.join(locals()[source])
    
    return dataset

def get_train_test(dataset, permutations, cams, external_variables):
    pairs = dataset.reset_index()[['time', 'cluster']]
    pairs['time'] = pairs['time'].dt.date

    train_split, test_split = train_test_split(pairs.drop_duplicates(), test_size=0.1, random_state=42)

    train_set = set(train_split.itertuples(index=False, name=None))
    test_set = set(test_split.itertuples(index=False, name=None))

    train_dataset = dataset[[t in train_set for t in zip(dataset.index.get_level_values('time').date, dataset.cluster)]].copy()
    test_dataset = dataset[[t in test_set for t in zip(dataset.index.get_level_values('time').date, dataset.cluster)]].copy()

    train_dataset[f'{variable}_cams'] = cams[variable]
    train_dataset = train_dataset.join(permutations).groupby(['time', 'cluster']).apply(
        interpolate_points,
        variable=variable, external_variables=external_variables
    ).reset_index().drop(columns=['level_2'])

    test_dataset[f'{variable}_cams'] = cams[variable]
    test_dataset = test_dataset.join(permutations).groupby(['time', 'cluster']).apply(
        interpolate_points,
        variable=variable, external_variables=external_variables
    ).reset_index().drop(columns=['level_2'])

    return train_dataset, test_dataset


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

def prepare_dataset(data, external_variables):
    data['hour'] = pd.to_datetime(data['time']).dt.hour
    data['day'] = pd.to_datetime(data['time']).dt.day
    data['month'] = pd.to_datetime(data['time']).dt.month
    data['year'] = pd.to_datetime(data['time']).dt.year

    features = [
        'cluster', 'Latitude', 'Longitude'] + external_variables + [
        F'{variable}_cams', f'{variable}_interp', 'dist', 'hour', 'day', 'month', 'year']

    X = data[features]
    y = data[f'{variable}_obs']

    return X, y

def get_model_sets(version, observations, permutations, topography, land_use, population, height, roads, cams, era5, era5_land):
    dataset = get_dataset(version,observations,
                                topography, land_use, population, height, roads,
                                cams, era5, era5_land)
    external_variables = dataset.columns[4:].tolist()

    train_dataset, test_dataset = get_train_test(dataset, permutations, cams, external_variables)
    X_train, y_train = prepare_dataset(train_dataset, external_variables)
    X_test, y_test = prepare_dataset(test_dataset, external_variables)

    return train_dataset, test_dataset, X_train, y_train, X_test, y_test

def save_results(dataset, X_test, y_pred, model_version):
    print(f'Saving results for model version {model_version}')
    stations=dataset.station
    date=dataset.time
    points=dataset[['Latitude', 'Longitude']]

    db = DatabaseTSPD(stations, date, points, values=y_pred)

    db.save_sql(model_version=int(model_version), drop_previous=True)

def get_datasets(stations, observations):
    region_box = region_box_by_cluster(stations)

    # Static Datasets
    topography = get_topography(region_box, stations)
    land_use = get_land_cover(region_box, stations)
    population = get_population(region_box, stations)
    height = get_height(region_box, stations)
    roads = get_roads(region_box, stations)

    # CAMS Dataset
    cams = get_cams(observations, stations)

    # Meteo Datasets
    era5 = get_era5(observations, stations)
    era5_land = get_era5_land(observations, stations)

    return topography, land_use, population, height, roads, cams, era5, era5_land
    
def main():

    start = time.time()
    # EEA Observations Dataset (to train and test the model)
    stations, observations = eea_stations(countries=countries)

    # Model Datasets
    topography, land_use, population, height, roads, cams, era5, era5_land = get_datasets(stations, observations)

    permutations = get_permutations(observations, n=5)

    print(f"It took {(time.time() - start)/60} minutes to load all datasets and create permutations")
    start = time.time()

    print()
    print('---- Running models ----')
    print()

    start = time.time()

    for version in versions_to_run:
        
        print(f'Running model version {version}')
        _, test_dataset, X_train, y_train, X_test, y_test = get_model_sets(
            version, observations, permutations,
            topography, land_use, population, height, roads,
            cams, era5, era5_land)
        
        # Initialize and train the model
        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate the model
        from sklearn.metrics import mean_squared_error, r2_score

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Root Mean Squared Error: {mse**0.5}')
        print(f'R² Score: {r2}')

        print(f"It took {(time.time() - start)/60} minutes to train model version {version}")
        start = time.time()

        # Save results
        save_results(test_dataset, X_test, y_pred, model_version=version)

        print(f"It took {(time.time() - start)/60} minutes to store the results for model version {version}")
        print()

        start = time.time()

    print('---- Models finished ----')

if __name__ == '__main__':
    main()
