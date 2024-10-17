import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree
from joblib import dump

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
from .utils import read_config


# VERSION NUMBERING:
# 1st number: 1 for time-based split, 2 for station-based split
split_method = [2] # This script uses only station-based split
# 2nd - 3rd number: combination of datasets
datasets_to_run = ["00", "01", "02", "03", "04", "06", "07", "08", "09", "10"] # 2nd + 3rd number
# 4th number: 0 for 2022 and 2023, 1 for 2022, 2 for 2023
year_code = [0, 1, 2]
# 5th number: None for Iberia, 1 for Italy, 2 for Poland
region_code = [None, 1]

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

# AUTOMATIC GENERATION OF VERSIONS BASED ON ABOVE
model_versions = {
    f"{method}{combination}{period}{region if region else ''}": datasets_combinations[combination]
    for combination in datasets_to_run
    for method in split_method
    for period in year_code
    for region in region_code
}

years_code = {
        0: [2022, 2023],
        1: [2022],
        2: [2023]
    }

country_code = {
    None: ["Spain", "Portugal"],
    1: ["Italy"],
    2: ["Poland"]
}

regions_code = {
    None: "iberia",
    1: "italy",
    2: "poland"
}

country_region = {
    "Spain": "iberia",
    "Portugal": "iberia",
    "Italy": "italy",
    "Poland": "poland"
}

region_cluster = {
        'iberia': [1, 2, 3],
        'italy': [4],
        'poland': [5]
    }

all_countries = []
for i in region_code:
    all_countries = all_countries + country_code[i]

all_regions = [regions_code[i] for i in region_code]

all_years = []
for i in year_code:
    all_years = all_years + years_code[i]

all_years = list(set(all_years))

config = read_config('/home/urbanaq/cams_downscaling/config')

pollutant_code = 1 # 1 for NO2
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

    stations = []
    for country in countries:
        stations.append(eea.load_stations(countries=[country],
                                     bbox=config['regions'][country_region[country]]['bbox'],
                                     data_path=Path(config['paths']['stations'])))
        
    stations = pd.concat(stations)

    stations, observations = eea.load_eea_data(stations=stations,
                                               pollutant=variable,
                                               years=all_years,
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
    print(f'Loading land cover data')
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

def get_roads(region_box, stations_all):
    print('Loading roads data')

    roads_dict = {}
    for region in all_regions:
        # Filter stations in the region
        clusters_in_region = [i for i in region_box.index if int(str(i)[0]) in region_cluster[region]]
        stations = stations_all[stations_all['cluster'].isin(clusters_in_region)]

        roads = pd.concat([
        load_osm(
            f"{config['paths']['osm']}/{region}.tif",
            {k: v for k, v in zip(['min_lat', 'max_lat', 'min_lon', 'max_lon'], region_box.loc[c][['Latitude', 'Longitude']].values)})
        .interpolate_discrete(
            lat=stations[stations.cluster == c]['Latitude'],
            lon=stations[stations.cluster == c]['Longitude'],
            grid=False).to_frame().set_index(stations[stations.cluster == c].index)
        for c in stations.cluster.unique()])

        roads_dict[region] = roads

    return roads_dict

def get_cams(all_observations, all_stations):
    print('Loading CAMS data')
    cams_dict = {}
    base_cams_path = config['paths']['cams']

    for region in all_regions:
        # Filter stations in the region
        clusters_in_region = [i for i in all_stations['cluster'].unique() if int(str(i)[0]) in region_cluster[region]]
        stations = all_stations[all_stations['cluster'].isin(clusters_in_region)]
        observations = all_observations[all_observations['cluster'].isin(clusters_in_region)]

        cams_path = Path(base_cams_path, variable.lower(), region)

        dates = observations.index.get_level_values('time').unique()
        cams = load_cams(cams_path, dates=dates)
        cams =  cams.interpolate(
            lat=stations.sort_index()['Latitude'],
            lon=stations.sort_index()['Longitude'],
            grid=False).to_frame()
        cams.index = observations.sort_index().index

        cams.columns = [v.upper() for v in cams.columns]

        cams_dict[region] = cams

    return cams_dict

def get_era5(all_observations, all_stations):
    print('Loading ERA5 data')
    era5_dict = {}
    for region in all_regions:
        # Filter stations in the region
        clusters_in_region = [i for i in all_stations['cluster'].unique() if int(str(i)[0]) in region_cluster[region]]
        stations = all_stations[all_stations['cluster'].isin(clusters_in_region)]
        observations = all_observations[all_observations['cluster'].isin(clusters_in_region)]

        dates = observations.index.get_level_values('time').unique()
        era5 = load_era5(Path(config['paths']['era5']), region, dates)
        era5 = era5.interpolate(
            lat=stations.sort_index()['Latitude'],
            lon=stations.sort_index()['Longitude'],
            grid=False).to_frame()
        era5.index = observations.sort_index().index

        era5_dict[region] = era5

    return era5_dict

def get_era5_land(all_observations, all_stations):
    print('Loading ERA5-Land data')
    era5_land_dict = {}
    for region in all_regions:
        # Filter stations in the region
        clusters_in_region = [i for i in all_stations['cluster'].unique() if int(str(i)[0]) in region_cluster[region]]
        stations = all_stations[all_stations['cluster'].isin(clusters_in_region)]
        observations = all_observations[all_observations['cluster'].isin(clusters_in_region)]

        dates = observations.index.get_level_values('time').unique()
        era5_land = load_era5_land(Path(config['paths']['era5_land']), region, dates)
        era5_land = era5_land.interpolate(
            lat=stations.sort_index()['Latitude'],
            lon=stations.sort_index()['Longitude'],
            grid=False).to_frame()
        era5_land.index = observations.sort_index().index

        era5_land_dict[region] = era5_land

    return era5_land

def get_dataset(version,observations,
                                topography, land_use, population, height, roads,
                                cams, era5, era5_land):
    dataset = observations
    for source in model_versions[version]:
        dataset = dataset.join(locals()[source])
    
    return dataset

def get_train_test(dataset, permutations, cams, external_variables):
 
    pairs = dataset.reset_index()[['cluster', 'station']]

    test_split = pairs.drop_duplicates().groupby('cluster').apply(lambda x: x.sample(n = int(0.2 * x.shape[0] + 1), random_state=42))
    train_split = pairs[~pairs.index.isin(test_split.index)]

    train_set = set(train_split.itertuples(index=False, name=None))

    train_dataset = dataset[[t in train_set for t in zip(dataset.cluster, dataset.index.get_level_values('station'))]].copy()

    train_dataset[f'{variable}_cams'] = cams[variable]
    train_dataset = train_dataset.join(permutations).groupby(['time', 'cluster']).apply(
        interpolate_points,
        variable=variable, external_variables=external_variables
    ).reset_index().drop(columns=['level_2'])

    return train_dataset

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

def get_model_train_set(version, observations, permutations, topography, land_use, population, height, roads, cams, era5, era5_land):
    dataset = get_dataset(version,observations,
                                topography, land_use, population, height, roads,
                                cams, era5, era5_land)
    external_variables = dataset.columns[4:].tolist()

    train_dataset = get_train_test(dataset, permutations, cams, external_variables)
    X_train, y_train = prepare_dataset(train_dataset, external_variables)

    return X_train, y_train

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

def get_years(version):
    year_code = int(version[3])
    return years_code[year_code]

def get_region_countries(version):
    if len(version) == 5:
        return {
            1: 'italy',
            2: 'poland'
        }[int(version[4])], country_code[int(version[4])]
    elif len(version) == 4:
        return 'iberia', ["Spain", "Portugal"]
    
def main():

    start = time.time()
    # EEA Observations Dataset (to train and test the model)
    stations, observations = eea_stations(countries=all_countries)

    # Model Datasets
    topography, land_use, population, height, all_roads, all_cams, all_era5, all_era5_land = get_datasets(stations, observations)

    permutations = get_permutations(observations, n=5)

    print(f"It took {(time.time() - start)/60} minutes to load all datasets and create permutations")
    start = time.time()

    print()
    print('---- Training models ----')
    print()

    start = time.time()

    for version in model_versions.keys():

        global years, region, countries
        years = get_years(version)
        region, countries = get_region_countries(version) # type: ignore
        roads = all_roads[region]
        cams = all_cams[region]
        era5 = all_era5[region]
        era5_land = all_era5_land[region]
        
        print(f'Training model version {version}')
        X_train, y_train = get_model_train_set(
            version, observations, permutations,
            topography, land_use, population, height, roads,
            cams, era5, era5_land)
        
        # Initialize and train the model
        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)

        print(f"It took {(time.time() - start)/60} minutes to train model version {version}")
        start = time.time()

        # Save model
        model_path = Path(config['paths']['models'], f'{version}.joblib')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        dump(model, model_path)
        
        print(f"It took {(time.time() - start)/60} minutes to store parametrers for model version {version}")
        start = time.time()

    print('---- Models finished ----')

if __name__ == '__main__':
    main()
