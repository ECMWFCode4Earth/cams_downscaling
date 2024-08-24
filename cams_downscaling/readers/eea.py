import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from ..utils import read_config, get_db_connection


config = read_config('/home/urbanaq/cams_downscaling/config')
COUNTRY_CODES = config["places"]["country_codes"]


def load_stations(countries: list[str], bbox: dict, data_path: Path) -> pd.DataFrame:
    stations = pd.read_csv(
        data_path / 'stations.csv',
        usecols=["Air Quality Station EoI Code", "Longitude", "Latitude", "Country"],
        index_col="Air Quality Station EoI Code"
    )

    # Filter stations by name. Only stations for selected countries and bbox
    stations.index.name = 'station'
    stations = stations[~stations.index.duplicated(keep='first')]
    stations = stations[stations["Country"].isin(countries)].drop(columns=["Country"])
    stations = stations[(stations['Longitude'] >= bbox['min_lon']) &
                        (stations['Longitude'] <= bbox['max_lon']) &
                        (stations['Latitude'] >= bbox['min_lat']) &
                        (stations['Latitude'] <= bbox['max_lat'])]
    
    return stations
    

def load_eea_data(data_path: Path, stations: pd.DataFrame, pollutant: str, years: list[int], countries: list[str]) -> pd.DataFrame:

    contries_data = []

    for country in countries:
        country_code = COUNTRY_CODES[country]
        for year in years:
            year = str(year)
            if os.path.isdir(data_path / year):
                if f'{country_code}.csv' not in os.listdir(data_path / year):
                    continue
                country_data = pd.read_csv(
                    str(data_path / year / f'{country_code}.csv'),
                    usecols=['time', 'station', pollutant],
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
    observations = observations.groupby('station').filter(lambda x: x[pollutant].notnull().any())
    # Drop observations where Latitude is NaN
    observations = observations.groupby('station').filter(lambda x: x['Latitude'].notnull().any())
    # Drop stations where NO2 is NaN
    stations = stations[stations.index.isin(observations.index.get_level_values('station').unique())]

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