import os
from pathlib import Path

import pandas as pd

from ..utils import read_config


config = read_config('/home/urbanaq/cams_downscaling/config')
COUNTRY_CODES = config["places"]["country_codes"]


def read_stations(countries: list[str], bbox: dict, data_path: Path) -> pd.DataFrame:
    stations = pd.read_csv(
        data_path + '/stations.csv',
        usecols=["Air Quality Station EoI Code", "Longitude", "Latitude", "Country"],
        index_col="Air Quality Station EoI Code"
    )

    # Filter stations by name. Only stations for selected countries and bbox
    stations.index.name = 'station'
    stations = stations.drop_duplicates()
    stations = stations[stations["Country"].isin(countries)].drop(columns=["Country"])
    stations = stations[(stations['Longitude'] >= bbox['min_lon']) &
                        (stations['Longitude'] <= bbox['max_lon']) &
                        (stations['Latitude'] >= bbox['min_lat']) &
                        (stations['Latitude'] <= bbox['max_lat'])]
    
    return stations
    

def read_eea_data(pollutant: str, years: list[int], countries: list[str], bbox: dict, data_path: Path) -> pd.DataFrame:
    stations = read_stations(countries, bbox, data_path)
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
