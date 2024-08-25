""" 
This scripts reads the raw CAMS data and interpolates it at the stations point.
It can also save the results in the database.

We chose that the verion of the model would be negative to avoid conflicts with the other models.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itertools import repeat

import pandas as pd

from cams_downscaling.readers.cams import load_cams
from cams_downscaling.utils import get_db_connection, read_config
from cams_downscaling.readers import eea


config = read_config('/home/urbanaq/cams_downscaling/config')
variable = "NO2"

def get_stations(countries: list, region: str, years: list):

    stations = eea.load_stations(countries=countries,
                                 bbox=config['regions'][region]['bbox'],
                                 data_path=Path(config['paths']['stations']))

    stations, observations = eea.load_eea_data(stations=stations,
                                               pollutant=variable,
                                               years=years,
                                               countries=countries,
                                               data_path=Path(config['paths']['stations']))

    stations = eea.get_clusters(stations)
    observations = observations.join(stations['cluster'], on='station', how='inner')
    return stations, observations

def get_cams(region, date_range):
    
    return load_cams(Path(config["paths"]["cams"], "no2", region), date_range)


def interpolate(cams, stations, observations) -> pd.DataFrame:
    cams = cams.interpolate(
        lat=stations.sort_index()['Latitude'],
        lon=stations.sort_index()['Longitude'],
        grid=False).to_frame()
    cams.index = observations.sort_index().index
    cams.columns = [v.upper() for v in cams.columns]
    return cams

def store_db(cams, model_version):
    cams = cams.reset_index()
    connection = get_db_connection()
    cursor = connection.cursor()

    data_query = """INSERT INTO {} (date, model_version, value) VALUES (%s, %s, %s)"""
    
    total_stations = len(cams['station'].unique())
    for i, station in enumerate(cams['station'].unique()):
        print(f"{i+1}/{total_stations}", end="\r")
        cursor.execute(f"DELETE FROM {station} WHERE model_version={model_version}")

        station_data = cams[cams['station'] == station]
        values = [(date, model_version, value) 
                      for date, model_version, value in zip(station_data.time, 
                                                            repeat(model_version), 
                                                            station_data["NO2"])]
        cursor.executemany(data_query.format(station), values)
        
        connection.commit()
    print()
    cursor.close()
    connection.close()

def main(countries: list, region: str, years: list) -> pd:
    stations, observations = get_stations(countries, region, years)
    cams = get_cams(region, observations.index.levels[0])
    return interpolate(cams, stations, observations)
    

if __name__ == "__main__":
    #cams = main(["Spain", "Portugal"], "iberia", [2022, 2023])
    #model_version = -1000  # Negative
    #store_db(cams, model_version)

    #cams = main(["Italy"], "italy", [2022, 2023])
    #model_version = -10001  # Negative
    #store_db(cams, model_version)

    #cams = main(["Poland"], "poland", [2023])
    #model_version = -10002  # Negative
    #store_db(cams, model_version) # since we have to do the two years, comment the line to delete the data in mysql as needed

    pass
