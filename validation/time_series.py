from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt
import os

import MySQLdb as mysql
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from cams_downscaling.utils import read_config


# MODIFY THE FOLLOWING VARIABLES ACCORDING TO YOUR NEEDS
date_ini = dt.date(2022, 1, 1)
date_end = dt.date(2022, 12, 31)

model_versions_to_plot = [1000, 1010, 1020, 1030, 1040, 1060, 1070, 1080, 1090, 1100]
model_versions_to_plot = [1001, 1011, 1021, 1031, 1041, 1061, 1071, 1081, 1091, 1101]

countries = ['Spain', 'Portugal']
# END OF VARIABLES TO MODIFY

config = read_config('/home/urbanaq/cams_downscaling/config')
DATA_PATH = Path(config["paths"]["stations"])
OUTPUT_PATH = Path(config["paths"]["validation_results"]) / "time_series"
COUNTRY_CODES = config["places"]["country_codes"]
CLUSTER_NAMES = config["places"]["cluster_names"]


def get_stations(connector, cluster: int) -> list[str]:
    cursor = connector.cursor()
    query = f"""SELECT station_id FROM stations WHERE cluster={cluster}"""
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()

    stations = [row[0] for row in data]

    return stations

def get_country_data(country: str, pollutant: str, date_ini: dt.date, date_end: dt.date) -> pd.DataFrame:
    df = []
    for year in os.listdir(DATA_PATH):
        if os.path.isdir(os.path.join(DATA_PATH, year)):

            if f'{COUNTRY_CODES[country]}.csv' not in os.listdir(os.path.join(DATA_PATH, year)):
                continue
            csv_path = DATA_PATH / year / f"{COUNTRY_CODES[country]}.csv"
            df_year = pd.read_csv(csv_path, parse_dates=['time'])
            
            df_year = df_year[['time', 'station', pollutant]].dropna()
            df_year = df_year.rename(columns={'time': 'date', pollutant: 'obs_values'})

            df.append(df_year)
    
    df = pd.concat(df)

    return df

def get_station_data(df_country: pd.DataFrame, station: str) -> pd.DataFrame:
    df_station = df_country[df_country['station'] == station]
    df_station = df_station.drop(['station'], axis=1)

    return df_station

def get_model_data(connector, version: int, station: str, date_ini: dt.date, date_end: dt.date) -> pd.DataFrame:
    cursor = connector.cursor()

    query = f"SELECT `date`, `value` FROM `{station}` WHERE `model_version`={version}"

    if date_ini:
        query += f" AND date >= '{date_ini}'"
    if date_end:
        query += f" AND date <= '{date_end}'"

    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()

    df = pd.DataFrame(data, columns=['date', 'mod_values'])

    return df

def get_clusters(connector, country: str) -> list[int]:
    country_code = COUNTRY_CODES[country]
    cursor = connector.cursor()
    query = f"""SELECT DISTINCT cluster FROM stations WHERE station_id LIKE '{country_code}%' AND cluster IS NOT NULL"""
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()

    clusters = [row[0] for row in data]

    return clusters

def get_days(df: pd.DataFrame) -> list[dt.date]:
    days = df['date'].dt.date.unique()
    return list(days)

def plot_time_series(df_station: pd.DataFrame, df_model: pd.DataFrame, station: str, cluster: int, version: int):
    # Plot time series of observed and modelled values only for the hours in df_model, using broken x-axis
    if df_model.empty:
        return
    days = get_days(df_model)

    for day in days.copy():
        if df_station[df_station['date'].dt.date == day].empty:
            days.remove(day)
    
    if not days:
        return

    fig, axs = plt.subplots(1, len(days), figsize=(15, 5), sharey=True)

    for i, day in enumerate(days):
        df_day = df_station[df_station['date'].dt.date == day]
        df_model_day = df_model[df_model['date'].dt.date == day]

        df_merged = pd.merge(df_day, df_model_day, on='date', how='right')

        axs[i].plot(df_merged['date'], df_merged['obs_values'], label='Observed (EEA stations)', color='blue') # type: ignore
        axs[i].plot(df_merged['date'], df_merged['mod_values'], label='Modelled (AI)', color='red') # type: ignore

        ticks = [None]*24
        ticks[12] = day.strftime('%d-%m-%Y') # type: ignore
        axs[i].set_xticks(df_merged['date']) # type: ignore
        axs[i].set_xticklabels(ticks, rotation=45) # type: ignore

        handles, labels = axs[i].get_legend_handles_labels() # type: ignore

        # if i == 0 or i != len(days) - 1:
        #     axs[i].spines['right'].set_visible(False)
        # if i == len(days) - 1 or i != 0:
        #     axs[i].spines['left'].set_visible(False)
    
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    fig.suptitle(f'Comparison between model and observation in station ' + r"$\bf{" + f'{station}' + "}$" +
                 f' for the validation days\nCluster: {CLUSTER_NAMES[cluster]} - Model version: {version}')
    
    # Add a custom legend, with blue meaning Modelled and red meaning Observed
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.ylabel('NO2 concentration (µg/m³)')

    path = OUTPUT_PATH / str(version) / CLUSTER_NAMES[cluster]
    os.makedirs(path, exist_ok=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(path / f'{station}_{cluster}_{version}.png')

def main():
    conn = mysql.connect(host='127.0.0.1',
                         user='user',
                         password='password',
                         database='results')
    
    for version in model_versions_to_plot:
        for country in countries:
            country_data = get_country_data(country, 'NO2', date_ini=date_ini, date_end=date_end)

            clusters = get_clusters(conn, country)

            for cluster in clusters:
                stations = get_stations(conn, cluster)
                for station in stations:
                    station_data = get_station_data(country_data, station)
                    model_data = get_model_data(conn, version, station, date_ini, date_end)

                    plot_time_series(station_data, model_data, station, cluster, version)

if __name__ == "__main__":
    main()