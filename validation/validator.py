from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
from argparse import ArgumentParser
from datetime import datetime
import datetime as dt

import pandas as pd
import numpy as np

from cams_downscaling.utils import get_db_connection, read_config


config = read_config('/home/urbanaq/cams_downscaling/config')
DATA_PATH = Path(config['paths']['stations'])

POLLUTANTS = config["pollutants"]["pollutants"]

REF_VALUES = pd.DataFrame({
        "pollutant": ["NO2", "O3", "PM10", "PM2.5"],
        "U_r(RV)": [0.24, 0.18, 0.28, 0.36],
        "RV": [200, 120, 50, 25],
        "alpha": [0.20, 0.79, 0.25, 0.50],
        "N_p": [5.2, 11, 20, 20],
        "N_np": [5.5, 3, 1.5, 1.5]
    }).set_index('pollutant')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model_version", type=int,
                        help="Version of the ML model. 4-digit integer")
    parser.add_argument("--stations", type=str, nargs='*', default=[],
                        help="List of stations to be validated. Empty list means all")
    parser.add_argument("--countries", type=str, nargs='*', default=[],
                        help="List of countries to be validated. Empty list means all")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Initial date of the period to be validated. Format: YYYYMMDD")
    parser.add_argument("--until", dest="until_date", default=None,
                        help="Final date of the period to be validated. Format: YYYYMMDD")
    parser.add_argument("--only_mqi90th", action="store_true", help="If used, only MQI90 is calculated")
    # parser.add_argument("--target_diagram", action="store_true", help="If used, target diagram is calculated")
    args = parser.parse_args()

    if args.model_version is None:
        print("Error: --model_version is required.")
        sys.exit(1)

    #if not (1000 <= args.model_version <= 9999):
    #    print("Error: --model_version must be a 4-digit integer (between 100 and 999).")
    #    sys.exit(1)

    if args.from_date:
        try:
            args.from_date = dt.datetime.strptime(args.from_date, '%Y%m%d')
        except ValueError:
            print("Error: --from_date must be in the format YYYYMMDD.")
            sys.exit(1)

    if args.until_date:
        try:
            args.until_date = dt.datetime.strptime(args.until_date, '%Y%m%d')
        except ValueError:
            print("Error: --until_date must be in the format YYYYMMDD.")
            sys.exit(1)

    return args

def get_stations(connector, country: str) -> list[str]:
    cursor = connector.cursor()
    query = f"""SELECT station_id FROM stations WHERE station_id LIKE '{country}%'"""
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()

    stations = [row[0] for row in data]

    return stations

def get_country_data(country: str, pollutant: str, date_ini: dt.date, date_end: dt.date) -> pd.DataFrame:
    df = []
    for year in os.listdir(DATA_PATH):
        if os.path.isdir(os.path.join(DATA_PATH, year)):
            if f'{country}.csv' not in os.listdir(os.path.join(DATA_PATH, year)):
                continue
            csv_path = DATA_PATH / year / f"{country}.csv"
            df_year = pd.read_csv(csv_path, parse_dates=['time'])

            if date_ini:
                df_year_year = df_year[df_year['time'] >= datetime(year=int(year), month=1, day=1)]
            if date_end:
                df_year = df_year[df_year['time'] <= datetime(year=int(year), month=12, day=31)]
            
            df_year = df_year[['time', 'station', pollutant]].dropna()
            df_year = df_year.rename(columns={'time': 'dates', pollutant: 'obs_values'})

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

    df = pd.DataFrame(data, columns=['dates', 'mod_values'])

    return df

def obs_uncertainty(obs: np.ndarray, pollutant: str):
    Ur = REF_VALUES.loc[pollutant, "U_r(RV)"]
    rv = REF_VALUES.loc[pollutant, "RV"]
    alpha = REF_VALUES.loc[pollutant, "alpha"]

    uncertainty = Ur*np.sqrt((1-alpha**2) * obs**2 + alpha**2 * rv**2) # type: ignore

    return uncertainty

def calculate_MQI(
        model_data: pd.DataFrame, obs_data: pd.DataFrame, pollutant: str, date_ini: dt.date, date_end: dt.date
                  ) -> tuple[float | None, float | None, float | None, float | None, float | None]:

    if model_data.empty or obs_data.empty:
        return None, None, None, None, 0.0
    
    data = pd.merge(model_data, obs_data, on='dates', how='inner')

    obs = np.array(data['obs_values'].values)
    mod = np.array(data['mod_values'].values)
    uncert = obs_uncertainty(obs, pollutant)
    beta = 2 # Coefficient of proportionality

    mse = np.square(np.subtract(obs,mod)).mean()
    rmse = np.sqrt(mse) # Root Mean Square Error

    ms_u = np.square(uncert).mean()
    rms_u = np.sqrt(ms_u) # Root Mean Sqare of the uncertanty

    mqi = rmse/(beta * rms_u) # Modelling Quality Indicator (MQI)

    # Calculate fraction of hours used for validation with respect to the total hours in the period
    init_time = dt.datetime.combine(date_ini, dt.datetime.min.time())
    end_time = dt.datetime.combine(date_end, dt.datetime.max.time())
    hours_in_period = (end_time - init_time).total_seconds() / 3600
    fraction = len(data)/hours_in_period

    return rmse, rms_u, mqi, beta, fraction

def calculate_bias(model_data: pd.DataFrame, obs_data: pd.DataFrame) -> float | None:
    if model_data.empty or obs_data.empty:
        return None
    
    data = pd.merge(model_data, obs_data, on='dates', how='inner')

    obs = np.array(data['obs_values'].values)
    mod = np.array(data['mod_values'].values)
    
    bias = mod.mean() - obs.mean()

    return bias

def calculate_crmse(model_data: pd.DataFrame, obs_data: pd.DataFrame) -> float | None:
    if model_data.empty or obs_data.empty:
        return None
    
    data = pd.merge(model_data, obs_data, on='dates', how='inner')

    obs = np.array(data['obs_values'].values)
    mod = np.array(data['mod_values'].values)

    cumulative_error = np.subtract(mod-mod.mean(), obs-obs.mean())
    cmse = np.square(cumulative_error).mean()
    crmse = np.sqrt(cmse) # Cumulative Root Mean Square Error

    return crmse

def calculate_MQI90(connector, model_version, date_ini, date_end):
    cursor = connector.cursor()

    query = f"""
    SELECT MQI
    FROM validation
    WHERE MQI IS NOT NULL
    AND model_version = '{model_version}'
    AND time_ini = '{date_ini}'
    AND time_end = '{date_end}'
    """

    cursor.execute(query)
    results = cursor.fetchall()
    connector.commit()
    cursor.close()
    
    mqi_values = sorted([float(row[0]) for row in results])

    n = len(mqi_values)

    if n==1:
        mqi90 = mqi_values[0]*0.9
        return mqi90
    
    s90 = int(n * 0.9)
    dist = n * 0.9 - s90
    
    # Compute MQI_90th using linear interpolation
    mqi90 = mqi_values[s90-1] + (mqi_values[s90] - mqi_values[s90-1]) * dist
    
    return mqi90

def save_stats(connector, model_version: int, station: str,
               time_ini: dt.date, time_end: dt.date, rmse, rms_u, mqi, beta, fraction, bias, crmse):
    cursor = connector.cursor()

    if fraction == 0.0:
        return

    query = """
    REPLACE INTO validation (model_version, station, time_ini, time_end, MQI, fraction, RMSE, beta, RMS_U, bias, CRMSE)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    values = (model_version, station, time_ini, time_end, mqi, fraction, rmse, beta, rms_u, bias, crmse)

    print(query % values)
    cursor.execute(query, values)
    connector.commit()
    cursor.close()

    return

def main():
    args = parse_arguments()
    conn = get_db_connection()

    pollutant = POLLUTANTS[int(str(abs(args.model_version))[:1])] # Get pollutant from model version

    if not args.only_mqi90th:
        for country in args.countries:
            stations = get_stations(conn, country)
            obs_country = get_country_data(country, pollutant, args.from_date, args.until_date)

            for i, station in enumerate(stations):
                df_mod = get_model_data(conn, args.model_version, station, args.from_date, args.until_date)
                df_obs = get_station_data(obs_country, station)

                if df_mod.empty or df_obs.empty:
                    continue
                
                rmse, rms_u, mqi, beta, fraction = calculate_MQI(df_mod, df_obs, pollutant, args.from_date, args.until_date)
                bias = calculate_bias(df_mod, df_obs)
                crmse = calculate_crmse(df_mod, df_obs)

                print(f"\n\nbias^2 + CRMSE^2 = {bias**2 + crmse**2}") # type: ignore
                print(f"RMSE^2 = {rmse**2}\n") # type: ignore

                print(f"MQI^2 = {mqi**2}") # type: ignore
                print(bias**2/(beta*rms_u)**2 + crmse**2/(beta*rms_u)**2) # type: ignore

                print(f"Station {station} ({i+1}/{len(stations)})\nMQI = {mqi}\nusing {fraction*100: .1f}%\n") # type: ignore
                save_stats(conn, args.model_version, station, args.from_date, args.until_date,
                           rmse, rms_u, mqi, beta, fraction, bias, crmse)
    
    mqi90 = calculate_MQI90(conn, args.model_version, args.from_date, args.until_date)
    save_stats(conn, args.model_version, "90th", args.from_date, args.until_date,
               None, None, mqi90, 2.0, None, None, None)
    
    conn.close()

    return

if __name__ == "__main__":
    main()
