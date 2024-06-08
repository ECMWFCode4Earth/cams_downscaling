ls
import requests
from typing import Generator, NamedTuple
import io
import logging
import zipfile
import datetime as dt
from argparse import ArgumentParser

import pandas as pd
import numpy as np



class Variable(NamedTuple):
    """
    Helper class to store variable information
    """
    code: str
    vocab_id: str
    has_negative_values: bool = False


ENDPOINT_URL = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile"
VOCABULARY_POLLUTANT_URL = "http://dd.eionet.europa.eu/vocabulary/aq/pollutant/"

# Available datasets
DATASET_REALTIME_UNVALIDATED = 1
DATASET_VALIDATED_E1A = 2           # 2013 to two years ago
DATASET_HISTORICAL_AIRBASE = 3      # 2002 to 2012

# Available variables
PM10 = Variable("PM10", 5)
PM25 = Variable("PM2.5", 6001)
NO2 = Variable("NO2", 8)
NO = Variable("NO", 38)
SO2 = Variable("SO2", 1)
O3 = Variable("O3", 7)

# What we need for each country
VARIABLES    = {
    'ES': [C6H6, PM10, CO, NO2, NOX_AS_NO2, NO, SO2, O3, PM25],
    'PT': [C6H6, PM10, CO, NO2, SO2, O3, PM25],
    'AD': [PM10, CO, NO2, NOX_AS_NO2, NO, SO2, O3]
}


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def download_parquet(country: str, vars: list[Variable], dataset: int) -> zipfile.ZipFile:
    """
    Downloads the ZIP file that contains all the Parquet files for the given country and variables
    """
    request_body = {
        "countries": [country],
        "cities": [],
        "properties": list(map(lambda var: f"{VOCABULARY_POLLUTANT_URL}{var.vocab_id}", vars)),
        "datasets": [dataset],
        "source": "Api"
    }

    req = requests.post(ENDPOINT_URL, json=request_body)

    if req.status_code != 200:
        raise Exception(f"Error downloading data: {req.status_code}")
    
    return zipfile.ZipFile(io.BytesIO(req.content))


def read_zip_file(zip_parquet: zipfile.ZipFile, data_start: dt.datetime | pd.Timestamp, data_end: dt.datetime | pd.Timestamp) -> pd.DataFrame:
    """
    Reads the Parquet files from the ZIP file and returns a DataFrame with all the data.
    We will only read the data from the given date onwards (optional).
    """
    logger = logging.getLogger(__name__)

    all_files = zip_parquet.infolist()
    all_df = []
    
    for f in all_files:
        df = pd.read_parquet(io.BytesIO(zip_parquet.read(f)), 
                             columns=["Samplingpoint", "Pollutant", "Start", "Value", "Validity"])
        if data_start:
            df = df[df["Start"] >= data_start]
        if data_end:
            df = df[df["Start"] <= data_end]
        all_df.append(df)
    del zip_parquet

    logger.info("Read all files!")

    df = pd.concat(all_df)
    del all_df

    logger.info("Concatenated all files!")

    return df
    


def replace_pollutant_id(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Replaces the pollutant id (int) with the corresponding code (ObsDB column name)
    """
    pollutants = df["Pollutant"].unique()
    pollutants_replacement = {}
    for pollutant in pollutants:
        # pollutant is an int, corresponding to Variable.vocab_id. we search for the corresponding Variable.code
        var = next((var for var in VARIABLES[country] if var.vocab_id == pollutant), None)
        if not var:
            raise ValueError(f"Unknown pollutant id: {pollutant}")
        pollutants_replacement[pollutant] = var.code

    df["Pollutant"] = df["Pollutant"].cat.rename_categories(pollutants_replacement)

    return df


def replace_station_id(df: pd.DataFrame, country: str, keep_stations: list[str] | None) -> pd.DataFrame:
    """
    Replaces the Sampling Point Id with the corresponding Air Quality Station EoI Code.

    It also checks if new stations appear in the data that are still not available in the ObsDB.

    It will only keep the stations in keep_stations, if provided.
    """
    stations = pd.read_csv(f'{country}_stations.csv', 
                           usecols=["Air Quality Station EoI Code", "Sampling Point Id"],
                           index_col="Sampling Point Id").squeeze("columns")
    stations.index = f"{country}/" + stations.index

    # .replace() introduces some overhead and is slower than .map()
    df["Samplingpoint"] = df["Samplingpoint"].map(stations.to_dict().get)

    return df


def cleanup_negative_values(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    We will replace negative values with None. Only variables that cannot have them.
    """
    for var in VARIABLES[country]:
        if not var.has_negative_values:
            df.loc[(df["Value"] < 0) & (df["Pollutant"] == var.code), "Value"] = None

    return df


def pivot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with all the data for a country, pivot it so we have a DataFrame with the data for each station.
    Columns will be the pollutants and the values will be the measurements.
    """
    df_pivot = df.pivot(index=["Samplingpoint", "time"], columns="Pollutant", values="Value")
    del df

    # MySQL does not support NaN values, so we will replace them with None
    df_pivot.replace({np.nan: None}, inplace=True)

    return df_pivot


def get_station_data(df: pd.DataFrame) -> Generator[tuple[str, pd.DataFrame], None, None]:
    """
    Given a DataFrame with the data for a country, yield the data for each station.
    It cleans up the data by removing the station from the DataFrame to reduce memory usage.
    """

    for station in df.index.get_level_values(0).unique():
        data = df.loc[station].reset_index()
        df.drop(station, inplace=True)
        yield station, data


def parse_country(country: str, 
                  data_start: dt.datetime | pd.Timestamp | None, 
                  data_end: dt.datetime | pd.Timestamp | None,
                  stations: list[str] | None = None, 
                  dataset: int = DATASET_REALTIME_UNVALIDATED):
    """
    Downloads the data for the given country, cleans it up and stores it in the ObsDB.
    """

    logger = logging.getLogger(__name__)

    logger.info(f"{'*' * 4} Processing country {country} {'*' * 4}")

    zip_parquet = download_parquet(country, VARIABLES[country], dataset)

    logger.info("Downloaded all the data!")

    df = read_zip_file(zip_parquet, data_start, data_end)
    del zip_parquet

    df = df[df["Validity"] == 1]
    df = df.drop(columns="Validity")
    df["Value"] = df["Value"].astype(float)
    df["Pollutant"] = df["Pollutant"].astype("category")
    df = replace_pollutant_id(df, country)
    df = replace_station_id(df, country, stations)
    df = cleanup_negative_values(df, country)
    df["Samplingpoint"] = df["Samplingpoint"].astype("category")
    

    #####################################################################################################
    #                           VERY IMPORTANT INFORMATION  
    # According to the docs, the date for HOURLY variables (Parquet: AggType) is in UTC+1
    # The date for DAILY variables depends on the country
    # Since we only have hourly variables, we will use the "Start" column as the date, since it will be
    # the "End" column in UTC
    #####################################################################################################
    df.rename(columns={"Start": "time"}, inplace=True)

    logger.info("Data cleaned up!")

    df = pivot_df(df)


    logger.info("Data restructured! Ready to store in the ObsDB.")

    total_stations = len(df.index.get_level_values(0).unique())
    current_station = 1
    for station, df_station in get_station_data(df):
        logger.info(f"[{current_station}/{total_stations}] Processing station {station}")

        continue  
        
        current_station += 1

    logger.info(f"{'*' * 4} Finished processing {country} {'*' * 4}")


def main():
    logger = logging.getLogger(__name__)

    parser = ArgumentParser()
    parser.add_argument("--days", type=int, default=None, help="Number of days to update")
    parser.add_argument("--since", type=str, default=None, help="First date (start of day - UTC). Format: YYYY-MM-DD")
    parser.add_argument("--until", type=str, default=None, help="Last date (end of day - UTC). Format: YYYY-MM-DD")
    parser.add_argument('--stations', nargs='*', default=None, type=str, help="List of stations to update")
    parser.add_argument('--dataset', type=str, default='DATASET_REALTIME_UNVALIDATED', choices=['DATASET_REALTIME_UNVALIDATED', 'DATASET_VALIDATED_E1A', 'DATASET_HISTORICAL_AIRBASE'], help="Dataset to download")
    args = parser.parse_args()

    dataset = globals()[args.dataset]

    if args.since and args.days:
        raise ValueError("You cannot use both --since and --days")
    
    data_start = None
    if args.days:
        data_start = pd.to_datetime(dt.date.today() - pd.Timedelta(days=args.days))
    elif args.since:
        try:
            dt.datetime.strptime(args.since, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Incorrect date format. Use YYYY-MM-DD")
        
        data_start = pd.to_datetime(f"{args.since} 00:00:00")

    data_end = None
    if args.until:
        try:
            dt.datetime.strptime(args.until, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Incorrect date format. Use YYYY-MM-DD")
        
        data_end = pd.to_datetime(f"{args.until} 23:59:59")

    stations = args.stations

    logger.info("Starting download of EEA data")

    for country in VARIABLES.keys():
        parse_country(country, data_start, data_end, stations, dataset)



if __name__ == "__main__":
    setup_logger()
    main()
