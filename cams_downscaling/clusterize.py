# TODO: do we need to read the data and drop the stations with NaN values?

import pandas as pd
from sklearn.cluster import DBSCAN

from .utils import read_config, get_db_connection
from .readers.eea import load_eea_data


countries = ['Spain', 'Portugal']
region = 'iberia'

CONFIG = read_config('/home/urbanaq/aquv/config')
COUNTRY_CODES = CONFIG['places']['country_codes']

epsilons = {
    'Spain': 0.1,
    'Portugal': 0.1,
    'Andorra': 0.1,
    'Italy': 0.1,
    'Poland': 0.3,
}

country_prefixes = {
    'Spain': '10',
    'Portugal': '20',
    'Andorra': '30',
    'Italy': '40',
    'Poland': '50',
}


def list_dataframes_by_country(dataframe: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    # Take a dataset and split it by country, using the first two letters of the index station
    return [(country, dataframe[dataframe.index.str.startswith(COUNTRY_CODES[country])]) for country in countries]

def clusterize(stations: pd.DataFrame) -> pd.DataFrame:
    '''The following commented code was ran only once.
    Then, the following code was used to get the clusters from the SQL database'''

    # Define the clustering model
    df_all = []
    for country, df_country in list_dataframes_by_country(stations):
        dbscan = DBSCAN(eps=epsilons[country], min_samples=5)
        # Fit the model
        clusters = dbscan.fit_predict(df_country[['Longitude', 'Latitude']])
        df_country.loc[:, 'cluster'] = clusters
        df_country = df_country[df_country.cluster != -1]
        df_country.cluster = df_country.cluster.apply(lambda x: float(f'{country_prefixes[country]}{x}'))
        df_all.append(df_country)

    # Get a single dataframe with all the countries
    stations = pd.concat(df_all)

def update_db(stations: pd.DataFrame):
    conn = get_db_connection()
    cursor = conn.cursor()

    for _, row in stations.iterrows():
        update_query = """
        UPDATE stations
        SET cluster = %s
        WHERE station_id = %s
        """
        data = (row['cluster'], row.name)
        cursor.execute(update_query, data)

    conn.commit()

    cursor.close()
    conn.close()

    print("Clusters updated in SQL")

def main():
    data_path = CONFIG['data_path']
    stations, _ = load_eea_data('NO2', [2019], countries, CONFIG['bbox'][region], data_path)
    stations = clusterize(stations)
    update_db(stations)