import os
import yaml

import MySQLdb as mysql


def read_config(directory) -> dict:
    """Reads configuration files."""
    return {
        f[:-len('.yml')]: yaml.safe_load(open(os.path.join(directory, f)))
        for f in os.listdir(directory) if f.endswith('.yml')}

def get_db_connection(database="results") -> mysql.Connection:
    return mysql.connect(host='127.0.0.1', user='user', password='password', database=database)
