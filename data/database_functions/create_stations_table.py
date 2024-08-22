# Connect to table "results" in database "127.0.0.1:3306". It is a mariadb database with user: "user" and password: "password".
# Then read file data/eea/stations.csv and insert the data into the table.
#
# The table should have the following columns:
# - station_id: int
# - station_name: varchar(255)
# - lat: float
# - lon: float
# - hgt: float
# - country: varchar(255)

import pandas as pd
import MySQLdb as mysql

# Connect to the database
conn = mysql.connect(host='127.0.0.1', user='user', password='password', database='results')

# Read the data
data = pd.read_csv('/home/urbanaq/data/eea/stations.csv')


# Insert the data into the table
cursor = conn.cursor()

# Create table if it does not exist
cursor.execute(
    "CREATE TABLE IF NOT EXISTS stations ("
    "station_id VARCHAR(255) PRIMARY KEY, "
    "station_name VARCHAR(255), "
    "lat FLOAT, "
    "lon FLOAT, "
    "hgt FLOAT, "
    "country VARCHAR(255)"
    ")"
)

# Get all unique "Air Quality Station EoI Code" values from the dataframe
for station in data["Air Quality Station EoI Code"].unique():
    # Get the first row with this station id
    row = data[data["Air Quality Station EoI Code"] == station].iloc[0]

    cursor.execute(
        "INSERT INTO stations (station_id, station_name, lat, lon, hgt, country) VALUES (%s, %s, %s, %s, %s, %s)",
        (row['Air Quality Station EoI Code'], row['Air Quality Station Name'], row['Latitude'], row['Longitude'], row['Altitude'], row['Country'])
    )

conn.commit()
cursor.close()
