import MySQLdb as mysql

# Connect to the database
conn = mysql.connect(host='127.0.0.1', user='user', password='password', database='results')

# Get all the station_id from the station table
cursor = conn.cursor()
cursor.execute("SELECT station_id FROM stations")
stations = cursor.fetchall()
cursor.close()

# For each station_id, create a table with the following columns:
# - date: datetime
# - model_version: int
# - NO2: float4
cursor = conn.cursor()
for station in stations:
    # cursor.execute(
    # f"DROP TABLE IF EXISTS `{station[0].decode()}`"
    # )
    
    # cursor.execute(
    #     f"CREATE TABLE `{station[0].decode()}` ("
    #     "date DATETIME, "
    #     "model_version INT, "
    #     "value FLOAT"
    #     ")"
    # )

    cursor.execute(
        f"ALTER TABLE `{station[0].decode()}` ADD CONSTRAINT date_model_version PRIMARY KEY (date, model_version)"
    )

    #     cursor.execute(
    #     f"CREATE TABLE IF NOT EXISTS `{station[0].decode()}` ("
    #     "date DATETIME, "
    #     "model_version INT, "
    #     "value FLOAT"
    #     ")"
    # )
        
cursor.close()
conn.commit()
conn.close()
