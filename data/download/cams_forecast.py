import cdsapi

region = "iberia"
area = [44.1, -9.9, 35.7,4.5,]

region = "italy"
area = [47.3, 6.7, 36.5, 18.6, ]

region = "poland"
area = [54.5, 13.9, 48.8, 24.3, ]

region ="europe"

print(f"Downloading NO2 forecast data for {region} region")

dataset = "cams-europe-air-quality-forecasts"
request = {
    'variable': ['nitrogen_dioxide'],
    'model': ['ensemble'],
    'level': ['0'],
    'date': ['2024-01-01/2024-09-06'],
    'type': ['forecast'],
    'time': ['00:00'],
    'leadtime_hour': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
    'data_format': 'grib',
    #'area': area
}

client = cdsapi.Client()
client.retrieve(dataset, request, f'/home/urbanaq/data/cams/2024/no2_{region}.grib')
