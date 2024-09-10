import cdsapi

dataset = "cams-europe-air-quality-forecasts-optimised-at-observation-sites"
request = {
    'variable': ['nitrogen_dioxide'],
    'country': ['italy', 'poland', 'spain'],
    'type': ['mos_optimised'],
    'leadtime_hour': ['0-23'],
    'year': ['2024'],
    'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
    'include_station_metadata': 'yes'
}

client = cdsapi.Client()
client.retrieve(dataset, request, f'/home/urbanaq/data/cams_optimized/no2.csv')
