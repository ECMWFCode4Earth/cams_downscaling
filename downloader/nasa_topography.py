# Download NASA Shuttle Radar Topography Mission Global 1 arc second V003
import requests
import time

MAX_RETRIES = 10000
AWAIT_TIME = 30

params = {
    'AUTH': ('meteosim', '3LDGaYuxiaVHaW3@'),
    'lat': [-26.0, 34],
    'lon': [34, 72],
    'output_path': '/home/urbanaq/data/nasa_topo2'}

# Login to get the token
response = requests.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=params['AUTH'])
token = response.json().get('token')

# Submit the request
response = requests.post(
    'https://appeears.earthdatacloud.nasa.gov/api/task',
    params={
        'task_type': 'area',
        'task_name': 'SRTMGL1',
        'layer': 'SRTMGL3_NC.003,SRTMGL3_DEM',
        'start_date': '02-11-2000',
        'end_date': '02-21-2000',
        'bbox': f'{params["lon"][0]},{params["lat"][0]},{params["lon"][1]},{params["lat"][1]}',
        'file_type' :'geotiff',
        'projection_name':'geographic'},
    headers={'Authorization': 'Bearer {0}'.format(token)})

task_id = response.json().get('task_id')

# Check the status of the request
for _ in range(MAX_RETRIES):
    status = requests.get(
        'https://appeears.earthdatacloud.nasa.gov/api/status/{0}'.format(task_id),
        headers={'Authorization': 'Bearer {0}'.format(token)})

    print('Request status: {0}'.format(status.json().get('status')))

    if status.json().get('status') == 'done':
        # Download the data
        bundle = requests.get(
            'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}'.format(task_id),
            headers={'Authorization': 'Bearer {0}'.format(token)})
        
        for file in bundle.json()['files']:
            filename = file['file_name'].split('/')[-1]
            
            print('Downloading {0}...'.format(file['file_name']))
            download = requests.get(
                'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}'.format(task_id, file['file_id']),
                headers={'Authorization': 'Bearer {0}'.format(token)})

            with open(f"{params.get('output_path', '.')}/{filename}", 'wb') as f:
                f.write(download.content)

        print('Downloaded the data successfully!')
        break

    # Wait for the next check
    print('Request is not ready yet. Waiting for {0} seconds...'.format(AWAIT_TIME))
    time.sleep(AWAIT_TIME) 
