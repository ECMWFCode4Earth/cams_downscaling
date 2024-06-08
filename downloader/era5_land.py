import argparse
from pathlib import Path
import concurrent.futures
import time

import cdsapi


def download(year: int, month: int):

    c = cdsapi.Client()

    start = time.perf_counter()

    print(f'Downloading {month:02d}/{year}...')

    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'surface_net_solar_radiation', 'total_precipitation', '2m_dewpoint_temperature',
            ],
            'year': str(year),
            'month': f'{month:02d}',
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'format': 'grib',
            'area': [
                73, -19, 33, 34,
            ]
        },
        f'/home/urbanaq/data/era5_land/{args.year}/{month:02d}.grib')
    
    end = time.perf_counter()
    
    print(f'Finished downloading {month:02d}/{year} in {(end - start)/60} minutes.')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int)
    args = parser.parse_args()

    Path(f'/home/urbanaq/data/era5_land/{args.year}').mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(4) as tp:
        tasks = []

        print(f'Starting data download for {args.year} - 4 threads...')

        for month in range(1, 13):
            tasks.append(tp.submit(download, args.year, month))

        for task in concurrent.futures.as_completed(tasks):
            tasks.remove(task)

        print('All done! Goodbye!')
