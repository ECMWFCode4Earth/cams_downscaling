import argparse
from pathlib import Path
import time

import cdsapi


MONTHS = [
    #['01', '02', '03', '04', '05', '06'],
    #['07', '08', '09', '10', '11', '12']
    ['07', '08']
]


def download(year: int, months: list):
    c = cdsapi.Client()

    start = time.perf_counter()

    print(f'Downloading half {year}...')

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                'boundary_layer_height', 'instantaneous_eastward_turbulent_surface_stress',
                'instantaneous_northward_turbulent_surface_stress'
            ],
            'year': str(year),
            'month': months,
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
            'area': [
                73, -19, 33, 34,
            ],
        },
        f'/home/urbanaq/data/era5/{args.year}/{months[0]}-{months[-1]}.grib')
    
    end = time.perf_counter()
    
    print(f'Finished downloading half {year} in {(end - start)/60} minutes.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int)
    args = parser.parse_args()

    Path(f'/home/urbanaq/data/era5/{args.year}').mkdir(parents=True, exist_ok=True)

    for months in MONTHS:
        download(args.year, months)
