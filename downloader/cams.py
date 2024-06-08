import argparse
from pathlib import Path
import concurrent.futures
import time

import cdsapi


VARIABLES =  [
    'nitrogen_dioxide', 'nitrogen_monoxide',
    'ozone', 'particulate_matter_10um', 'particulate_matter_2.5um',
    'sulphur_dioxide'
]


def download(year: int, month: int, variable: str):

    c = cdsapi.Client()

    print(f'Downloading {variable} for {month:02d}/{year}...')

    start = time.perf_counter()

    c.retrieve(
        'cams-europe-air-quality-reanalyses',
        {
            'variable': [
                variable
            ],
            'model': 'ensemble',
            'level': '0',
            'year': [
                str(year)
            ],
            'month': [
                f'{month:02d}'
            ],
            'format': 'tgz',
            'type': [
                'interim_reanalysis', 'validated_reanalysis'
            ]
        },
        f'/home/urbanaq/data/cams/{year}/{month:02d}-{variable}.tar.gz'
        )
    
    end = time.perf_counter()
    
    print(f'Finished downloading {variable} for {month:02d}/{year} in {(end - start)/60} minutes.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int)
    args = parser.parse_args()

    Path(f'/home/urbanaq/data/cams/{args.year}').mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(4) as tp:
        tasks = []

        print(f'Starting data download for {args.year} - 4 threads...')

        for month in range(1, 13):
            for variable in VARIABLES:
                tasks.append(tp.submit(download, args.year, month, variable))

        for task in concurrent.futures.as_completed(tasks):
            tasks.remove(task)

        print('All done! Goodbye!')
