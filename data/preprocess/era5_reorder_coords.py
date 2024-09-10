""" 
The result of `era5_to_tiff.py` generates tiffs with flipped latitudes. This script inverts the latitudes of the tiffs to match the original ERA5 data.

However, you'll still need to reorder the lat.tiff file using the `reorder_lat_tiff.py`.
The reason this is not done here is to only have one lat.tiff and lon.tiff for the ERA5-Land data. The tiff generator creates one
for each variable. Then, we manually need to get one of them and put them in one folder.
"""
import concurrent.futures
from pathlib import Path
from glob import glob
import os
import gc

import rasterio
import numpy as np


base_path = Path('/data1/data_prep')
region = 'poland'
vars = ['iews', 'inss', 'blh']


def invertir_lats(input_file: str, output_file:str) -> str:
    with rasterio.open(input_file) as src:
        num_bandes = src.count

        inverted_bands = []
        for i in range(1, num_bandes + 1):
            data = src.read(i)
            inverted_data = np.flipud(data) # Flip up/down, perquè només les lats estaven decreixents
            inverted_bands.append(inverted_data)
        
        profile = src.profile
        with rasterio.open(output_file, 'w', **profile) as dst:
            for i in range(num_bandes):
                dst.write(inverted_bands[i], i + 1)
    
    return output_file


def crear_dir(path_dir):
    path = Path(path_dir)
    if not path.exists():
        path.mkdir(parents=True)
        print(f"\nDirectori creat: {path_dir}")


for var in vars:
    input_path = base_path / 'era5_old' / var / region
    output_path = base_path / 'era5' / var / region

    crear_dir(output_path)

    input_files = sorted(glob(str(input_path / '*00.tiff')))
    files_created = 0

    with concurrent.futures.ProcessPoolExecutor(4) as tp:
        results = []
        for input_file in input_files:
            file_name = os.path.basename(input_file)
            output_file = os.path.join(output_path, file_name)
            results.append(tp.submit(invertir_lats, input_file, output_file))
        
        for t in concurrent.futures.as_completed(results):
            files_created += 1
            print(f'Tiff creat ({files_created}/{len(input_files)}): {t.result()}', end='\r')
            results.remove(t)
            if files_created % 100 == 0:
                gc.collect()
        print()
