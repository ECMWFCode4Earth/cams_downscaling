import rasterio
import concurrent.futures
import numpy as np
from pathlib import Path
from glob import glob
import os

base_path = Path('/data1/data_prep')
region = 'iberia'
vars = ['d2m', 'ssr', 't2m', 'tp', 'v10', 'u10']

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
    input_path = base_path / 'era5_land_old' / var / region
    output_path = base_path / 'era5_land' / var / region

    crear_dir(output_path)

    input_files = glob(str(input_path / '*00.tiff'))

    with concurrent.futures.ThreadPoolExecutor(2) as tp:
        results = []
        for input_file in input_files:
            file_name = os.path.basename(input_file)
            output_file = os.path.join(output_path, file_name)
            results.append(tp.submit(invertir_lats, input_file, output_file))
        
        for t in concurrent.futures.as_completed(results):
            print(f'Tiff creat: {t.result()}')
            results.remove(t)