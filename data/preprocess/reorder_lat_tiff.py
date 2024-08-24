""" 
This file is useful to flip the lat.tiff file after running the followinf scripts:

- `era5land_reorder_coords.py`
- `era5_reorder_coords.py`
"""
from pathlib import Path

import rasterio
import numpy as np


tiff_path = Path('/data1/data_prep/era5_land/lat/poland.tiff')


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


invertir_lats(tiff_path, tiff_path)
