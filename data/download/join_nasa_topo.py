import os
import zipfile
import rasterio
from rasterio.merge import merge
import numpy as np

def extract_zip_files(input_directory, temp_directory):
    """
    Extracts all ZIP files in the input_directory into the temp_directory.
    
    :param input_directory: Directory containing the ZIP files.
    :param temp_directory: Directory to temporarily store the extracted .hgt files.
    """
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    for item in os.listdir(input_directory):
        if item.endswith(".zip"):
            file_path = os.path.join(input_directory, item)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_directory)
                print(f'Extracted {item} into {temp_directory}')

def combine_hgt_files(temp_directory, output_file):
    """
    Combines all .hgt files in the temp_directory into a single output file.
    
    :param temp_directory: Directory containing the extracted .hgt files.
    :param output_file: File to store the combined output.
    """
    hgt_files = [os.path.join(temp_directory, f) for f in os.listdir(temp_directory) if f.endswith(".hgt")]
    datasets = [rasterio.open(f) for f in hgt_files]

    mosaic, out_trans = merge(datasets)

    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    for dataset in datasets:
        dataset.close()

    print(f'Combined .hgt files into {output_file}')

if __name__ == "__main__":
    input_directory = "/home/urbanaq/data/nasa_topo"  # Change to your directory containing ZIP files
    temp_directory = "/home/urbanaq/data/nasa_topo/kk"  # Change to a temporary directory for extracted files
    output_file = "/home/urbanaq/data/nasa_topo/output.tif"  # Change to your desired output file

    extract_zip_files(input_directory, temp_directory)
    combine_hgt_files(temp_directory, output_file)
    print("All .hgt files have been extracted and combined into a single output file.")
