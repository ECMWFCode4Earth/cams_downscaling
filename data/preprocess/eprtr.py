""" 
Reads the KML file with the E-PRTR data and extracts the coordinates of the points.
"""
from pathlib import Path

import pandas as pd
from fastkml import kml

DATA_FOLDER = "/home/urbanaq/data/eprtr"

with open(str(Path(DATA_FOLDER) / "doc.kml"), 'rb') as f:
    k = kml.KML()
    k.from_string(f.read())


all_points = []

for doc in k.features():
    for sector in doc.features():
        for category in sector.features():
            if not category:
                continue
            try:
                for poi in category.features():
                    lon, lat = poi.geometry.x, poi.geometry.y
                    all_points.append((sector.name, category.name, lon, lat))
            except:
                try:
                    lon, lat = category.geometry.x, category.geometry.y
                    all_points.append((sector.name, None, lon, lat))
                except:
                    continue



df = pd.DataFrame(all_points, columns=["sector", "category", "lon", "lat"])
df.to_csv(str(Path(DATA_FOLDER) / "points.csv"), index=False)
