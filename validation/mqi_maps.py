import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cams_downscaling.utils import get_db_connection


conn = get_db_connection()

model_versions = [100, 101, 102, 103, 104]

for model_version in model_versions:

    query = """
    SELECT s.station_id, s.station_name, s.lat, s.lon, v.MQI
    FROM validation v
    JOIN stations s ON v.station = s.station_id
    WHERE v.model_version = %s AND v.MQI IS NOT NULL;
    """

    data = pd.read_sql(query, conn, params=(model_version,))

    stations = pd.DataFrame(data, columns=['lat', 'lon', 'MQI'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 15), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.coastlines(resolution='10m')

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.viridis

    sc = ax.scatter(stations['lon'], stations['lat'], c=stations['MQI'], cmap=cmap, norm=norm, s=50, edgecolor='k', linewidth=0.5, transform=ccrs.PlateCarree(), label='Estaciones')

    ax.set_xlim(stations['lon'].min() - 2, stations['lon'].max() + 2)
    ax.set_ylim(stations['lat'].min() - 2, stations['lat'].max() + 2)

    ax.set_title(f'Model version: {model_version}')

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical')
    cbar.set_label('MQI')

    # Guardar el mapa en un archivo PNG
    plt.savefig(f'{model_version}.png', dpi=300)

    print(f'{model_version}.png')

# Cerrar la conexión
conn.close()
