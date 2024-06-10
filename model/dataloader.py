import xarray as xr

def load_cams(cams_path):
    """Read CAMS data from a file."""
    cams = xr.open_dataset(cams_path)
    cams = cams.rename({'latitude': 'lat', 'longitude': 'lon'})
    cams = cams.sel(lat=slice(70, 30), lon=slice(-10, 40))
    return cams

if __name__ == '__main__':
    cams_path = '/home/urbanaq/data/cams/no2/data.nc'
    cams = load_cams(cams_path)
    print(cams.head())
