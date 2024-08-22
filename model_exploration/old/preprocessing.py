import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline

def interpolate_grid(ds, new_lat, new_lon, grid=True):
    lat, lon = ds.lat, ds.lon

    Z = ds.values
    interpolator = RectBivariateSpline(x, y, Z, kx=1, ky=1)
    new_Z = interpolator(new_x, new_y, grid=grid).flatten()
    
    new_index = pd.MultiIndex.from_product([new_x, new_y], names=df.index.names)
    return pd.Series(new_Z, index=new_index)
    
def interpolate_data(df, new_x, new_y, grid=True):
    """
    Interpolates the given DataFrame to a new grid defined by new_x and new_y.

    Parameters:
    df (pd.DataFrame): The DataFrame to interpolate.
    new_x (array-like): The new x values at which to interpolate.
    new_y (array-like): The new y values at which to interpolate.
    grid (bool): Whether to interpolate the DataFrame to a grid or not.

    Returns:
    pd.DataFrame: The interpolated DataFrame.
    """
    levels = df.index.names

    if len(levels) > 2:
        # Unstack all levels except the last two, interpolate and stack back
        new_df = df.unstack(list(range(0, len(levels) - 2)))
        new_df = new_df.apply(interpolate_grid, new_x=new_x, new_y=new_y, grid=grid)
        new_df = new_df.stack(list(range(-len(levels) + 2, 0)), future_stack=True)
        return new_df.reorder_levels(levels).sort_index()
    
    elif len(levels) == 2:
        # Interpolate directly
        return interpolate_grid(df, new_x=new_x, new_y=new_y, grid=grid)
    
    else:
        raise ValueError("Dataframe must have at least 2 index levels")
