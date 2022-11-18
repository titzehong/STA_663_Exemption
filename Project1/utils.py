from typing import List
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import pymc as pm
import pandas as pd 


def calc_time_months(month, year):
    
    year_conv = {2017:0,
                 2018:1,
                 2019:2}
    
    year_convert = year.apply(lambda x: year_conv[x])
    
    return month + year_convert*12

def conv_to_period(month, period=4):
    """
    period=6: biannual
    period=4: quarter
    period=2: bimonthly
    period=1: monthly
    """
    
    return month//4

def process_dates(date_series, period=4):
    month_series = date_series.apply(lambda x: x.month)
    year_series = date_series.apply(lambda x: x.year)

    month_chrono = calc_time_months(month_series, year_series)
    month_chrono = month_chrono - month_chrono.min()

    period_series = month_chrono.apply(lambda x: conv_to_period(x, period))
    
    return period_series




def create_comp_grid(lon_min: float,
                     lat_min: float,
                     horz_cells: int,
                     vert_cells: int,
                     cell_size: float):
    

    grid_cells = []


    x0_range = [lon_min+x*cell_size for x in range(horz_cells)]
    y0_range = [lat_min+x*cell_size for x in range(vert_cells)]

    for x0 in x0_range:
        for y0 in y0_range:
            # bounds
            x1 = x0-cell_size
            y1 = y0+cell_size
            grid_cells.append( box(x0, y0, x1, y1)  )

    comp_grid = gpd.GeoDataFrame(grid_cells, columns=['geometry']).set_crs(epsg=4326)
    comp_grid['Cell_id'] = list(range(len(comp_grid)))
    
    return comp_grid



def expand_comp_grid(comp_grid: gpd.GeoDataFrame,
                     time_ids: List[int]):
    """
    Expands computational grid so that it is cartesian product with time
    time_ids: List[int]: list of input time points
    """
    
    comp_grid_time = []
    for t in time_ids:

        t_grid = comp_grid.copy()
        t_grid['Time'] = t

        comp_grid_time.append(t_grid)

    comp_grid_time = pd.concat(comp_grid_time)
    
    return comp_grid_time

def create_kron_struct_single(comp_grid_crime: gpd.GeoDataFrame,
                       comp_grid: gpd.GeoDataFrame,
                       crime_type: str,
                       agg_vars: List[str]
                       ):
    
    comp_grid_crime_type = comp_grid_crime[comp_grid_crime['Primary Type']==crime_type]
    
    # Aggregate over cell id and period
    agg_crime_type = comp_grid_crime_type[agg_vars+['No Events']].groupby(agg_vars, as_index=False).sum()
    agg_crime_type['Cell_id'] = agg_crime_type['Cell_id'].astype('int')
    
    # Combine with comp grid to get lat lon
    if 'Time' not in agg_vars:
        kron_data_struct = comp_grid.merge(agg_crime_type,on='Cell_id',how='left')
    else:
        # Expand comp grid first
        comp_grid_expanded = expand_comp_grid(comp_grid, agg_crime_type['Time'].unique())
        kron_data_struct = comp_grid_expanded.merge(agg_crime_type,
                                                    on=['Cell_id','Time'],
                                                    how='left')

    
    kron_data_struct['No Events'] = kron_data_struct['No Events'].fillna(0)

    # Get coordinates
    kron_data_struct['coords'] = kron_data_struct['geometry'].apply(lambda x: x.representative_point().coords[:])
    kron_data_struct['coords'] = [coords[0] for coords in kron_data_struct['coords']]
    kron_data_struct['Lon'] = [x[0] for x in kron_data_struct['coords']]
    kron_data_struct['Lat'] = [x[1] for x in kron_data_struct['coords']]

    if 'Time' not in agg_vars:
        # Sort by Lat  then Lon so arrangement is in cartesian product of Lon and Lat
        kron_data_struct = kron_data_struct.sort_values(['Lon','Lat'])
        kron_lon = kron_data_struct['Lon'].unique()#.sort()
        kron_lat = kron_data_struct['Lat'].unique()#.sort()
        kron_lon.sort()
        kron_lat.sort()

        X1 = kron_lon[:, None]
        X2 =  kron_lat[:, None]
        cell_counts = kron_data_struct['No Events'].values
        correct_grid =  pm.math.cartesian(kron_lon[:, None], kron_lat[:, None])
        arrange_input_grid = kron_data_struct[['Lon','Lat']].values

        assert(np.all(correct_grid == arrange_input_grid))
        
        return kron_data_struct, [X1,X2], cell_counts

    else:
        kron_data_struct = kron_data_struct.sort_values(['Lon','Lat','Time'])
        kron_lon = kron_data_struct['Lon'].unique()#.sort()
        kron_lat = kron_data_struct['Lat'].unique()#.sort()
        kron_time = kron_data_struct['Time'].unique()
        kron_lon.sort()
        kron_lat.sort()
        kron_time.sort()

        X1 =  kron_lon[:, None]
        X2 =  kron_lat[:, None]
        X3 =  kron_time[:, None]

        cell_counts = kron_data_struct['No Events'].values
        correct_grid =  pm.math.cartesian(kron_lon[:, None], kron_lat[:, None], kron_time[:, None])
        arrange_input_grid = kron_data_struct[['Lon','Lat','Time']].values

        assert(np.all(correct_grid == arrange_input_grid))
        
        
        return kron_data_struct, [X1,X2,X3], cell_counts