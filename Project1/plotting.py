from typing import List
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import box
import geopandas as gpd
import pandas as pd

def plot_crime_by_district(crime_data: pd.DataFrame,
                           map_data: gpd.GeoDataFrame,
                           crime_types: List[str],
                           sub_date_min: pd.Timestamp,
                           sub_date_max: pd.Timestamp):

    chloro_map_data = crime_data[['Primary Type','Date', 'Community Areas']].copy()
    #sub_date_min = pd.to_datetime('2018-03-30')
    #sub_date_max = pd.to_datetime('2018-06-30')
    
    matplotlib.rcParams.update({'font.size': 24})
    
    n_crime_types = len(crime_types)
    
    fig, ax = plt.subplots(1,n_crime_types,figsize=(50,20))
    plt.rc('legend',fontsize=20)
    crime_stats = []
    for i, crime_type in enumerate(crime_types):

        chloro_data_sub = chloro_map_data[(chloro_map_data['Date'] > sub_date_min) & 
                        (chloro_map_data['Date'] < sub_date_max) & 
                                          (chloro_map_data['Primary Type']==crime_type)].copy()

        chloro_data_sub['No Events'] = 1
        chloro_plot = chloro_data_sub[['Community Areas',
                                       'No Events']].groupby('Community Areas',
                                                             as_index=False).sum()

        chloro_plot = map_data.merge(chloro_plot,
                                           on='Community Areas',
                                           how='left')

        chloro_plot.plot(column='No Events', ax=ax[i],
                         cmap='Reds',legend=False,edgecolor='black');

        crime_stats.append(chloro_plot)

        ax[i].set_axis_off()

        ax[i].title.set_text(f'No Of {crime_type} Observed')

        # Annotate with community no
        for idx, row in map_data.iterrows():
            ax[i].annotate(text=str(int(row['Community Areas'])), xy=row['coords'],
                         horizontalalignment='center', fontsize=12)



def plot_point_data(base_map: gpd.GeoDataFrame,
                    crime_geo_data: gpd.GeoDataFrame,
                    markersize: float=0.01):
    
    fig, ax = plt.subplots(figsize=(10,10))
    base = base_map.plot(color='white', edgecolor='black',ax=ax,alpha=0.5)
    crime_geo_data.plot(markersize=0.01, ax=base,alpha=0.5, color='red')
    ax.set_axis_off()



def plot_aggregated_crime_grid(comp_grid_crime: gpd.GeoDataFrame,
                               comp_grid: gpd.GeoDataFrame,
                               base_map: gpd.GeoDataFrame):
    """
    comp_grid_crime: crime information with events geo-tagged into cells
    comp_grid: dataframe with computation grid polygon info
    base_map: Map of chicago to use
    """

    agg_crime = comp_grid_crime[['Primary Type',
                                 'Cell_id',
                                 'No Events']].groupby(['Primary Type','Cell_id'],
                                                      as_index=False).sum().copy()

    agg_crime = gpd.GeoDataFrame(agg_crime.merge(comp_grid, on='Cell_id',how='left'))
    
    fig, ax = plt.subplots(1,3,figsize=(50,20))

    for i,crime_type in enumerate(['ASSAULT', 'BATTERY', 'THEFT']):

        base = base_map.plot(color='white', edgecolor='black',ax=ax[i])
        agg_crime_type = agg_crime[agg_crime['Primary Type']==crime_type]

        agg_crime_type.plot(column='No Events', ax=base,
                         cmap='Reds',legend=False,alpha=0.5);

        ax[i].set_axis_off()

        ax[i].title.set_text(f'No Of {crime_type} Observed')

        # Annotate with community no
        for idx, row in base_map.iterrows():
            ax[i].annotate(text=str(int(row['Community Areas'])), xy=row['coords'],
                         horizontalalignment='center', fontsize=12)
