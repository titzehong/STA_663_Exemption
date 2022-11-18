import pandas as pd
import numpy as np
from typing import List
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt 
import matplotlib

from utils import *

class Dataloader:
    
    def __init__(self, data_path: str):
        self.full_data = pd.read_csv(data_path)
        
        self._get_base_stats(self.full_data)
        
        # Set some variables
        self.crime_types = self.unique_crimes
        self.wanted_areas = None
        
        
    def _get_base_stats(self, full_data: pd.DataFrame):
        """ Function that prints some simple data statistics

        Args:
            full_data (pd.DataFrame): The full dataset
        """
        
        self.min_date = pd.to_datetime(full_data['Date'].max())
        self.max_date = pd.to_datetime(full_data['Date'].max())
        self.unique_crimes = full_data['Primary Type'].unique()
        
        print("Max Date: ", self.min_date)
        print("Min Date: ", self.max_date)
        print("Types of Unique Crimes: ", len(self.unique_crimes))
        
    def subset_data_crime(self, crime_types: List[str],
                          date_min: pd.Timestamp,
                          date_max: pd.Timestamp,
                          del_full: bool=True):
        """ Subsets the full crime dataset to a specified crime type and date range.

        Args:
            crime_types (List[str]): List of input crimes to subset
            date_min (pd.Timestamp): Start date
            date_max (pd.Timestamp): End date
            del_full (bool, optional): Whether to delele the full data to clear memory. Defaults to True.
        """
        
        self.crime_types = crime_types
        
        subset_data = self.full_data[self.full_data['Primary Type'].isin(crime_types)]

        subset_data['Date'] = pd.to_datetime(subset_data['Date'])

        subset_data = subset_data[(subset_data['Date']>=date_min) &
                                  (subset_data['Date']<=date_max)]
        
        subset_data = subset_data.rename(columns={'Date':'Datetime'})
        subset_data['Date'] = subset_data['Datetime'].apply(lambda x: x.date())
        self.subset_data = subset_data

        # Weirdly there are two columns for community areas, 'Community Area' and 'Community Areas'
        # Community Area should be the correct one but since we have used 'Community Areas' we just replace
        # column
        subset_data['Community Areas'] = subset_data['Community Area'].copy()
        
        
        if del_full:
            del self.full_data
        
        # Convert to a GPD DataFrame
        crimes_geo = subset_data[['Primary Type',
                                          'Date',
                                          'Community Areas',
                                          'Latitude',
                                          'Longitude']]

        crimes_geo = gpd.GeoDataFrame(crimes_geo,
                             geometry=gpd.points_from_xy(crimes_geo.Longitude,
                                                         crimes_geo.Latitude)).set_crs(epsg=4326)
        
        # Drop a weirdly anomalous point
        crimes_geo = crimes_geo[crimes_geo['Latitude']>38]
        
        self.subset_data = subset_data
        self.crimes_geo = crimes_geo

            
    def clean_data(self):
        """ Removes NAs from the dataset
        """
        na_mask = (self.subset_data['Longitude'].isna()) | \
                  (self.subset_data['Latitude'].isna()) | \
                  (self.subset_data['Date'].isna())
        
        print("No. NA: ",np.sum(na_mask))
        
        self.subset_data = self.subset_data[~na_mask]
        
    
    def map_import(self, shp_path_communities: str, plot: bool=True,fig_size=(10,10)):
        """Loads the .shp files and plots them. 

        Args:
            shp_path_communities (str): Filepath of the .shp file of community boundaries
            plot (bool, optional): Whether to plot. Defaults to True.
            fig_size (tuple, optional): Figure size of plots. Defaults to (10,10).
        """

        # By Communities
        df_communities = gpd.read_file(shp_path_communities).to_crs({'init': 'epsg:4326'})
        df_communities['Community Areas'] = df_communities['area_num_1'].astype('float')
        
        # For plotting names into Communities
        df_communities['coords'] = df_communities['geometry'].apply(lambda x: x.representative_point().coords[:])
        df_communities['coords'] = [coords[0] for coords in df_communities['coords']]
        
        if plot:
            fig, ax = plt.subplots(figsize=fig_size)
            df_communities['geometry'].plot(ax=ax,edgecolor='dimgray',color='white')
            
            for idx, row in df_communities.iterrows():
                ax.annotate(text=str(int(row['Community Areas'])), xy=row['coords'],
                             horizontalalignment='center', fontsize=8)

            community_names = dict(zip(df_communities['Community Areas'].astype('int'),
                               df_communities['community']))
            
            ax.set_axis_off()
            
        self.df_communities = df_communities
        self.community_names = community_names
        
    
    def subset_map(self, wanted_areas: List[int], plot=True):
        """ Subsets the maps based on chosen areas (ids) in wanted_areas.

        Args:
            wanted_areas (List[int]): List of community area ids (1-77) that we want for the analysis
            plot (bool, optional): Plots the map to check chosen areas. Defaults to True.
        """
        self.wanted_areas = wanted_areas
        subset_communities = self.df_communities[self.df_communities['Community Areas'].isin(wanted_areas)]
        
        if plot:
            fig, ax = plt.subplots(figsize=(10,10))
            subset_communities['geometry'].plot(ax=ax,edgecolor='dimgray',color='white')

            for idx, row in subset_communities.iterrows():
                ax.annotate(text=str(int(row['Community Areas'])), xy=row['coords'],
                             horizontalalignment='center', fontsize=8)

            ax.set_axis_off()
        
        self.subset_df_communiies =  subset_communities
        
    def subset_data_geo(self):
        """Simple function to subset data by community (Note this is different from subset map which subsets the base map)
        """
        crimes_geo_subset = self.crimes_geo.copy()
        self.crimes_geo_subset = crimes_geo_subset[\
                                crimes_geo_subset['Community Areas'].isin(self.wanted_areas)]
        
        
        
    def time_bucketing(self, date_series:pd.Series, period:int=4):
        """Puts the dates into time indexes based on the level of aggregation
         quarterly=4, bimonthly=2,  monthly=1 

        Args:
            date_series (pd.Series): Input series of dates
            period (int, optional): The time periods (refer to description). Defaults to 4.

        Returns:
            _type_: _description_
        """
        
        return process_dates(date_series, period)

