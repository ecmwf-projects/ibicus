# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Metrics module - Standard metric definitions"""

import attrs
import numpy as np
import pandas as pd

from pylab import arange
from scipy.ndimage import measurements

import matplotlib as plt

import seaborn

import math

@attrs.define(eq = False)
class Metric:
    key: str = attrs.field(default="unknown", validator=attrs.validators.instance_of(str))
    name: str = attrs.field(default="unknown", validator=attrs.validators.instance_of(str))
    variable: str = attrs.field(default="unknown", validator=attrs.validators.instance_of(str))
    threshold_value: float = attrs.field(default="unknown", validator=attrs.validators.instance_of(float), converter = float)
    threshold_sign: str = attrs.field(default=None, validator=attrs.validators.instance_of((str, type(None))))
    
    def calculate_instances_of_threshold_exceedance(self, dataset):
        
        """
        Converts np.ndarray of input data (observations or climate projections) into 1-0 np.ndarray of same dimensions based on
        threshold value and sign. Assignes 1 if value is below/above specified threshold (exceedance over threshold - eot), 0 otherwise.

        Parameters
        ----------
        dataset: np.ndarray
            Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected
        threshold_name: str
            Name of threshold metric specified in the metrics dictionary
        """

        instances_of_threshold_exceedance = np.copy(dataset)

        if self.threshold_sign == "higher":

            instances_of_threshold_exceedance = (instances_of_threshold_exceedance > self.threshold_value).astype(int)

        elif self.threshold_sign == "lower":

            instances_of_threshold_exceedance = (instances_of_threshold_exceedance < self.threshold_value).astype(int)

        else:
            raise ValueError('Invalid threshold sign. Modify threshold_sign to either higher or lower in class instantiation')

        return instances_of_threshold_exceedance
    
    
    def filter_threshold_exceedances(self, dataset):
        
        eot_matrix = np.copy(dataset)
        
        if self.threshold_sign == "higher":
            eot_matrix[eot_matrix < self.threshold_value] = 0
            
        elif self.threshold_sign == "lower":  
            eot_matrix[eot_matrix > self.threshold_value] = 0
            
        else:
            raise ValueError('Invalid threshold sign. Modify threshold_sign to either higher or lower in class instantiation')

        return eot_matrix
            
        
    def calculate_exceedance_probability(self, dataset):
        
        """
        Calculates the probability of exceeding a specified threshold at each location,
        building on the function calculate_matrix.
        
        Parameters
        ----------
        dataset: np.ndarray
            Input data, either observations or climate projectionsdataset to be analysed, numeric entries expected
        threshold_name: str
            Name of threshold metric specified in the metrics dictionary
        """
        
        threshold_data = self.calculate_instances_of_threshold_exceedance(dataset)

        threshold_probability = np.einsum('ijk -> jk', threshold_data)/threshold_data.shape[0]

        return threshold_probability
        
        
    def calculate_exceedances_per_time_period(self, dataset, time_period):
        print('to-do')
        
        
        
    def calculate_spell_length(self, minimum_length, **climate_data):
        
        spell_length_array = np.empty((0, 3))
    
        for k in climate_data.keys():
    
            threshold_data = self.calculate_instances_of_threshold_exceedance(climate_data[k])
            spell_length = np.array([])
    
            for i in range(threshold_data.shape[1]):
                for j in range(threshold_data.shape[2]):
                    N = 0
                    for t in range(threshold_data.shape[0]):
                        if threshold_data[t, i, j] == 1:
                            N = N + 1
                        elif (threshold_data[t, i, j] == 0) and (N != 0):
                            spell_length = np.append(spell_length, N)
                            N = 0
    
            spell_length = spell_length[spell_length > minimum_length]
    
            spell_length_array = np.append(
                spell_length_array,
                np.transpose(
                    np.array(
                        [
                            [k] * len(spell_length),
                            [self.name] * len(spell_length),
                            np.transpose(spell_length),
                        ]
                    )
                ),
                axis=0,
            )
    
        plot_data = pd.DataFrame(spell_length_array, columns=["Correction Method", "Metric", "Spell length (days)"])
        plot_data["Spell length (days)"] = pd.to_numeric(plot_data["Spell length (days)"])
    
        return plot_data
    
    
        
    def calculate_spatial_clusters(self, **climate_data):
        
        clusters_array = np.empty((0, 3))

        for k in climate_data.keys():

            spatial_count = np.array([])

            number_gridpoints = climate_data[k].shape[1] * climate_data[k].shape[2]

            threshold_data = self.calculate_instances_of_threshold_exceedance(dataset=climate_data[k])

            for i in range(threshold_data.shape[0]):

                count = np.sum(threshold_data[i, :, :]) / number_gridpoints
                spatial_count = np.append(spatial_count, count)

            spatial_count = spatial_count[spatial_count != 0]

            clusters_array = np.append(
                clusters_array,
                np.transpose(
                    np.array(
                        [
                            [k] * len(spatial_count),
                            [self.name] * len(spatial_count),
                            np.transpose(spatial_count),
                        ]
                    )
                ),
                axis=0,
            )

        spatial_clusters = pd.DataFrame(
            clusters_array, columns=["Correction Method", "Metric", "Spatial extent (% of area)"]
        )
        spatial_clusters["Spatial extent (% of area)"] = pd.to_numeric(spatial_clusters["Spatial extent (% of area)"])

        return spatial_clusters
        
        
        
    def calculate_spatiotemporal_clusters(self, **climate_data):
        
        clusters_array = np.empty((0, 3))

        for k in climate_data.keys():

            threshold_data = self.calculate_instances_of_threshold_exceedance(dataset=climate_data[k])
            threshold_data_lw, threshold_data_num = measurements.label(threshold_data)
            area = measurements.sum(threshold_data, threshold_data_lw, index=arange(threshold_data_lw.max() + 1))

            clusters_array = np.append(
                clusters_array,
                np.transpose(
                    np.array(
                        [[k] * len(area), [self.name] * len(area), np.transpose(area)]
                    )
                ),
                axis=0,
            )

        spatiotemporal_clusters = pd.DataFrame(
            clusters_array, columns=["Correction Method", "Metric", "Spatiotemporal cluster size"]
        )
        spatiotemporal_clusters["Spatiotemporal cluster size"] = pd.to_numeric(
            spatiotemporal_clusters["Spatiotemporal cluster size"]
        )

        return spatiotemporal_clusters
        
    def calculate_amount_over_threshold_per_time_period(self, dataset, aggregation, dates, time_period):
        
        # TO-DO
        
        if aggregation=='year':
            
            eot_matrix = self.filter_threshold_exceedances(dataset)

            count_length = math.floor(dates.get(time_period).shape[0]/365)

            annual_count = np.empty([])

            for count in range(count_length):
    
                start = count*365
                annual_count = np.append(annual_count, np.sum(eot_matrix[start:start+365, 0,0]))
            
            
    
    
    def plot_clusters_violinplots(self, minimum_length, **climate_data):
        
        """
        Takes pandas dataframes of temporal, spatial and spatiotemporal extent as input and outputs three violinplot
        comparing observational data to the raw climate and all debiasers specified in the dataframes. 

        Parameters
        ----------
        temporal_data: pd.DataFrame
            pandas dataframe of type output by function _calculate_spell_length
        spatial_data: pd.DataFrame
            pandas dataframe of type output by function _calculate_spatial_clusters
        spatiotemporal_data: pd.DataFrame
            pandas dataframe of type output by function _calculate_spatiotemporal_clusters

        """
        
        temporal_data = self.calculate_spell_length(minimum_length, **climate_data)
        spatial_data = self.calculate_spatial_clusters(**climate_data)
        spatiotemporal_data = self.calculate_spatiotemporal_clusters(**climate_data)

        fig, ax = plt.subplots(1, 3, figsize=(16, 6))

        seaborn.violinplot(
            ax=ax[0], data=temporal_data, x="Metric", y="Spell length (days)", palette="colorblind", hue="Correction Method"
        )
        ax[0].set_title("Spell length (days)")

        seaborn.violinplot(
            ax=ax[1],
            data=spatial_data,
            x="Metric",
            y="Spatial extent (% of area)",
            palette="colorblind",
            hue="Correction Method",
        )
        ax[1].set_title("Spatial extent (% of area)")

        seaborn.violinplot(
            ax=ax[2],
            data=spatiotemporal_data,
            x="Metric",
            y="Spatiotemporal cluster size",
            palette="colorblind",
            hue="Correction Method",
        )
        ax[2].set_title("Spatiotemporal cluster size")
    


      

# pr metrics
dry_days = Metric(name = 'Dry days (< 1 mm/day)', variable = 'pr', threshold_value = 1/86400, threshold_sign = 'lower')
wet_days = Metric(name = 'Wet days (> 1 mm/day)', variable = 'pr', threshold_value = 1/86400, threshold_sign = 'higher')
R10mm = Metric(name = 'Very wet days (> 10 mm/day)', variable = 'pr', threshold_value = 10/86400, threshold_sign = 'higher')
R20mm = Metric(name = 'Extremely wet days (> 20 mm/day)', variable = 'pr', threshold_value = 20/86400, threshold_sign = 'higher')

# tas metrics
warm_days = Metric(name = 'Warm days (mean)', variable = 'tas', threshold_value = 295, threshold_sign = 'higher')
cold_days = Metric(name = 'Cold days (mean)', variable = 'tas', threshold_value = 275, threshold_sign = 'lower') 

# tasmin metrics 
frost_days = Metric(name = 'Frost days (tasmin<0°C)', variable = 'tas', threshold_value = 273.13, threshold_sign = 'lower') 

      
