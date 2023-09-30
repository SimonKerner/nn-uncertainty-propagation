# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:24:32 2023

@author: Selii
"""

import os
import sys
import pickle

from tqdm import tqdm

from pathlib import Path
import utils
import data_visualizations as dvis
from dataset_loader import load_dataframe
from dataset_loader import load_miss_dataframe
from model_loader import create_binary_model
from model_loader import load_binary_model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#import seaborn as sns
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import mean_squared_error as mse
#import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#from sklearn.metrics import classification_report

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import seaborn as sns
#import statsmodels.api as sm

import scipy
import scipy.stats as stats
from scipy import interpolate
from scipy.special import ndtr


##########################################################################################################################
"""
        # set important paths
"""
##########################################################################################################################


# set path to different folders

#image_path = os.path.join(os.getcwd(), 'images')
_results_path = os.path.join(os.getcwd(), 'sim_results')




##########################################################################################################################
"""
        information about the datasets:
            -[1] wdbc - all attributes are considered continious 
            -[2] climate_simulation - all attributes are considered continious 
            -[3] australian - attributes are mixed between continious and discret
    
        following all the different settings for this simulation run can be found
            -dataset = "choose dataset"
            -standardize_dataset = "used for standardizing the dataset -- values between 0 and 1 -- minmax"
"""
##########################################################################################################################

# set random state          
_RANDOM_STATE = 42


#choose working dataset: choose one of the datasets above
_dataset = "australian"
_simulate_test_set = False


# other constants
_INIT_DATA_BANDWIDTH = None
_PRED_BANDWIDTH = None # --> if None (default) "scott" is used


# further dataset settings
_standardize_data = True


# settings for visualization
_visiualize_data = True
_visualize_original_predictions = True
_visualize_imputed_predictions = True
_visualize_simulated_predictions = True


# train or load model
_train_model = False
_save_new_model = False
_load_old_model = True


# load uncertain dataset // if True, an already created one will be loaded
_load_dataframe_miss = True
_create_dataframe_miss = True

# metrics for new uncertain dataframe creation // "static" (amount of values in each row) // "percentage" (value between 0 and 1, randomly)
_DELETE_MODE = "static"     
_MISS_RATE = 3


# Visual KDE_VALUES of each column of a data set (these will be used for continious sampling) 
_compare_col_kde_distributions = True
_compare_col_kde_mode = "combined"    # choose between "single", "combined", "both" // combined can only be used when _visualize data == true
_normalize_kde= True # setting this to false could break the plots


# deterministic imputation of missing values
_IMPUTE = False

# stochastic imputation (simulation) of uncertain data
_SIMULATE = True

# mode of simulation // Monte Carlo Sampling or Latin Hypercube Sampling available
_monte_carlo = False
_latin_hypercube = True
_LHS_MODE = "fast" # makes differnce in cdf computation // recomended -> fast
_visualize_lhs_samples = False


# further Simulation metrics
_SIMULATION_LENGTH = 5
#_SIMULATION_RANGE = None
_SIMULATION_RANGE = range(0, 5, 1) # if set to None -- all rows will be simulated


_simulation_visualizations = False


_save_simulated_results = False
_load_simulated_results = False
_load_results_id = 0


##########################################################################################################################
"""
        # load original datasets with full data
"""

DATAFRAME_ORIGINAL, datatype_map = load_dataframe(_dataset, _standardize_data)
_column_names = DATAFRAME_ORIGINAL.columns
_unique_outcomes = len(DATAFRAME_ORIGINAL.Outcome.unique())

    

##########################################################################################################################
"""
    # visiualize true underlying data of Dataframe 
"""


if _visiualize_data:
    
    # Plotting combined distribution using histograms
    _hist = dvis.plot_dataframe(DATAFRAME_ORIGINAL, _column_names, "Dataframe Orig. without missing data")

    
    """
    # Visualizing correlation between variables using a heatmap
    corr_matrix = DATAFRAME.iloc[:,:-1].corr()
    plt._figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
    plt.tight_layout()
    plt.show()
    """
    
    


##########################################################################################################################
"""
        # choose frame mode and perform train - test - split
"""

    
X_original = DATAFRAME_ORIGINAL.iloc[:, 0:-1]
y_original = DATAFRAME_ORIGINAL[_column_names[-1]]


_X_original_train, _X_original_test, _y_original_train, _y_original_test = train_test_split(X_original, 
                                                                                            y_original, 
                                                                                            test_size=0.25,
                                                                                            random_state=_RANDOM_STATE)


##########################################################################################################################
"""
        # create standard vanilla feed forward feural network
"""


if _train_model: 
    
    model = create_binary_model(_dataset, _X_original_train, _y_original_train, 
                                _X_original_test, _y_original_test, _save_new_model)


##########################################################################################################################
"""
        # swap full dataset to test dataset and reset indices
"""


if _simulate_test_set:
    
    """
        # SWAP-Action:
            if this option is chosen, the calculations will be made for the test part of the dataset 
            -- for simplicity of not chaning the script, DATAFRAME_ORIGINAL will be ste to the ORIGINAL_DATAFRAME
    """
    
    #DATAFRAME_ORIGINAL_FULL_VALUES = DATAFRAME_ORIGINAL.copy()
    
    X_original = _X_original_test.reset_index(drop=True)
    y_original = _y_original_test.reset_index(drop=True)
    
    DATAFRAME_ORIGINAL = X_original.merge(y_original, left_index=True, right_index=True)
    
    # visualize the test set again:
        
    if _visiualize_data:
        
        _hist = dvis.plot_dataframe(DATAFRAME_ORIGINAL, _column_names, 'Dataframe Test without missing data')
        
    

##########################################################################################################################
"""
        # load model without training a new one
"""


if _load_old_model:
    
    # loading and compiling saved model structure
    model = load_binary_model(_dataset)
    



##########################################################################################################################
"""
        #   RESULTS // Original Precictions

        # in the following block, all the standard deterministic predictions on the original dataset can be inspected
        # singe prediction metrics with a perfectly trained model - no uncertainties -- deterministic as usual
"""


print("\nPredictions for complete Dataset without uncertainties:")

original_metrics = {}

_orig_start_sample_time = time.time()

original_metrics["y_hat"] = model.predict(X_original).flatten()

_orig_end_sample_time = time.time() - _orig_start_sample_time


original_metrics["y_hat_labels"] = (original_metrics["y_hat"]>0.5).astype("int32")
original_metrics["input_rmse"] = mse(DATAFRAME_ORIGINAL, DATAFRAME_ORIGINAL)
#original_metrics["rmse_to_original_label"] = mse(y_original, original_metrics["y_hat"])
original_metrics["pred_time"] = _orig_end_sample_time
original_metrics["roc_auc"] = roc_auc_score(y_original, original_metrics["y_hat"])
original_metrics["prc_auc"] = average_precision_score(y_original, original_metrics["y_hat"])


if _visualize_original_predictions: 
    
    dvis.plot_binary_predictions(original_metrics["y_hat"], original_metrics["y_hat_labels"], 
                                 title='Original (True) combined predictions')


if _visualize_imputed_predictions:
    
    dvis.roc_curves(y_original, [original_metrics])
    plt.show()       
      
    
    dvis.pre_recall_curve(y_original, [original_metrics])
    plt.show()     

    
print("\nOriginal Classification Metrics:")
original_metrics["original_statistics"] = utils.create_metrics(y_original, original_metrics["y_hat_labels"], print_report=False)




##########################################################################################################################
"""
        # Here in this step a new DATAFRAME is introduced. 
        # This contains missing data with a specific missing rate in each row
"""


DATAFRAME_MISS = load_miss_dataframe(_dataset, DATAFRAME_ORIGINAL, _MISS_RATE, _DELETE_MODE, _RANDOM_STATE,
                                     _load_dataframe_miss, _create_dataframe_miss, _simulate_test_set)


if _visiualize_data:
      
    dvis.plot_dataframe(DATAFRAME_MISS, _column_names, 
                        'Input without missing data')
    
    dvis.plot_frame_comparison(data={"DATAFRAME_ORIGINAL" : np.array(DATAFRAME_ORIGINAL.iloc[:,:-1]).flatten(), 
                                     "DATAFRAME_MISS" : np.array(DATAFRAME_MISS.iloc[:,:-1]).flatten()},
                               title='Original & Uncertain dataset as flattened histplot')




##########################################################################################################################
##########################################################################################################################
"""
        # experiments modul 1 -deterministic predictions on imputed dataframs 
        (missing data replaced with above impute metrics)
"""

"""
        various imputation methodsfor missing data for each simulation run can be tested
        - if True (with imputation):
            -choose between KDE_imputer (self), SimpleImputer and KNN_Imputer
        - if False (without imputation): 
            -DATAFRAME_IMPUTE will be equal to DATAFRAME_MISS
        
        
        > Further explanation:
            - with imputation: This method can be used to fill missing values inside of the dataset 
                               before further usage
                - uncertainties of miss data will be imputed with values from the above methods 
                               (can be considered deterministic)
                - withoud imputation:
                    - this method will not fill the missing values of the dataset, instead it can be 
                      used for stochastic simulation, with MonteCarlo methods - propagation of uncertainties 
                      is guranteed 
"""
##########################################################################################################################
##########################################################################################################################

    
if _IMPUTE:
    
    """
        # mean imputation
    """
    _mean_start_sample_time = time.time()
    
    _mean_imp = SimpleImputer(strategy="mean")
    _DATAFRAME_MEAN_IMPUTE = pd.DataFrame(_mean_imp.fit_transform(DATAFRAME_MISS.copy()), columns=_column_names)
    
    _mean_end_sample_time = time.time() - _mean_start_sample_time

    
    """
        # median imputation
    """
    _median_start_sample_time = time.time()
    
    _median_imp = SimpleImputer(strategy="median")
    _DATAFRAME_MEDIAN_IMPUTE = pd.DataFrame(_median_imp.fit_transform(DATAFRAME_MISS.copy()), columns=_column_names)

    _median_end_sample_time = time.time() - _median_start_sample_time

    
    """
        # mode imputation
    """
    _mode_start_sample_time = time.time()
    
    _mode_imp = SimpleImputer(strategy="most_frequent")
    _DATAFRAME_MODE_IMPUTE = pd.DataFrame(_mode_imp.fit_transform(DATAFRAME_MISS.copy()), columns=_column_names)
    
    _mode_end_sample_time = time.time() - _mode_start_sample_time
    

    """
        # knn imputation
    """
    _knn_start_sample_time = time.time()
    
    _knn_imp = KNNImputer(n_neighbors=5)
    _DATAFRAME_KNN_IMPUTE = pd.DataFrame(_knn_imp.fit_transform(DATAFRAME_MISS.iloc[:,:-1].copy()), columns=_column_names[:-1], index=X_original.index)
    _DATAFRAME_KNN_IMPUTE = _DATAFRAME_KNN_IMPUTE.merge(DATAFRAME_ORIGINAL["Outcome"], left_index=True, right_index=True)
    
    _knn_end_sample_time = time.time() - _knn_start_sample_time
    
    
    """
        # multiple imputation technique --> MICE Algo.
    """
    _iter_start_sample_time = time.time()
    
    _iter_imp = IterativeImputer(max_iter=10, random_state=_RANDOM_STATE)
    _DATAFRAME_ITER_IMPUTE = pd.DataFrame(_iter_imp.fit_transform(DATAFRAME_MISS.iloc[:,:-1].copy()), columns=_column_names[:-1], index=X_original.index)
    _DATAFRAME_ITER_IMPUTE = _DATAFRAME_ITER_IMPUTE.merge(DATAFRAME_ORIGINAL["Outcome"], left_index=True, right_index=True)

    _iter_end_sample_time = time.time() - _iter_start_sample_time
    
    time.sleep(1)
    
    
    _DATAFRAME_IMPUTE_COLLECTION = {"MEAN_IMPUTE" : _DATAFRAME_MEAN_IMPUTE,
                                    "MEDIAN_IMPUTE" : _DATAFRAME_MEDIAN_IMPUTE,
                                    "MODE_IMPUTE" : _DATAFRAME_MODE_IMPUTE,
                                    "KNN_IMPUTE" : _DATAFRAME_KNN_IMPUTE,
                                    "ITER_IMPUTE" : _DATAFRAME_ITER_IMPUTE}
    
    _IMPUTE_TIMES = {"MEAN_IMPUTE" : _mean_end_sample_time,
                     "MEDIAN_IMPUTE" : _median_end_sample_time,
                     "MODE_IMPUTE" : _mode_end_sample_time,
                     "KNN_IMPUTE" : _knn_end_sample_time,
                     "ITER_IMPUTE" : _iter_end_sample_time}
    
    _IMPUTE_RMSE = {"MEAN_IMPUTE" : mse(DATAFRAME_ORIGINAL, _DATAFRAME_MEAN_IMPUTE, squared=False),
                     "MEDIAN_IMPUTE" : mse(DATAFRAME_ORIGINAL, _DATAFRAME_MEDIAN_IMPUTE, squared=False),
                     "MODE_IMPUTE" : mse(DATAFRAME_ORIGINAL, _DATAFRAME_MODE_IMPUTE, squared=False),
                     "KNN_IMPUTE" :  mse(DATAFRAME_ORIGINAL, _DATAFRAME_KNN_IMPUTE, squared=False),
                     "ITER_IMPUTE" : mse(DATAFRAME_ORIGINAL, _DATAFRAME_ITER_IMPUTE, squared=False)}
    

    print("\nPredictions for dataset with uncertainties and imputed values:")
    
    impute_metrics = {}
    
    for _frame_key in _DATAFRAME_IMPUTE_COLLECTION:
        
        print(f"Calculating results for dataframe: {_frame_key}")
        
        # create input frame for model predictions
        _X_impute = _DATAFRAME_IMPUTE_COLLECTION[_frame_key].iloc[:, 0:-1]
         
        # get results of prediction
        
        _start_pred_time = time.time()
        _y_impute_hat = model.predict(_X_impute).flatten()
        _end_pred_time = time.time() - _start_pred_time
        
        time.sleep(1)
        
        _y_impute_hat_labels = (_y_impute_hat>0.5).astype("int32")             
        
        impute_metrics[_frame_key] = {"input_rmse" : _IMPUTE_RMSE[_frame_key],
                                      "y_hat" : _y_impute_hat,
                                      "y_hat_labels" : _y_impute_hat_labels,
                                      #"rmse_to_model_prediction" : mse(original_metrics["y_hat"], _y_impute_hat),
                                      #"rmse_to_original_label" : mse(y_original, _y_impute_hat),
                                      "sample_time" : _IMPUTE_TIMES[_frame_key],
                                      "pred_time" : _end_pred_time,
                                      "roc_auc" : roc_auc_score(y_original, _y_impute_hat),
                                      "prc_auc" : average_precision_score(y_original, _y_impute_hat)
                                      }
        
        
        if _visualize_imputed_predictions:
            
            # visualize predictions
            dvis.plot_binary_predictions(_y_impute_hat, _y_impute_hat_labels,
                                         f'Uncertain imputated dataframe combined output - Miss-Rate: {_MISS_RATE} - Impute-Method: {_frame_key.replace("_IMPUTE", "")}')
       
        
            
            print(f"\n{_frame_key} Imputed Classification Metrics:")
        
        impute_metrics[_frame_key]["impute_statistics"] = utils.create_metrics(y_original, _y_impute_hat_labels, print_report=False)




    if _visualize_imputed_predictions:
        
        dvis.roc_curves(y_original, impute_metrics)
        plt.show()       
          
        
        dvis.pre_recall_curve(y_original, impute_metrics)
        plt.show()       
        
    
    


    
"""    
def categorical_latin_hypercube_sampler(dataframe, key, sim_length, random_state):
    
    
    
    
    dataframe = DATAFRAME_ORIGINAL 
    key = "Attribute: 4"
    sim_length=1000000#_SIMULATION_LENGTH
    random_state = _RANDOM_STATE
    
    # get frame column wit categorical data 
    column = dataframe.loc[:,key]
    
    # get unique values and normalize to probabilities // nan values get deleted
    unique = column.value_counts(normalize=True).sort_index()
    
    # get sorted categories & probabilities
    categories = np.array(list(unique.index))
    probabilities = unique.values
    
    
    # create cummulative probabilities
    cum_probs = np.cumsum(probabilities)
    
    # TODO


    # sample in 1-dimension with specific simulation length
    lhs_sampler = stats.qmc.LatinHypercube(1, seed=random_state)
    lhs_sample = lhs_sampler.random(n=sim_length) 

    # scale the created lhs samples to min and max cdf values
    lhs_sample_scaled = stats.qmc.scale(lhs_sample, min(cum_probs), max(cum_probs)).flatten()

    # create a histogram, where the edges of the bins are corresponding to the cummulative probabilities
    # fill in with scaled lhs samples
    
    
    
    hist, _ = np.histogram(column, np.linspace(0, 5, 20))

    bin_midpoints = (bins[:-1] + bins[1:])/2
    


    
    value_bins = np.searchsorted(cum_probs, lhs_sample_scaled)

    random_from_cdf = bin_midpoints[value_bins]
    
    plt.hist(random_from_cdf, density=True)


    plt.hist(column, density=True) 
    
    
        
    return None#generate_samples
"""     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

##########################################################################################################################
# experiments module 2 -- col wise simulations ----------> get kde values of dataframe
##########################################################################################################################

"""
        DISCLAIMER: DATAFRAME_SIMULATE is equal to DATAFRAME_MISS (including missing values) - naming because
"""

if _SIMULATE:
    
    print("\nPredictions for dataset with uncertainties and simulated values:")
    
    _DATAFRAME_SIMULATE = DATAFRAME_MISS.copy()
    _SIMULATE_METHOD = "KDE_Simulation"


    """
            KDE COLLECTION -- ORIGINAL 
            --> is equal to the true distribution of the underlying data of the specific dataset
            --> to be able to get the true distribution we will use the original dataset with full certain data and no missing values
    """

    kde_collection_original = utils.kde_collection_creator(DATAFRAME_ORIGINAL, _column_names, _INIT_DATA_BANDWIDTH)
    
       
    """
            KDE COLLECTION -- UNCERTAIN 
            --> is equal to the uncertain distribution of the underlying data of the specific dataset with missing values
            --> for the uncertain distribution we will use the dataset including missing data (=uncertain data) 
            --> for computing, all of the missing data has to be dropped first, to retrieve the uncertain distribution of the rest
    """
    
    kde_collection_uncertain = utils.kde_collection_creator(_DATAFRAME_SIMULATE, _column_names, _INIT_DATA_BANDWIDTH)


    if _compare_col_kde_distributions: 
    
        
        if _compare_col_kde_mode == "single" or _compare_col_kde_mode == "both":
            
            dvis.column_wise_kde_plot(DATAFRAME_ORIGINAL, _DATAFRAME_SIMULATE, "Original Distribution", 
                                      "Uncertain Distribution", _column_names, _MISS_RATE, _SIMULATE_METHOD,
                                      _INIT_DATA_BANDWIDTH)
            
                
        elif (_compare_col_kde_mode == "combined" or _compare_col_kde_mode == "both") and _visiualize_data==True:
    
            """
                This upcoming plot provides an overall overview over all the single column kde distributuion in a single plot, 
                instead of a single plot for each column
            """
        
            dvis.combined_col_kde_plot(DATAFRAME_ORIGINAL, _DATAFRAME_SIMULATE, "Original Distribution", 
                                      "Uncertain Distribution", _hist, _column_names, _INIT_DATA_BANDWIDTH)
 
        else: 
            
            print("Error in chosen col-kde comparison mode!")


    

    # step 0 --> first for loop for getting row and simulating this specific row
    
    if _SIMULATION_RANGE == None:
        _SIMULATION_RANGE = range(len(_DATAFRAME_SIMULATE))
        
    
    #x-axis ranges from 0 and 1 with .001 steps -- is also used for sigmoid accuracy
    # x-axis can be interpreted as sigmoid values between 0 and 1 with above mentioned steps (accuracy)
    if _SIMULATION_LENGTH < 10000: _x_axis = np.arange(0.0, 1.0, 0.000005)
    elif _SIMULATION_LENGTH <= 50000: _x_axis = np.arange(0.0, 1.0, 0.0005)
    else: _x_axis = np.arange(0.0, 1.0, 0.001)
    
    _scaler = MinMaxScaler()
    

    

    """
        in the next step (for-loop) the main simulation part is carried out
            - the for loop will itterate through all rows inside a given dataset
            - in each cycle two main simulation predictions will be creaded
                - one given a true distribution (kde)
                - and one given a uncertain distribution (kde)
            
    """
    
    _main_sim_start_time = time.time()
    
    
    # simulation row results is holding all the row wise informaiton form the simulation
    # index is equal to collected row results 
    SIMULATION_ROW_RESULTS = []
    

    _adjusted_random_state = _RANDOM_STATE

    for _row in tqdm(_SIMULATION_RANGE):
        
        """
            # step 1: get current row to perform simulation with
        """
        _DATAFRAME_SIMULATE_ROW = pd.DataFrame(_DATAFRAME_SIMULATE.loc[_row])
        
        """
            # step 2: find all the attributes with nan values and save to variable as keys for the kde-dictionary
        """
        
        _uncertain_attributes = np.where(_DATAFRAME_SIMULATE_ROW.isna().all(axis=1))[0]
        _uncertain_attributes = [_column_names[i] for i in _uncertain_attributes]
    
    
        """
            # step 3: sample a value from the specific kde of the missing value - aka. beginning of MonteCarlo Simulation
            # --> and safe sampled values for this row in a history
            
            &
            
            # step 4: create DATAFRAME for faster simulation (basis input) and replace missing values with sampled ones   
            # index length of DATAFRAME_MISS_ROW is now equal to number of simulations
            

            step 5: main predictions on collected samples/data
        """
        
        
        """
        # step 5.1.a: row-wise predictions on uncertain samples
            # -----> Simulation procedure for uncertain kde induced simulation frames
        """
        
        # for each increament in the simulated row, a different random state will be used
        if _adjusted_random_state == None: pass
        else:  _adjusted_random_state+=1
        
        _uncertain_sim_row_metrics = utils.binary_sample_and_predict(model = model, 
                                                                     simulation_row = _DATAFRAME_SIMULATE_ROW,
                                                                     #original_input_row = _original_df_mc_row_input,
                                                                     #original_input_row_outcome =_original_df_mc_row_outcome,
                                                                     dataframe_categorical = _DATAFRAME_SIMULATE, 
                                                                     uncertain_attributes = _uncertain_attributes, 
                                                                     
                                                                     standardize_data = _standardize_data, 
                                                                     datatype_map = datatype_map, 
                                                                     column_names = _column_names,
                                                                     
                                                                     simulation_length = _SIMULATION_LENGTH, 
                                                                     random_state = _adjusted_random_state, 
                                                                     
                                                                     monte_carlo = _monte_carlo,
                                                                     kde_collection = kde_collection_uncertain,
                                                                     normalize_kde = _normalize_kde,
                                                                     bw_method = _PRED_BANDWIDTH,
                                                                     x_axis = _x_axis,
                                                                     
                                                                     latin_hypercube = _latin_hypercube,
                                                                     lhs_mode = _LHS_MODE, 
                                                                     visualize_lhs_samples = _visualize_lhs_samples, 
                                                                     lhs_prefix=" Uncertain"
                                                                     )
        
        
        """
                #step 5.1.b: row-wise predictions on original samples
                    -----> Simulation procedure for true original kde induced simulation frames
        """
        
        # for each increament in the simulated row, a different random state will be used
        if _adjusted_random_state == None: pass
        else:  _adjusted_random_state+=3
        
        _original_sim_row_metrics = utils.binary_sample_and_predict(model = model, 
                                                                    simulation_row = _DATAFRAME_SIMULATE_ROW,
                                                                    #original_input_row = _original_df_mc_row_input,
                                                                    #original_input_row_outcome = _original_df_mc_row_outcome,
                                                                    dataframe_categorical = DATAFRAME_ORIGINAL, 
                                                                    uncertain_attributes = _uncertain_attributes, 
                                                                    
                                                                    standardize_data = _standardize_data, 
                                                                    datatype_map = datatype_map, 
                                                                    column_names = _column_names,
                                                                     
                                                                    simulation_length = _SIMULATION_LENGTH, 
                                                                    random_state = _adjusted_random_state, 
                                                                     
                                                                    monte_carlo = _monte_carlo,
                                                                    kde_collection = kde_collection_original,
                                                                    normalize_kde = _normalize_kde,
                                                                    bw_method = _PRED_BANDWIDTH,
                                                                    x_axis = _x_axis,
                                                                     
                                                                    latin_hypercube = _latin_hypercube,
                                                                    lhs_mode = _LHS_MODE, 
                                                                    visualize_lhs_samples = _visualize_lhs_samples, 
                                                                    lhs_prefix=" Original"
                                                                    )
        
        
        # append simulation row results
        SIMULATION_ROW_RESULTS.append({"0_Overall_Row_Data" : {"0.1_row_id" : _row,
                                                               "0.2_dataset" : _dataset,
                                                               "0.3_miss_rate" : _MISS_RATE,
                                                               "0.4_miss_rate_%" : round(_DATAFRAME_SIMULATE_ROW.isnull().sum().sum() * 100 / len(_DATAFRAME_SIMULATE_ROW[:-1]),2),
                                                               "0.5_Simulation_length" : _SIMULATION_LENGTH,
                                                               "0.6_Simulated_row" : _DATAFRAME_SIMULATE_ROW,
                                                               "0.7_uncertain_attributes" : _uncertain_attributes,
                                                               "0.8_monte_carlo" : _monte_carlo,
                                                               "0.9_latin_hypercube" : _latin_hypercube
                                                               },
                                       "Original_Simulation" : _original_sim_row_metrics,
                                       "Uncertain_Simulation" : _uncertain_sim_row_metrics
                                       })

    # some accessory time metrics for comparison 
    _main_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - _main_sim_start_time))  
    print('\n\nSimulation execution time:', _main_elapsed_time)

    
    print('\n\nCreated checkpoint before postprocessing!')
    # checkpoint:
    _results_file_name = os.path.join(_results_path, "postprocess_checkpoint_"+_dataset + "_" + str(_MISS_RATE) + "_" + str(_SIMULATION_RANGE))
    
    pickle.dump({"SIMULATION_ROW_RESULTS" : SIMULATION_ROW_RESULTS}, open(_results_file_name, "wb"))

    
    
    """
            Post-Processing ROW-Metrics: # Simulation Results postprocess with list comprehensions for faster execution
    """
    
    print('\n\nStart of results postprocessing procedure!')
    _postprocess_start_time = time.time()
    
    # calculate further metrics for each row of the simulation, to gain deeper insights into each simulated row
    SIMULATION_ROW_RESULTS, SIMULATION_MEAN_RESULTS = utils.calculate_simulation_results(DATAFRAME_ORIGINAL, SIMULATION_ROW_RESULTS, _SIMULATION_LENGTH,
                                                                                          _SIMULATION_RANGE, _PRED_BANDWIDTH, _normalize_kde)
    
    _postprocess_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - _postprocess_start_time))  
    print('\n\Postprocess execution time:', _main_elapsed_time) 
    
    
    
    
    
    
    # visualizations for binary simulation // comparison plots
    if _simulation_visualizations:
        
        for _row in _SIMULATION_RANGE:
            
            # get last item of SIMULATION_ROW_RESULTS list and plot results
            dvis.simulation_hist_plot(SIMULATION_ROW_RESULTS, y_original, original_metrics, plotmode="autosearch", row=_row)
        
            dvis.simulation_kde_plot(_x_axis, SIMULATION_ROW_RESULTS, y_original, original_metrics, impute_metrics, plotmode="autosearch", row=_row)
    
        
    
    
        
    if _visualize_simulated_predictions:
        
        # reference found in data_visualizations 
        names = ["Orig_Mean", "Orig_Median", "Orig_Mode", "Uncert_Mean", "Uncert_Median", "Uncertain_Mode"]
        
        _enumeration = enumerate([SIMULATION_MEAN_RESULTS["Original_Mean"], SIMULATION_MEAN_RESULTS["Original_Median"], SIMULATION_MEAN_RESULTS["Original_Mode"],
                     SIMULATION_MEAN_RESULTS["Uncertain_Mean"], SIMULATION_MEAN_RESULTS["Uncertain_Median"], SIMULATION_MEAN_RESULTS["Uncertain_Mode"]])
        
        for _i, key in _enumeration:
    
            fpr, tpr, thresholds = roc_curve(y_original[_SIMULATION_RANGE], key)
    
            # AUC score that summarizes the ROC curve
            roc_auc = roc_auc_score(y_original[_SIMULATION_RANGE], key)
            
            plt.plot(fpr, tpr, lw = 2, label = names[_i] + ' ROC AUC: {:.2f}'.format(roc_auc))
            
        plt.plot([0, 1], [0, 1],
                 linestyle = '--',
                 color = (0.6, 0.6, 0.6),
                 label = 'random guessing')
        plt.plot([0, 0, 1], [0, 1, 1],
                 linestyle = ':',
                 color = 'black', 
                 label = 'perfect performance')
            
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('Receiver Operator Characteristic')
        
            
        plt.legend(loc = "lower right", fontsize=9)
        plt.tight_layout()   
        plt.show()
          
        
        
        # reference found in data_visualizations 
        for i, key in enumerate([SIMULATION_MEAN_RESULTS["Original_Mean"], SIMULATION_MEAN_RESULTS["Original_Median"], SIMULATION_MEAN_RESULTS["Original_Mode"],
                     SIMULATION_MEAN_RESULTS["Uncertain_Mean"], SIMULATION_MEAN_RESULTS["Uncertain_Median"], SIMULATION_MEAN_RESULTS["Uncertain_Mode"]]):
            
            precision, recall, thresholds = precision_recall_curve(y_original[_SIMULATION_RANGE], key)
            
            # AUC score that summarizes the precision recall curve
            avg_precision = average_precision_score(y_original[_SIMULATION_RANGE], key)
            
            plt.plot(recall, precision, lw = 2, label = names[i] + ' PRC AUC: {:.2f}'.format(avg_precision))
            
        plt.xlabel('Recall')  
        plt.ylabel('Precision')  
        plt.title('Precision Recall Curve')
        plt.legend(loc = "lower left", fontsize=9)
        plt.tight_layout()
        plt.show()
    








"""
        OVERALL RESULTS COLLECTION:
"""       


# exit if statement if no further simulations will be made
if _IMPUTE == False and _SIMULATE == False:
    sys.exit()

else:     
                
       
    # time is calculated as time needed per sampled/predicted row
    _sim_mean_times = pd.Series(data={"Uncertain_Sample_Time": np.mean([i["Uncertain_Simulation"]["sample_time"] for i in SIMULATION_ROW_RESULTS]),
                                                        "Original_Sample_Time": np.mean([i["Original_Simulation"]["sample_time"] for i in SIMULATION_ROW_RESULTS]),
                                                        "Uncertain_Prediction_Time": np.mean([i["Uncertain_Simulation"]["prediction_time"] for i in SIMULATION_ROW_RESULTS]),
                                                        "Original_Prediction_Time": np.mean([i["Uncertain_Simulation"]["prediction_time"] for i in SIMULATION_ROW_RESULTS]),           
                                                        })  
       
    
    
    
    # calculate the Input RMSE between the original DataFrame and the Uncertain DataFrames    
    DATAFRAME_INPUT_ANALYSIS = pd.Series(data={"Mean_Impute_df" : impute_metrics["MEAN_IMPUTE"]["input_rmse"],
                                               "Mode_Impute_df" : impute_metrics["MODE_IMPUTE"]["input_rmse"],
                                               "Median_Impute_df" : impute_metrics["MEDIAN_IMPUTE"]["input_rmse"],
                                               "KNNImp_Impute_df" : impute_metrics["KNN_IMPUTE"]["input_rmse"],
                                               "IterImp_Impute_df" : impute_metrics["ITER_IMPUTE"]["input_rmse"],
                                               "Uncertain_Mean_Sim_Input_RMSE" : SIMULATION_MEAN_RESULTS["Uncertain_Simulation_Input_RMSE"],
                                               "Original_Mean_Sim_Input_RMSE" : SIMULATION_MEAN_RESULTS["Original_Simulation_Input_RMSE"],
                                               }, name="RMSE")
    
    
    
    
    DATAFRAME_TIME_ANALYSIS = pd.DataFrame(data=[[0, original_metrics["pred_time"]],
                                                 [impute_metrics["MEAN_IMPUTE"]["sample_time"], impute_metrics["MEAN_IMPUTE"]["pred_time"]],
                                                 [impute_metrics["MODE_IMPUTE"]["sample_time"], impute_metrics["MODE_IMPUTE"]["pred_time"]],
                                                 [impute_metrics["MEDIAN_IMPUTE"]["sample_time"], impute_metrics["MEDIAN_IMPUTE"]["pred_time"]],
                                                 [impute_metrics["KNN_IMPUTE"]["sample_time"], impute_metrics["KNN_IMPUTE"]["pred_time"]],
                                                 [impute_metrics["ITER_IMPUTE"]["sample_time"], impute_metrics["ITER_IMPUTE"]["pred_time"]],
                                                 [_sim_mean_times["Uncertain_Sample_Time"], _sim_mean_times["Uncertain_Prediction_Time"]],
                                                 [_sim_mean_times["Original_Sample_Time"], _sim_mean_times["Original_Prediction_Time"]]
                                                 ], 
                                           index=["Original Model", "Mean Imputation", "Mode_Imputation", "Median_Imputation", "KNN-Imputation", "Iter. Imputation", "Uncertain_Simulation", "Original_Simulation"], 
                                           columns=["Sample Time", "Prediction Time"])
        
    
    
    
    SIMULATION_OUTPUT_ANALYSIS = {"original_mc_accuracy" : SIMULATION_MEAN_RESULTS["Original_Mean_Accuracy"],
                                  
                                  "original_mean_metrics" : {
                                          "metrics" : utils.create_metrics(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Mean_Labels"], print_report=False),
                                          "roc_auc" : roc_auc_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Mean"]),
                                          "prc_auc" : average_precision_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Mean"]),
                                          },
                                  
                                  "original_median_metrics" : {
                                          "metrics" : utils.create_metrics(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Median_Labels"], print_report=False),
                                          "roc_auc" : roc_auc_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Median"]),
                                          "prc_auc" : average_precision_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Median"]),
                                          },
                                  
                                  "original_mode_metrics" : {
                                          "metrics" : utils.create_metrics(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Mode_Labels"], print_report=False),
                                          "roc_auc" : roc_auc_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Mode"]),
                                          "prc_auc" : average_precision_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Mode"]),
                                          },
                                  
                                  "original_probability_metrics" : utils.create_metrics(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Original_Probability_Labels"], print_report=False),
    
                                    
    
                                  "uncertain_mc_accuracy" : SIMULATION_MEAN_RESULTS["Uncertain_Mean_Accuracy"],
                                  
                                  "uncertain_mean_metrics" : {
                                          "metrics" : utils.create_metrics(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Mean_Labels"], print_report=False),
                                          "roc_auc" : roc_auc_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Mean"]),
                                          "prc_auc" : average_precision_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Mean"]),
                                          },
                                  
                                  "uncertain_median_metrics" : {
                                          "metrics" : utils.create_metrics(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Median_Labels"], print_report=False),
                                          "roc_auc" : roc_auc_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Median"]),
                                          "prc_auc" : average_precision_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Median"]),
                                          },
                                    
                                  "uncertain_mode_metrics" : {
                                          "metrics" : utils.create_metrics(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Mode_Labels"], print_report=False),
                                          "roc_auc" : roc_auc_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Mode"]),
                                          "prc_auc" : average_precision_score(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Mode"]),
                                          },
        
                                  "uncertain_probability_metrics" : utils.create_metrics(y_original[_SIMULATION_RANGE], SIMULATION_MEAN_RESULTS["Uncertain_Probability_Labels"], print_report=False)
                                  }  
      
    

if _save_simulated_results:
    
    # save results and collections to folder
    # file name contains "dataset", "miss rate" and "simulation range" as identifier
    
    _results_id = len(os.listdir(_results_path))
    _results_file_name = os.path.join(_results_path, str(_results_id) + "_" + _dataset + "_" + str(_MISS_RATE) + "_" + str(_SIMULATION_RANGE))
    
    
    if _IMPUTE == True and _SIMULATE == True:
        pickle.dump({
            "SIMULATION_ROW_RESULTS" : SIMULATION_ROW_RESULTS,
            "SIMULATION_MEAN_RESULTS" : SIMULATION_MEAN_RESULTS,
            "DATAFRAME_INPUT_ANALYSIS" : DATAFRAME_INPUT_ANALYSIS,
            "DATAFRAME_TIME_ANALYSIS" : DATAFRAME_TIME_ANALYSIS,
            "SIMULATION_OUTPUT_ANALYSIS" : SIMULATION_OUTPUT_ANALYSIS
            }, open(_results_file_name, "wb"))
    else:
        pickle.dump({
            "SIMULATION_ROW_RESULTS" : SIMULATION_ROW_RESULTS,
            "SIMULATION_MEAN_RESULTS" : SIMULATION_MEAN_RESULTS,
            "SIMULATION_OUTPUT_ANALYSIS" : SIMULATION_OUTPUT_ANALYSIS
            }, open(_results_file_name, "wb"))



if _load_simulated_results:

    # list all available files:
    _available_files = os.listdir(_results_path)
    
    _file_found = False
    for _i in _available_files:
        if _i[0] == str(_load_results_id):
            
            _get_file = _available_files[int(_load_results_id)]
            
            if Path(os.path.join(_results_path, _get_file)).exists():
                SIMULATION_RESULTS_LOADED = pickle.load(open(os.path.join(_results_path, _get_file), "rb"))
                
                print(f"File '{_get_file}' was successfully loaded!")
                _file_found = True
                
            # break for loop if file was found
            break
        
    if _file_found == False: 
        print(f"No file with ID '{_load_results_id}' was found!")



