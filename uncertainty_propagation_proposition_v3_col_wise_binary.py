# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:24:32 2023

@author: Selii
"""

import os
import sys
import pickle

from tqdm import tqdm


import utils
import data_visualizations as dvis
from dataset_loader import load_dataframe
from dataset_loader import load_miss_dataframe
from model_loader import create_binary_model
from model_loader import load_binary_model



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import mean_squared_error as mse
import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


import statsmodels.api as sm

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

#choose working dataset: "australian" or "climate_simulation", "wdbc" -> Breast Cancer Wisconsin (Diagnostic)
_dataset = "wdbc"
_simulate_test_set = False

# set random state          
_RANDOM_STATE = 24

# other constants
_INIT_DATA_BANDWIDTH = None
_PRED_BANDWIDTH = None # --> if None (default) "scott" is used
#_KDE_WEIGHTS = None


# further dataset settings
_standardize_data = True


# settings for visualization
_visiualize_data = False
_visualize_original_predictions = False
_visualize_imputed_predictions = False


# train or load model
_train_model = False
_save_new_model = False
_load_old_model = True


# DATAFRAME_MISS settings - Introduction to missing values in the choosen Dataframe
# load DataFrame_MISS // if True, an already created one will be loaded, else a new one will be created
_load_dataframe_miss = True
_create_dataframe_miss = False

_DELETE_MODE = "static"     # static (amount of values in row deleted) // percentage (value between 0 and 1)
_MISS_RATE = 3


#KDE_VALUES OF EACH COLUMN - affected frames are DATAFRAME_SIMULATE -> Uncertain and DATAFRAME -> Certain/True
_compare_col_kde_distributions = False
_compare_col_kde_mode = "combined"    # choose between "single", "combined", "both" // combined can only be used when _visualize data == true


# modes for deterministic/stochastic experiments on missing dataframes
# choose between kde_imputer, mean, median, most_frequent, KNNImputer
_IMPUTE = True


_SIMULATE = True

_monte_carlo = False
_latin_hypercube = True

_LHS_MODE = "fast"
_SIMULATION_LENGTH = 100
#_SIMULATION_RANGE = None
_SIMULATION_RANGE = range(44, 45, 1)
_simulation_visualizations = True
_norm= True
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
    
    model = create_binary_model(_dataset, _X_original_train, _X_original_test, 
                                _y_original_train, _y_original_test, _save_new_model)


if _simulate_test_set:
    
    """
        # SWAP-Action:
            if this option is chosen, the calculations will be made for the test part of the dataset 
            -- for simplicity of not chaning the script, DATAFRAME_ORIGINAL will be ste to the ORIGINAL_DATAFRAME
    """
    
    DATAFRAME_ORIGINAL_FULL_VALUES = DATAFRAME_ORIGINAL.copy()
    
    X_original = _X_original_test
    y_original = _y_original_test
    
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
original_metrics["y_hat"] = model.predict(X_original).flatten()
original_metrics["y_hat_labels"] = (original_metrics["y_hat"]>0.5).astype("int32")


if _visualize_original_predictions: 
    
    dvis.plot_binary_predictions(original_metrics["y_hat"], original_metrics["y_hat_labels"], 
                                 title='Original (True) combined predictions')



    
print("\nOriginal Classification Metrics:")
original_metrics["metrics"] = utils.create_metrics(y_original, original_metrics["y_hat_labels"], 
                                                   print_report=True)




##########################################################################################################################
"""
        # Here in this step a new DATAFRAME is introduced. 
        # This contains missing data with a specific missing rate in each row
"""


DATAFRAME_MISS = load_miss_dataframe(_dataset, DATAFRAME_ORIGINAL, _MISS_RATE, _DELETE_MODE, _RANDOM_STATE,
                                     _load_dataframe_miss, _create_dataframe_miss)


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
    
    # mean imputation
    _mean_imp = SimpleImputer(strategy="mean")
    _DATAFRAME_MEAN_IMPUTE = pd.DataFrame(_mean_imp.fit_transform(DATAFRAME_MISS.copy()), columns=_column_names)
    
    _INPUT_RMSE_MEAN = mse(DATAFRAME_ORIGINAL, _DATAFRAME_MEAN_IMPUTE, squared=False)
    
    # median imputation
    _median_imp = SimpleImputer(strategy="median")
    _DATAFRAME_MEDIAN_IMPUTE = pd.DataFrame(_median_imp.fit_transform(DATAFRAME_MISS.copy()), columns=_column_names)

    _INPUT_RMSE_MEDIAN = mse(DATAFRAME_ORIGINAL, _DATAFRAME_MEDIAN_IMPUTE, squared=False)

    # mode imputation
    _mode_imp = SimpleImputer(strategy="most_frequent")
    _DATAFRAME_MODE_IMPUTE = pd.DataFrame(_mode_imp.fit_transform(DATAFRAME_MISS.copy()), columns=_column_names)
    
    _INPUT_RMSE_MODE = mse(DATAFRAME_ORIGINAL, _DATAFRAME_MODE_IMPUTE, squared=False)
    
    # knn imputation
    _knn_imp = KNNImputer(n_neighbors=5)
    _DATAFRAME_KNN_IMPUTE = pd.DataFrame(_knn_imp.fit_transform(DATAFRAME_MISS.iloc[:,:-1].copy()), columns=_column_names[:-1])
    _DATAFRAME_KNN_IMPUTE = _DATAFRAME_KNN_IMPUTE.merge(DATAFRAME_ORIGINAL["Outcome"], left_index=True, right_index=True)
    
    _INPUT_RMSE_KNNIMP = mse(DATAFRAME_ORIGINAL, _DATAFRAME_KNN_IMPUTE, squared=False)

    
    # multiple imputation technique --> MICE Algo.
    _iter_imp = IterativeImputer(max_iter=10, random_state=_RANDOM_STATE)
    _DATAFRAME_ITER_IMPUTE = pd.DataFrame(_iter_imp.fit_transform(DATAFRAME_MISS.iloc[:,:-1].copy()), columns=_column_names[:-1])
    _DATAFRAME_ITER_IMPUTE = _DATAFRAME_ITER_IMPUTE.merge(DATAFRAME_ORIGINAL["Outcome"], left_index=True, right_index=True)

    _INPUT_RMSE_ITERIMP = mse(DATAFRAME_ORIGINAL, _DATAFRAME_ITER_IMPUTE, squared=False)



    _DATAFRAME_IMPUTE_COLLECTION = {"MEAN_IMPUTE" : _DATAFRAME_MEAN_IMPUTE,
                                    "MEDIAN_IMPUTE" : _DATAFRAME_MEDIAN_IMPUTE,
                                    "MODE_IMPUTE" : _DATAFRAME_MODE_IMPUTE,
                                    "KNN_IMPUTE" : _DATAFRAME_KNN_IMPUTE,
                                    "ITER_IMPUTE" : _DATAFRAME_ITER_IMPUTE}
    


    
    print("\nPredictions for dataset with uncertainties and imputed values:")
    
    impute_metrics = {}
    
    for _frame_key in _DATAFRAME_IMPUTE_COLLECTION:
        
        print(f"Calculating results for dataframe: {_frame_key}")
        
        # create input frame for model predictions
        _X_impute = _DATAFRAME_IMPUTE_COLLECTION[_frame_key].iloc[:, 0:-1]
         
        # get results of prediction 
        _y_impute_hat = model.predict(_X_impute).flatten()
        _y_impute_hat_labels = (_y_impute_hat>0.5).astype("int32")             
        
        impute_metrics[_frame_key] = {"y_hat" : _y_impute_hat,
                                      "y_hat_labels" : _y_impute_hat_labels,
                                      }
        
        
        if _visualize_imputed_predictions:
            
            # visualize predictions
            dvis.plot_binary_predictions(_y_impute_hat, _y_impute_hat_labels,
                                         f'Uncertain imputated dataframe combined output - Miss-Rate: {_MISS_RATE} - Impute-Method: {_frame_key.replace("_IMPUTE", "")}')
       
        
            
        print(f"\n{_frame_key} Imputed Classification Metrics:")
        
        impute_metrics[_frame_key]["classification_metrics"] = utils.create_metrics(y_original, _y_impute_hat_labels, 
                                                                                    print_report=True)
 



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
    
    for _row in tqdm(_SIMULATION_RANGE):
        
        """
            # step 1: get current row to perform simulation with
        """
        _DATAFRAME_SIMULATE_ROW = pd.DataFrame(_DATAFRAME_SIMULATE.loc[_row])
        
        # for rmse calculation later on
        _original_df_mc_row_input = pd.DataFrame(X_original.loc[_row]).copy().transpose()
        _original_df_mc_row_input = pd.concat([_original_df_mc_row_input] * _SIMULATION_LENGTH, ignore_index=True)
        
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
        
        _uncertain_sim_row_metrics = utils.binary_sample_and_predict(model = model, 
                                                                     simulation_row = _DATAFRAME_SIMULATE_ROW,
                                                                     original_input_row = _original_df_mc_row_input,
                                                                     dataframe_categorical = _DATAFRAME_SIMULATE, 
                                                                     uncertain_attributes = _uncertain_attributes, 
                                                                     
                                                                     standardize_data = _standardize_data, 
                                                                     datatype_map = datatype_map, 
                                                                     column_names = _column_names,
                                                                     
                                                                     simulation_length = _SIMULATION_LENGTH, 
                                                                     random_state = _RANDOM_STATE, 
                                                                     
                                                                     monte_carlo = _monte_carlo,
                                                                     kde_collection = kde_collection_uncertain,
                                                                     normalize_kde = _norm,
                                                                     bw_method = _PRED_BANDWIDTH,
                                                                     x_axis = _x_axis,
                                                                     
                                                                     latin_hypercube = _latin_hypercube,
                                                                     lhs_mode = _LHS_MODE, 
                                                                     visualize_lhs_samples = False, 
                                                                     lhs_prefix=" Uncertain"
                                                                     )
        
        
        """
                #step 5.1.b: row-wise predictions on original samples
                    -----> Simulation procedure for true original kde induced simulation frames
        """
        
        _original_sim_row_metrics = utils.binary_sample_and_predict(model = model, 
                                                                    simulation_row = _DATAFRAME_SIMULATE_ROW,
                                                                    original_input_row = _original_df_mc_row_input,
                                                                    dataframe_categorical = DATAFRAME_ORIGINAL, 
                                                                    uncertain_attributes = _uncertain_attributes, 
                                                                    
                                                                    standardize_data = _standardize_data, 
                                                                    datatype_map = datatype_map, 
                                                                    column_names = _column_names,
                                                                     
                                                                    simulation_length = _SIMULATION_LENGTH, 
                                                                    random_state = _RANDOM_STATE, 
                                                                     
                                                                    monte_carlo = _monte_carlo,
                                                                    kde_collection = kde_collection_original,
                                                                    normalize_kde = _norm,
                                                                    bw_method = _PRED_BANDWIDTH,
                                                                    x_axis = _x_axis,
                                                                     
                                                                    latin_hypercube = _latin_hypercube,
                                                                    lhs_mode = _LHS_MODE, 
                                                                    visualize_lhs_samples = False, 
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





        # visualizations for binary simulation // comparison plots
        if _simulation_visualizations:
            
            # get last item of SIMULATION_ROW_RESULTS list and plot results
            dvis.simulation_hist_plot(SIMULATION_ROW_RESULTS[-1], y_original, original_metrics, plotmode="specific")


            dvis.simulation_kde_plot(_x_axis, SIMULATION_ROW_RESULTS[-1], y_original, original_metrics, impute_metrics, plotmode="specific")



            
            
    
    
    # TODO
    sys.exit()
    
    
    
    # simulation collection is holding all summarized information
    SIMULATION_COLLECTION = {
        "0_Simulation_Info" : {
            "0.1_random_state" : _RANDOM_STATE,
            "0.2_dataset" : _dataset,
            "0.3_dataset_size" : DATAFRAME_ORIGINAL.size,
            "0.4_miss_rate" : _MISS_RATE,
            "0.5_num_missing" : DATAFRAME_MISS.isnull().sum().sum(),
            "0.6_miss_rate_%" : round(DATAFRAME_MISS.isnull().sum().sum() * 100 / DATAFRAME_ORIGINAL.size, 2),
            "0.7_simulation_length" : _SIMULATION_LENGTH,
            "0.8_elapsed_sim_time" : "",
            "0.9_simulated_rows" : len(_SIMULATION_RANGE)
            },
        "1_Uncertain_Simulation" : {
            "1.0_Input_RMSE" : [],
            "1.1_Means" : [],
            "1.2_Mean_Labels" : [],
            "1.3_Mean_Label_Frequenzy" : [],
            "1.4_Stds" : [],
            
            
            
            "1.5_Max_Density_Sigmoid" : [],
            "1.6_Max_Density_Sig_Label" : [],
            "1.7_Max_Density_Sig_Label_Frequency" : [],
            "1.8_Lower_Bound_Probability" : [],
            "1.9_Upper_Bound_Probability" : []
            },
        "2_Original_Simulation" : {
            "2.0_Input_RMSE" : [],
            "2.1_Means" : [],
            "2.2_Mean_Labels" : [],
            "2.3_Mean_Label_Frequenzy" : [],
            "2.4_Stds" : [],
            
            
            "2.5_Max_Density_Sigmoid" : [],
            "2.6_Max_Density_Sig_Label" : [],
            "2.7_Max_Density_Sig_Label_Frequency" : [],
            "2.8_Lower_Bound_Probability" : [],
            "2.9_Upper_Bound_Probability" : []
            }
        }
    

    # some accessory time metrics for comparison 
    _main_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - _main_sim_start_time))    
        
    SIMULATION_COLLECTION["0_Simulation_Info"]["0.8_elapsed_sim_time"] = str(_main_elapsed_time)
    
    SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.3_Mean_Label_Frequenzy"] = pd.Series(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.2_Mean_Labels"]).value_counts()
    SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.7_Max_Density_Sig_Label_Frequency"] = pd.Series(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.6_Max_Density_Sig_Label"]).value_counts()
    SIMULATION_COLLECTION["2_Original_Simulation"]["2.3_Mean_Label_Frequenzy"] = pd.Series(SIMULATION_COLLECTION["2_Original_Simulation"]["2.2_Mean_Labels"]).value_counts()
    SIMULATION_COLLECTION["2_Original_Simulation"]["2.7_Max_Density_Sig_Label_Frequency"] = pd.Series(SIMULATION_COLLECTION["2_Original_Simulation"]["2.6_Max_Density_Sig_Label"]).value_counts()
            
    
    SIMULATION_COLLECTION["3_Uncert_vs_Orig_KDE"] = {
        "3.1_Explanation" : "Analysing the differences between Uncertain and Original KDE Simulations",
        "3.2_Sim_Mean_rmse" : mse(SIMULATION_COLLECTION["2_Original_Simulation"]["2.1_Means"], SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.1_Means"], squared=False),
        "3.3_Sim_Stds_rmse" : mse(SIMULATION_COLLECTION["2_Original_Simulation"]["2.4_Stds"], SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.4_Stds"], squared=False),
        "3.4_Sim_Max_Density_Sigmoid_rmse" : mse(SIMULATION_COLLECTION["2_Original_Simulation"]["2.5_Max_Density_Sigmoid"], SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.5_Max_Density_Sigmoid"], squared=False)
        }
        

    print('\n\nSimulation execution time:', _main_elapsed_time)
    
    
    
    """
        Below: combined Comparisons between the prediction results of Uncertain and Certain KDE simulations

    
    _fig, _axs = plt.subplots(2, 2, figsize=(17, 11))
    
    # visualize predictions - uncertain density
    sns.histplot(data={"Sigmoid Activations" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.5_Max_Density_Sigmoid"], 
                       "density_label" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.6_Max_Density_Sig_Label"]}, 
                 x="Sigmoid Activations", 
                 hue="density_label", 
                 bins=15, 
                 binrange=(0, 1), 
                 stat="count", 
                 kde=False, 
                 kde_kws={"cut":0}, 
                 ax=_axs[0, 0]).set(title=f'Simulation (uncertain_kde) - Miss-Rate: {_MISS_RATE} - Metric: Sim. Density')

    # visualize predictions - original density
    sns.histplot(data={"Sigmoid Activations" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.5_Max_Density_Sigmoid"], 
                       "density_label" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.6_Max_Density_Sig_Label"]}, 
                 x="Sigmoid Activations", 
                 hue="density_label", 
                 bins=15, 
                 binrange=(0, 1), 
                 stat="count", 
                 kde=False, 
                 kde_kws={"cut":0}, 
                 ax=_axs[0, 1]).set(title=f'Simulation (original_kde) - Miss-Rate: {_MISS_RATE} - Metric: Sim. Density')
    
        
    # visualize predictions - uncertain mean simulation
    #plt._figure(figsize=(10, 6))
    sns.histplot(data={"Sigmoid Activations" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.1_Means"], 
                       "mean_label" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.2_Mean_Labels"]}, 
                 x="Sigmoid Activations", 
                 hue="mean_label", 
                 bins=15, 
                 binrange=(0, 1), 
                 stat="count", 
                 kde=False, 
                 kde_kws={"cut":0}, 
                 ax=_axs[1, 0]).set(title=f'Simulation (uncertain_kde) - Miss-Rate: {_MISS_RATE} - Metric: Sim. Mean')


    # visualize predictions - original mean simulation
    #plt._figure(figsize=(10, 6))
    sns.histplot(data={"Sigmoid Activations" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.1_Means"], 
                       "mean_label" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.2_Mean_Labels"]}, 
                 x="Sigmoid Activations", 
                 hue="mean_label", 
                 bins=15, 
                 binrange=(0, 1), 
                 stat="count", 
                 kde=False, 
                 kde_kws={"cut":0}, 
                 ax=_axs[1, 1]).set(title=f'Simulation (original_kde) - Miss-Rate: {_MISS_RATE} - Metric: Sim. Mean')
    plt.show()
    """






# exit if statement if no further simulations will be made
if _IMPUTE == False and _SIMULATE == False:
    sys.exit()


if _IMPUTE == True and _SIMULATE == True:
        
        _min_idx = min(_SIMULATION_RANGE)
        _max_idx = max(_SIMULATION_RANGE) + 1


        DATAFRAME_COMBINED_ROW_RESULTS = pd.DataFrame(data={"Original_Label" : y_original[_min_idx:_max_idx],
                                                        "0_Prediction" : original_metrics["y_hat"][_min_idx:_max_idx],
                                                        "0_Predicted_Label" : original_metrics["y_hat_labels"][_min_idx:_max_idx],
                                                        
                                                        "1_Imputation-Mean" : impute_metrics["MEAN_IMPUTE"]["y_hat"][_min_idx:_max_idx],
                                                        "1_Imputation-Mean_Label" : impute_metrics["MEAN_IMPUTE"]["y_hat_labels"][_min_idx:_max_idx],    
                                                        "1_Imputation-Mode" : impute_metrics["MODE_IMPUTE"]["y_hat"][_min_idx:_max_idx],
                                                        "1_Imputation-Mode_Label" : impute_metrics["MODE_IMPUTE"]["y_hat_labels"][_min_idx:_max_idx],
                                                        "1_Imputation-Median" : impute_metrics["MEDIAN_IMPUTE"]["y_hat"][_min_idx:_max_idx],
                                                        "1_Imputation-Median_Label" : impute_metrics["MEDIAN_IMPUTE"]["y_hat_labels"][_min_idx:_max_idx],
                                                        "1_Imputation-KNNImp" : impute_metrics["KNN_IMPUTE"]["y_hat"][_min_idx:_max_idx],
                                                        "1_Imputation-KNNIMP_Label" : impute_metrics["KNN_IMPUTE"]["y_hat_labels"][_min_idx:_max_idx],
                                                        "1_Imputation-ITERIMP" : impute_metrics["ITER_IMPUTE"]["y_hat"][_min_idx:_max_idx],
                                                        "1_Imputation-ITERIMP_Label" :impute_metrics["ITER_IMPUTE"]["y_hat_labels"][_min_idx:_max_idx],
                                                        
                                                        "2_Orig_Sim_Mean" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.1_Means"],
                                                        "2_Orig_Sim_Mean_Label" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.2_Mean_Labels"],
                                                        "2_Orig_Sim_Mean_Std" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.4_Stds"],
                                                        "2_Orig_Sim_Max_Density" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.5_Max_Density_Sigmoid"],
                                                        "2_Orig_Sim_Max_Density_Label" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.6_Max_Density_Sig_Label"],
                                                        "2_Orig_Lower_Bound_Probability" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.8_Lower_Bound_Probability"],
                                                        "2_Orig_Upper_Bound_Probability" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.9_Upper_Bound_Probability"],
                                                                         
                                                        "3_Uncert_Sim_Mean" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.1_Means"],
                                                        "3_Uncert_Sim_Mean_Label" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.2_Mean_Labels"],
                                                        "3_Uncert_Sim_Mean_Std" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.4_Stds"],
                                                        "3_Uncert_Sim_Max_Density" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.5_Max_Density_Sigmoid"],
                                                        "3_Uncert_Sim_Max_Density_Label" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.6_Max_Density_Sig_Label"],
                                                        "3_Uncert_Lower_Bound_Probability" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.8_Lower_Bound_Probability"],
                                                        "3_Uncert_Upper_Bound_Probability" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.9_Upper_Bound_Probability"]
                                                        }).transpose()


        # calculate the distance to model prediction
        _COMBINED_DISTANCES_PREDICTION = pd.Series(data={"1_Imp_Mean_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Prediction"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mean"], squared=False),                                                      
                                                        "1_Imp_Mode_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Prediction"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mode"], squared=False),                 
                                                        "1_Imp_Median_distancs" :  mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Prediction"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Median"], squared=False),
                                                        "1_Imp_KNN_distancs" :  mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Prediction"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-KNNImp"], squared=False),
                                                        "1_Imp_ITERIMP_distancs" :  mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Prediction"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-ITERIMP"], squared=False),
                                                        "2_Orig_Sim_Mean_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Prediction"], DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Mean"], squared=False),
                                                        "2_Orig_Sim_Max_Density_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Prediction"], DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Max_Density"], squared=False),
                                                        "3_Uncert_Sim_Mean_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Prediction"], DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Mean"], squared=False),
                                                        "3_Uncert_Sim_Max_Density_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Prediction"], DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Max_Density"], squared=False)
                                                        }, name="to Prediction")
        
        
        # calculate the distance to predicted model label
        _COMBINED_DISTANCES_PREDICTION_LABEL = pd.Series(data={"1_Imp_Mean_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mean"], squared=False),                                                      
                                                               "1_Imp_Mode_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mode"], squared=False),                 
                                                               "1_Imp_Median_distancs" :  mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Median"], squared=False),
                                                               "1_Imp_KNN_distancs" :  mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-KNNImp"], squared=False),
                                                               "1_Imp_ITERIMP_distancs" :  mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-ITERIMP"], squared=False),
                                                               "2_Orig_Sim_Mean_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Mean"], squared=False),
                                                               "2_Orig_Sim_Max_Density_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Max_Density"], squared=False),
                                                               "3_Uncert_Sim_Mean_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Mean"], squared=False),
                                                               "3_Uncert_Sim_Max_Density_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Max_Density"], squared=False)
                                                               }, name="to Prediction Label")
        
        
        # calculate the distance to predicted model label
        _COMBINED_DISTANCES_ORIGINAL = pd.Series(data={"1_Imp_Mean_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mean"], squared=False),                                                      
                                                       "1_Imp_Mode_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mode"], squared=False),                 
                                                       "1_Imp_Median_distancs" :  mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Median"], squared=False),
                                                       "1_Imp_KNN_distancs" :  mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-KNNImp"], squared=False),
                                                       "1_Imp_ITERIMP_distancs" :  mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-ITERIMP"], squared=False),
                                                       "2_Orig_Sim_Mean_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Mean"], squared=False),
                                                       "2_Orig_Sim_Max_Density_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Max_Density"], squared=False),
                                                       "3_Uncert_Sim_Mean_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Mean"], squared=False),
                                                       "3_Uncert_Sim_Max_Density_distances" : mse(DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"], DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Max_Density"], squared=False)
                                                       }, name="to Original Label")



        COMBINED_DISTANCES_ANALYSIS = pd.DataFrame(data={_COMBINED_DISTANCES_ORIGINAL.name : _COMBINED_DISTANCES_ORIGINAL,
                                                         _COMBINED_DISTANCES_PREDICTION_LABEL.name : _COMBINED_DISTANCES_PREDICTION_LABEL,
                                                         _COMBINED_DISTANCES_PREDICTION.name : _COMBINED_DISTANCES_PREDICTION
                                                         })  
        min_distance = pd.Series(COMBINED_DISTANCES_ANALYSIS.idxmin(axis=0), name="4_Min Distance").to_frame().T
        COMBINED_DISTANCES_ANALYSIS = pd.concat([COMBINED_DISTANCES_ANALYSIS, min_distance])
        
        
        
        DATAFRAME_COMBINED_LABELS = pd.DataFrame(data={"Original_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"],
                                                       
                                                       "0_Predicted_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"],
                                                       "0_Predicted_Label_Corr_Asign" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"]),
                                                       
                                                       "1_Imp-Mean_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mean_Label"],
                                                       "1_Imp-Mean_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mean_Label"]),
                                                       "1_Imp-Mean_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mean_Label"]),
                                                       
                                                       "1_Imp-Mode_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mode_Label"],
                                                       "1_Imp-Mode_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mode_Label"]),
                                                       "1_Imp-Mode_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mode_Label"]),
                                                       
                                                       "1_Imp-Median_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Median_Label"],
                                                       "1_Imp-Median_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Median_Label"]),
                                                       "1_Imp-Median_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Median_Label"]),
                                                       
                                                       "1_Imp-KNNIMP_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-KNNIMP_Label"],
                                                       "1_Imp-KNNIMP_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-KNNIMP_Label"]),
                                                       "1_Imp-KNNIMP_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-KNNIMP_Label"]),
                                                       
                                                       "1_Imp-ITERIMP_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-ITERIMP_Label"],
                                                       "1_Imp-ITERIMP_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-ITERIMP_Label"]),
                                                       "1_Imp-ITERIMP_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-ITERIMP_Label"]),
                                                       
                                                       "2_Orig_Sim_Mean_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Mean_Label"],
                                                       "2_Orig_Sim_Mean_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Mean_Label"]),
                                                       "2_Orig_Sim_Mean_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Mean_Label"]),
                                                       
                                                       "2_Orig_Sim_Max_Density_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Max_Density_Label"],
                                                       "2_Orig_Sim_Max_Density_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Max_Density_Label"]),
                                                       "2_Orig_Sim_Max_Density_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["2_Orig_Sim_Max_Density_Label"]),
                                                       
                                                       "3_Uncert_Sim_Mean_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Mean_Label"],
                                                       "3_Uncert_Sim_Mean_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Mean_Label"]),
                                                       "3_Uncert_Sim_Mean_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Mean_Label"]),
                                                       
                                                       "3_Uncert_Sim_Max_Density_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Max_Density_Label"],
                                                       "3_Uncert_Sim_Max_Density_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Max_Density_Label"]),
                                                       "3_Uncert_Sim_Max_Density_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["3_Uncert_Sim_Max_Density_Label"]),
                                                       }).transpose()
        
        

    
        DATAFRAME_COMBINED_LABELS_ANALYSIS = pd.Series(data={"Correct Orig. labels assigned by 0_Predicted_Label": DATAFRAME_COMBINED_LABELS.loc["0_Predicted_Label_Corr_Asign"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 1_Imp-Mean_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-Mean_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 1_Imp-Mean_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-Mean_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 1_Imp-Mode_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-Mode_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 1_Imp-Mode_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-Mode_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 1_Imp-Median_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-Median_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 1_Imp-Median_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-Median_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 1_Imp-KNNIMP_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-KNNIMP_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 1_Imp-KNNIMP_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-KNNIMP_Label_Corr_Asign_Predicted"].value_counts(True)[0],                          
                                                             
                                                             "Correct Orig. labels assigned by 1_Imp-ITERIMP_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-ITERIMP_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 1_Imp-ITERIMP_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imp-ITERIMP_Label_Corr_Asign_Predicted"].value_counts(True)[0],   
                                                             
                                                             "Correct Orig. labels assigned by 2_Orig_Sim_Mean_Label": DATAFRAME_COMBINED_LABELS.loc["2_Orig_Sim_Mean_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 2_Orig_Sim_Mean_Label": DATAFRAME_COMBINED_LABELS.loc["2_Orig_Sim_Mean_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 2_Orig_Sim_Max_Density_Label": DATAFRAME_COMBINED_LABELS.loc["2_Orig_Sim_Max_Density_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 2_Orig_Sim_Max_Density_Label": DATAFRAME_COMBINED_LABELS.loc["2_Orig_Sim_Max_Density_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 3_Uncert_Sim_Mean_Label": DATAFRAME_COMBINED_LABELS.loc["3_Uncert_Sim_Mean_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 3_Uncert_Sim_Mean_Label": DATAFRAME_COMBINED_LABELS.loc["3_Uncert_Sim_Mean_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 3_Uncert_Sim_Max_Density_Label": DATAFRAME_COMBINED_LABELS.loc["3_Uncert_Sim_Max_Density_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 3_Uncert_Sim_Max_Density_Label": DATAFRAME_COMBINED_LABELS.loc["3_Uncert_Sim_Max_Density_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             })
    

        



        # calculate the Input RMSE between the original DataFrame and the Uncertain DataFrames    
        DATAFRAME_INPUT_ANALYSIS = pd.Series(data={"Mean_Impute_df" : _INPUT_RMSE_MEAN,
                                                   "Mode_Impute_df" : _INPUT_RMSE_MODE,
                                                   "Median_Impute_df" : _INPUT_RMSE_MEDIAN,
                                                   "KNNImp_Impute_df" : _INPUT_RMSE_KNNIMP,
                                                   "IterImp_Impute_df" : _INPUT_RMSE_ITERIMP,
                                                   "Uncertain_Mean_Sim_Input_RMSE" : np.mean(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.0_Input_RMSE"]),
                                                   "Original_Mean_Sim_Input_RMSE" : np.mean(SIMULATION_COLLECTION["2_Original_Simulation"]["2.0_Input_RMSE"])
                                                   }, name="RMSE")



if _save_simulated_results:
    
    # save results and collections to folder
    # file name contains "dataset", "miss rate" and "simulation range" as identifier
    
    _results_id = len(os.listdir(_results_path))
    _results_file_name = os.path.join(_results_path, str(_results_id) + "_" + _dataset + "_" + str(_MISS_RATE) + "_" + str(_SIMULATION_RANGE))
    
    
    if _IMPUTE == True and _SIMULATE == True:
        pickle.dump({"SIMULATION_COLLECTION": SIMULATION_COLLECTION,
                     "SIMULATION_ROW_RESULTS" : SIMULATION_ROW_RESULTS,
                     "DATAFRAME_COMBINED_ROW_RESULTS" : DATAFRAME_COMBINED_ROW_RESULTS,
                     "COMBINED_DISTANCES_ANALYSIS" : COMBINED_DISTANCES_ANALYSIS,
                     "DATAFRAME_COMBINED_LABELS" : DATAFRAME_COMBINED_LABELS,
                     "DATAFRAME_COMBINED_LABELS_ANALYSIS" : DATAFRAME_COMBINED_LABELS_ANALYSIS
                     }, open(_results_file_name, "wb"))
    else:
        pickle.dump({"SIMULATION_COLLECTION": SIMULATION_COLLECTION,
                     "SIMULATION_ROW_RESULTS" : SIMULATION_ROW_RESULTS,
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



