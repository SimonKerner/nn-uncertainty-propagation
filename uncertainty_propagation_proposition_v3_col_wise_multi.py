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
# set important paths
##########################################################################################################################


#image_path = os.path.join(os.getcwd(), 'images')
_model_path = os.path.join(os.getcwd(), 'models')
_results_path = os.path.join(os.getcwd(), 'sim_results')




##########################################################################################################################
"""
information about the datasets:
    -[1] wdbc - all attributes are considered continious - outcome is binary 
    -[2] climate_simulation - 
    -[3] australian - 
    -[4] predict+students+dropout+and+academic+success - three outcomes aka. "students"
    
following all the different settings for this simulation run can be found
    -dataset = "choose dataset"
    -standardize_dataset = "used for standardizing the dataset -- values between 0 and 1 -- minmax"
"""
##########################################################################################################################

#choose working dataset: "australian" or "climate_simulation", "wdbc" -> Breast Cancer Wisconsin (Diagnostic)
_dataset = "students"
_simulate_test_set = False

# set random state          
_RANDOM_STATE = 42

# other constants
_INIT_DATA_BANDWIDTH = None
_PRED_BANDWIDTH = None # --> if None (default) "scott" is used



# further dataset settings
_standardize_data = True


# settings for visualization
_visiualize_data = True
_visualize_original_predictions = True
_visualize_imputed_predictions = True


# train or load model
_train_model = False
_save_new_model = False
_load_model = True


# prediction metrics
_get_original_prediction_metrics = False
_get_imputed_prediction_metrics = False
_get_simulated_prediction_metrics = False


# DATAFRAME_MISS settings - Introduction to missing values in the choosen Dataframe
# load DataFrame_MISS // if True, an already created one will be loaded, else a new one will be created
_load_dataframe_miss = True
_create_dataframe_miss = True

_DELETE_MODE = "static"     # static (amount of values in row deleted) // percentage (value between 0 and 1)
_MISS_RATE = 2


#KDE_VALUES OF EACH COLUMN - affected frames are DATAFRAME_SIMULATE -> Uncertain and DATAFRAME -> Certain/True
_compare_col_kde_distributions = False
_compare_col_kde_mode = "combined"    # choose between "single", "combined", "both"


# modes for deterministic/stochastic experiments on missing dataframes
# choose between kde_imputer, mean, median, most_frequent, KNNImputer
_IMPUTE = True


_SIMULATE = True
_monte_carlo = False
_latin_hypercube = True
_LHS_MODE = "fast"
_SIMULATION_LENGTH = 10000
#_SIMULATION_RANGE = None
_SIMULATION_RANGE = range(0, 2, 1)
_simulation_visualizations = True
_norm= True
_save_simulated_results = False
_visualize_lhs_samples =False
_load_simulated_results = False
_load_results_id = 0


##########################################################################################################################
"""
    # load original datasets with full data
"""
##########################################################################################################################

    
DATAFRAME_ORIGINAL, datatype_map = load_dataframe(_dataset, _standardize_data)
_column_names = DATAFRAME_ORIGINAL.columns
_unique_outcomes = len(DATAFRAME_ORIGINAL.Outcome.unique())

    


##########################################################################################################################
"""
    # visiualize true underlying data of Dataframe 
"""
##########################################################################################################################


if _visiualize_data:
    
    # Plotting combined distribution using histograms
    _hist = DATAFRAME_ORIGINAL.hist(column=_column_names, 
                                    bins=10, 
                                    figsize=(20, 12), 
                                    density=False, 
                                    sharey=False, 
                                    sharex=False)
    plt.title('Input without missing data')
    plt.tight_layout()
    plt.show()

    
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
##########################################################################################################################

    
X_original = DATAFRAME_ORIGINAL.iloc[:, 0:-1]
y_original = DATAFRAME_ORIGINAL[_column_names[-1]]



# y labels have to be changed to categorical data - num _classes is equal to len unique values in DATAFRAME
y_original_categorical = keras.utils.to_categorical(y_original, num_classes=len(DATAFRAME_ORIGINAL.Outcome.unique()))

_X_original_train, _X_original_test, _y_original_train, _y_original_test = train_test_split(X_original, 
                                                                                        y_original_categorical, 
                                                                                        #stratify=y_original,
                                                                                        test_size=0.3,
                                                                                        random_state=_RANDOM_STATE)



##########################################################################################################################
"""
    # create standard vanilla feed forward feural network
"""
##########################################################################################################################


if _train_model:
    
    tf.keras.backend.clear_session()
    
    # layers of the network
    _inputs = keras.Input(shape=(X_original.shape[1]))
    
    #_x = layers.Dense(32, activation='relu')(_inputs)  
    
    #_x = layers.Dropout(0.4)(_x)
    
    _x = layers.Dense(16, activation='relu')(_inputs)
    _x = layers.Dense(8, activation='relu')(_x)
    
    """
        --> Multivariate Model 
    """
    
    _logits = layers.Dense(_unique_outcomes, activation=None)(_x)

    
    # model with two output heads
    #   1. Head: outputs the sigmoid without any activation function
    #   2. Head: outputs a default softmax layer for classification
    model = keras.Model(inputs=_inputs, outputs={"sigmoid" : tf.nn.sigmoid(_logits, name="sigmoid"), 
                                                "softmax" : tf.nn.softmax(_logits, name="softmax")}) #_logits
    
    
    # compile model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss={"sigmoid": keras.losses.BinaryCrossentropy(), #lambda y_true, y_pred: 0.0, #
                        "softmax": keras.losses.CategoricalCrossentropy()}, # lambda y_true, y_pred: 0.0
                  metrics=["accuracy"])
    
    
    
    # fit model        
    model_history = model.fit(_X_original_train, 
                              {"sigmoid": _y_original_train, 
                               "softmax": _y_original_train}, 
                              validation_data=[_X_original_test, _y_original_test], 
                              batch_size=150,
                              epochs=75,
                              verbose=0)
    
    model.summary()
    
    
    if _save_new_model:
        # save new model
        model.save(os.path.join(_model_path, _dataset + "_multi_model.keras"))


    # plot model
    dvis.plot_history(model_history, model_type="multi")
    



##########################################################################################################################
# load model without training
##########################################################################################################################


if _load_model:
    
    # load model, but do not compile (because of custom layer). 
    # Compiling in a second step 
    model = keras.models.load_model(os.path.join(_model_path, _dataset + "_multi_model.keras"), compile=False)
    
    # compile model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss={"sigmoid": keras.losses.BinaryCrossentropy(),#lambda y_true, y_pred: 0.0,
                        "softmax": keras.losses.CategoricalCrossentropy()},
                  metrics=["accuracy"])
    
    
    # print model summary
    print("\nShowing trained model summary:\n")
    model.summary()




##########################################################################################################################
# singe prediction metrics with a perfectly trained model - no uncertainties -- deterministic as usual
"""
    in the following block, all the standard deterministic predictions on the original dataset can be inspected
"""
##########################################################################################################################


print("\nPredictions for complete Dataset without uncertainties:")

    
y_original_hat = model.predict(X_original)

y_original_hat_label = {"softmax" : np.argmax(y_original_hat["softmax"], axis=1),
                        "sigmoid" : np.argmax(y_original_hat["sigmoid"], axis=1)} 

y_original_hat_label_soft_freq = pd.Series(y_original_hat_label["softmax"]).value_counts()
y_original_hat_label_sig_freq = pd.Series(y_original_hat_label["sigmoid"]).value_counts()

if _visualize_original_predictions:
    
    
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=y_original_hat_label["softmax"], 
                 bins=10, 
                 stat="count")
    plt.xlabel('Softmax Activations')
    plt.ylabel('Frequency')
    plt.title('True Combined Output Hist Plot - Softmax')
    plt.tight_layout()
    plt.show()
    
    
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=y_original_hat_label["sigmoid"], 
                 bins=10, 
                 stat="count")
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Frequency')
    plt.title('True Combined Output Hist Plot - Sigmoid')
    plt.tight_layout()
    plt.show()


"""
if _get_original_prediction_metrics:
    

    utils.create_metrics(y_original, y_original_hat_labels_soft)
    plt.show()
"""



##########################################################################################################################
# introduce missing data - aka. aleatoric uncertainty
"""
    Here in this step a new DATAFRAME is introduced. This contains missing data with a specific missing rate in each row
"""
##########################################################################################################################


"""
        # Here in this step a new DATAFRAME is introduced. 
        # This contains missing data with a specific missing rate in each row
"""


DATAFRAME_MISS = load_miss_dataframe(_dataset, DATAFRAME_ORIGINAL, _MISS_RATE, _DELETE_MODE, _RANDOM_STATE,
                                     _load_dataframe_miss, _create_dataframe_miss, _simulate_test_set)





if _visiualize_data:
    
    # Plotting combined distribution using histograms
    DATAFRAME_MISS.hist(column=_column_names, 
                        bins=10, 
                        figsize=(20, 12), 
                        density=False, 
                        sharey=False, 
                        sharex=False)
    plt.title('Input without missing data')
    plt.tight_layout()
    plt.show()
    
    
    # comparison of original and uncertain DATAFRAME    
    plt.figure(figsize=(12, 6))
    sns.histplot(data={"DATAFRAME_ORIGINAL" : np.array(DATAFRAME_ORIGINAL.iloc[:,:-1]).flatten(), 
                       "DATAFRAME_MISS" : np.array(DATAFRAME_MISS.iloc[:,:-1]).flatten()})
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Original & Uncertain dataset as flattened histplot')
    plt.tight_layout()
    plt.show()
    



    
##########################################################################################################################
"""
    various imputation methodsfor missing data for each simulation run can be tested
    - if True (with imputation):
        -choose between KDE_imputer (self), SimpleImputer and KNN_Imputer
    - if False (without imputation): 
        -DATAFRAME_IMPUTE will be equal to DATAFRAME_MISS
        
        
    > Further explanation:
        - with imputation: This method can be used to fill missing values inside of the dataset before further usage
            - uncertainties of miss data will be imputed with values from the above methods (can be considered deterministic)
        - withoud imputation:
            - this method will not fill the missing values of the dataset, instead it can be used for stochastic simulation,
              with MonteCarlo methods - propagation of uncertainties is guranteed 
"""
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
    

    
    _DATAFRAME_IMPUTE_COLLECTION = {"MEAN_IMPUTE" : _DATAFRAME_MEAN_IMPUTE,
                                    "MEDIAN_IMPUTE" : _DATAFRAME_MEDIAN_IMPUTE,
                                    "MODE_IMPUTE" : _DATAFRAME_MODE_IMPUTE,
                                    "KNN_IMPUTE" : _DATAFRAME_KNN_IMPUTE}
    
    _IMPUTE_TIMES = {"MEAN_IMPUTE" : _mean_end_sample_time,
                     "MEDIAN_IMPUTE" : _median_end_sample_time,
                     "MODE_IMPUTE" : _mode_end_sample_time,
                     "KNN_IMPUTE" : _knn_end_sample_time}
    
    _IMPUTE_RMSE = {"MEAN_IMPUTE" : mse(DATAFRAME_ORIGINAL, _DATAFRAME_MEAN_IMPUTE, squared=False),
                     "MEDIAN_IMPUTE" : mse(DATAFRAME_ORIGINAL, _DATAFRAME_MEDIAN_IMPUTE, squared=False),
                     "MODE_IMPUTE" : mse(DATAFRAME_ORIGINAL, _DATAFRAME_MODE_IMPUTE, squared=False),
                     "KNN_IMPUTE" :  mse(DATAFRAME_ORIGINAL, _DATAFRAME_KNN_IMPUTE, squared=False)}
    



##########################################################################################################################
# experiments modul 1 - with imputation --> full data --> get_predictions
##########################################################################################################################


if _IMPUTE:
    
    print("\nPredictions for dataset with uncertainties and imputed values:")
    
    DATAFRAME_IMPUTE_RESULTS_COLLECTION = {}
    
    for _frame_key in _DATAFRAME_IMPUTE_COLLECTION:
        
        print(f"Calculating results for dataframe: {_frame_key}")
        
        # create input frame for model predictions
        _X_impute = _DATAFRAME_IMPUTE_COLLECTION[_frame_key].iloc[:, 0:-1]
         
        # get results of prediction 
        _y_impute_hat = model.predict(_X_impute)#.flatten()
        
        _y_impute_hat_labels_soft = np.argmax(_y_impute_hat["softmax"], axis=1)
        _y_impute_hat_labels_sig = np.argmax(_y_impute_hat["sigmoid"], axis=1)
        
        _y_impute_hat_label_soft_freq = pd.Series(_y_impute_hat_labels_soft).value_counts()
        _y_impute_hat_label_sig_freq = pd.Series(_y_impute_hat_labels_sig).value_counts()
        

        
        
        
        DATAFRAME_IMPUTE_RESULTS_COLLECTION[_frame_key] = {"softmax" : {"y_impute_hat" : _y_impute_hat["softmax"],
                                                                        "y_impute_hat_labels" : _y_impute_hat_labels_soft,
                                                                        "y_impute_hat_label_frequency" : _y_impute_hat_label_soft_freq
                                                                        },
                                                           "sigmoid" : {"y_impute_hat" : _y_impute_hat["sigmoid"],
                                                                        "y_impute_hat_labels" : _y_impute_hat_labels_sig,
                                                                        "y_impute_hat_label_frequency" : _y_impute_hat_label_sig_freq
                                                                        }
                                                           }
        
        
        if _visualize_imputed_predictions:
            
            # visualize predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(data=_y_impute_hat_labels_soft, bins=10, stat="count", kde=False, kde_kws={"cut":0})
            plt.xlabel('Softmax Activations')
            plt.ylabel('Frequency')
            plt.title(f'Uncertain imputated dataframe combined output - Miss-Rate: {_MISS_RATE} - Impute-Method: {_frame_key.replace("_IMPUTE", "")} - Softmax')
            plt.tight_layout()
            plt.show()
    
    
            # visualize predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(data=_y_impute_hat_labels_soft, bins=10, stat="count", kde=False, kde_kws={"cut":0})
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Frequency')
            plt.title(f'Uncertain imputated dataframe combined output - Miss-Rate: {_MISS_RATE} - Impute-Method: {_frame_key.replace("_IMPUTE", "")} - Sigmoid')
            plt.tight_layout()
            plt.show()

    




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

        


    """
        Comperative Visualization of Certain (True) and Uncertain Column Distribution
        --> good for analyzing the differences between the two distribtions
    """    
    
    
    if _compare_col_kde_distributions: 
    
        
        if _compare_col_kde_mode == "single" or _compare_col_kde_mode == "both":
            
            for _column in _column_names:
                
                # KDE Plot of column without missing data
                plt.figure(figsize=(8, 4))
                sns.kdeplot(data={"Certain Distribution // DATAFRAME_ORIGINAL" : DATAFRAME_ORIGINAL[_column], 
                                  "Uncertain Distribution // DATAFRAME_SIMULATE" : _DATAFRAME_SIMULATE[_column]}, 
                            common_grid=True, 
                            bw_method=_INIT_DATA_BANDWIDTH)
                plt.xlabel(_column)
                plt.ylabel('Density')
                plt.title(f'KDE Plot of Column: {_column} - Miss-Rate: {_MISS_RATE} - Method: {_SIMULATE_METHOD}')
                plt.tight_layout()
                plt.show()
                
                
        elif _compare_col_kde_mode == "combined" or _compare_col_kde_mode == "both":
    
            """
                This upcoming plot provides an overall overview over all the single column kde distributuion in a single plot, 
                instead of a single plot for each column
            """
        
            plt.figure(0, figsize=(18, 10))
            _column_count = 0
            
            for _i in range(_hist.shape[0]):
                for _j in range(_hist.shape[1]):
                    
                    if _column_count >= len(_column_names) or _column_names[_column_count] == _column_names[-1]:
                        continue
                    
                    _ax = plt.subplot2grid((_hist.shape[0],_hist.shape[1]), (_i,_j))
                    sns.kdeplot(data={"Certain Distribution // DATAFRAME_ORIGINAL" : DATAFRAME_ORIGINAL[_column_names[_column_count]], 
                                      "Uncertain Distribution // DATAFRAME_SIMULATE" : _DATAFRAME_SIMULATE[_column_names[_column_count]]}, 
                                common_grid=True, 
                                legend = False, 
                                bw_method=_INIT_DATA_BANDWIDTH)
                    plt.title(_column_names[_column_count])
                    _ax.plot()
                    
                    _column_count += 1
                
            plt.tight_layout(pad=1)
            plt.show()
            
        else: 
            
            print("Error in chosen col-kde comparison mode!")




##########################################################################################################################
# experiments modul 2 - with simulation --> missing data (row wise) --> useage of kde of columns to simulate outcome
##########################################################################################################################

# impute == false is equal to stochastic simulation approach
if _SIMULATE == True:
    
    print("\nPredictions for dataset with uncertainties and simulated values:")
    


    # step 0 --> first for loop for getting row and simulating this specific row
    
    if _SIMULATION_RANGE == None:
        _SIMULATION_RANGE = range(len(_DATAFRAME_SIMULATE))
        
    
    """
        some necessary variables and for further computation or history collection
    """
    
    _scaler = MinMaxScaler()
    
    #x-axis ranges from 0 and 1 with .001 steps -- is also used for sigmoid accuracy
    # x-axis can be interpreted as sigmoid values between 0 and 1 with above mentioned steps (accuracy)
    if _SIMULATION_LENGTH < 10000:
        _x_axis = np.arange(0.0, 1.0, 0.00005)
    elif _SIMULATION_LENGTH <= 50000:
        _x_axis = np.arange(0.0, 1.0, 0.0005)
    else:
        _x_axis = np.arange(0.0, 1.0, 0.001)
        
        
        
    # helper to handle multiple _classes
    _classes = ["Class " + str(i) for i in range(_unique_outcomes)]
    
    
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
            "Softmax" : {
                "1.1.0_Input_RMSE" : [],
                "1.1.1_Means" : [],
                "1.1.2_Mean_Labels" : [],
                "1.1.3_Mean_Label_Frequenzy" : [],
                "1.1.4_Stds" : [],
                "1.1.5_Max_Density_Activation" : [],
                "1.1.6_Max_Density_Act_Label" : [],
                "1.1.7_Max_Density_Act_Label_Frequency" : [],
                "1.1.8_Lower_Bound_Probability" : [],
                "1.1.9_Upper_Bound_Probability" : []
                },
            "Sigmoid" : {
                "1.2.0_Input_RMSE" : [],
                "1.2.1_Means" : [],
                "1.2.2_Mean_Labels" : [],
                "1.2.3_Mean_Label_Frequenzy" : [],
                "1.2.4_Stds" : [],
                "1.2.5_Max_Density_Activation" : [],
                "1.2.6_Max_Density_Act_Label" : [],
                "1.2.7_Max_Density_Act_Label_Frequency" : [],
                "1.2.8_Lower_Bound_Probability" : [],
                "1.2.9_Upper_Bound_Probability" : []
                }
            },
        "2_Original_Simulation" : {
            "Softmax" : {
                "2.1.0_Input_RMSE" : [],
                "2.1.1_Means" : [],
                "2.1.2_Mean_Labels" : [],
                "2.1.3_Mean_Label_Frequenzy" : [],
                "2.1.4_Stds" : [],
                "2.1.5_Max_Density_Activation" : [],
                "2.1.6_Max_Density_Act_Label" : [],
                "2.1.7_Max_Density_Act_Label_Frequency" : [],
                "2.1.8_Lower_Bound_Probability" : [],
                "2.1.9_Upper_Bound_Probability" : []
                },
            "Sigmoid" : {
                "2.2.0_Input_RMSE" : [],
                "2.2.1_Means" : [],
                "2.2.2_Mean_Labels" : [],
                "2.2.3_Mean_Label_Frequenzy" : [],
                "2.2.4_Stds" : [],
                "2.2.5_Max_Density_Activation" : [],
                "2.2.6_Max_Density_Act_Label" : [],
                "2.2.7_Max_Density_Act_Label_Frequency" : [],
                "2.2.8_Lower_Bound_Probability" : [],
                "2.2.9_Upper_Bound_Probability" : []
                }
            }
        }
     
    # simulation row results is holding all the row wise informaiton form the simulation
    # index is equal to collected row results 
    SIMULATION_ROW_RESULTS = []
    
    
    """
        in the next step (for-loop) the main simulation part is carried out
            - the for loop will itterate through all rows inside a given dataset
            - in each cycle two main simulation predictions will be creaded
                - one given a true distribution (kde)
                - and one given a uncertain distribution (kde)
            
    """
    
    _sim_start_time = time.time()
    
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
        """
        
        
        # for each increament in the simulated row, a different random state will be used
        if _adjusted_random_state == None: pass
        else:  _adjusted_random_state+=1
        
        INPUT_COLL_UNCERTAIN = utils.generate_simulation_sample_collection(uncertain_attributes = _uncertain_attributes, 
                                                                           dataframe_categorical = _DATAFRAME_SIMULATE, 
                                                                           kde_collection = kde_collection_uncertain, 
                                                                           monte_carlo = _monte_carlo, 
                                                                           latin_hypercube = _latin_hypercube, 
                                                                           standardize_data = _standardize_data, 
                                                                           datatype_map = datatype_map, 
                                                                           column_names = _column_names,                                            
                                                                           simulation_length = _SIMULATION_LENGTH, 
                                                                           random_state = _adjusted_random_state, 
                                                                           lhs_mode = _LHS_MODE, 
                                                                           visualize_lhs_samples = _visualize_lhs_samples, 
                                                                           lhs_prefix = " Uncertain")
        
        
        # PART 2: COMBINE(IMPUTE) DATAFRAM WITH SAMPLES TO CREATE INPUT
        
        _X_Uncertain_INPUT = utils.generate_simulation_inputs(simulation_row = _DATAFRAME_SIMULATE_ROW, 
                                                              simulation_length = _SIMULATION_LENGTH, 
                                                              uncertain_attributes = _uncertain_attributes, 
                                                              sample_collection = INPUT_COLL_UNCERTAIN).iloc[:,:-1]
        
        
        
        
        
        
        # for each increament in the simulated row, a different random state will be used
        if _adjusted_random_state == None: pass
        else:  _adjusted_random_state+=3
        
        INPUT_COLL_ORIGINAL = utils.generate_simulation_sample_collection(uncertain_attributes = _uncertain_attributes, 
                                                                           dataframe_categorical = DATAFRAME_ORIGINAL, 
                                                                           kde_collection = kde_collection_uncertain, 
                                                                           monte_carlo = _monte_carlo, 
                                                                           latin_hypercube = _latin_hypercube, 
                                                                           standardize_data = _standardize_data, 
                                                                           datatype_map = datatype_map, 
                                                                           column_names = _column_names,                                            
                                                                           simulation_length = _SIMULATION_LENGTH, 
                                                                           random_state = _adjusted_random_state, 
                                                                           lhs_mode = _LHS_MODE, 
                                                                           visualize_lhs_samples = _visualize_lhs_samples, 
                                                                           lhs_prefix = " Original")
        
        
        # PART 2: COMBINE(IMPUTE) DATAFRAM WITH SAMPLES TO CREATE INPUT
        
        _X_ORIGINAL_INPUT = utils.generate_simulation_inputs(simulation_row = _DATAFRAME_SIMULATE_ROW, 
                                                             simulation_length = _SIMULATION_LENGTH, 
                                                             uncertain_attributes = _uncertain_attributes, 
                                                             sample_collection = INPUT_COLL_ORIGINAL).iloc[:,:-1]
        
        



        """
        #step 5.2.a: row-wise predictions on uncertain samples
            -----> Simulation procedure for uncertain kde induced simulation frames
        """
        
        # predictions and labels
        _y_simulation_uncertain_hat = model.predict(_X_Uncertain_INPUT, verbose=0)
        
        _y_simulation_uncertain_hat_labels = {
            "softmax_labels" : np.argmax(_y_simulation_uncertain_hat["softmax"], axis=1),
            "sigmoid_lables" : np.argmax(_y_simulation_uncertain_hat["sigmoid"], axis=1)    
            }
        
        
        # simulation parametric statistics
        _y_simulation_uncertain_hat_mean = {
            "softmax_mean" :  _y_simulation_uncertain_hat["softmax"].mean(axis=0),
            "sigmoid_mean" : _y_simulation_uncertain_hat["sigmoid"].mean(axis=0)
            }
        

        _y_simulation_uncertain_hat_mean_label = {
            "softmax_mean_label" : np.argmax(_y_simulation_uncertain_hat_mean["softmax_mean"]),
            "sigmoid_mean_label" : np.argmax(_y_simulation_uncertain_hat_mean["sigmoid_mean"])
            }


        _y_simulation_uncertain_hat_std = {
            "softmax_std" : _y_simulation_uncertain_hat["softmax"].std(axis=0),
            "sigmoid_std" : _y_simulation_uncertain_hat["sigmoid"].std(axis=0)
            }


        # simulation non-parametric statistics
        _uncertain_simulation_result_kde = {
            "softmax_class" : [],
            "sigmoid_class" : []
            }
        
        
        # calculate pdf of each class of each prediction
        for _activation in _uncertain_simulation_result_kde:
            
            _temp_uncertain_kde = []
            
            for _preds in _y_simulation_uncertain_hat[_activation.replace("_class", "")].transpose():
                _temp_uncertain_kde.append(stats.gaussian_kde(_preds, bw_method=_PRED_BANDWIDTH, weights=utils.adjust_edgeweight(_preds, _PRED_BANDWIDTH)))

            # to convert lists to dictionary
            _uncertain_simulation_result_kde[_activation] = {_classes[i] : _temp_uncertain_kde[i] for i in range(_unique_outcomes)}


        """
        _temp_uncertain_kde = []
        for i in _y_simulation_uncertain_hat["sigmoid"].transpose():
             _temp_uncertain_kde.append(stats.gaussian_kde(i, bw_method=_PRED_BANDWIDTH, weights=adjust_edgeweight(i)))

        
        _uncertain_simulation_result_kde["sigmoid_class"] = {_classes[i] : _temp_uncertain_kde[i] for i in range(_unique_outcomes)}
        """


        _uncertain_kde_pdfs = {
            "softmax_class" : {i : _uncertain_simulation_result_kde["softmax_class"][i].pdf(_x_axis) for i in _classes},
            "sigmoid_class" : {i : _uncertain_simulation_result_kde["sigmoid_class"][i].pdf(_x_axis) for i in _classes}
            }
        
        _uncertain_kde_density_peak_indices = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _uncertain_kde_density_peak_pdf = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _uncertain_kde_stats = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _uncertain_kde_lower_probability = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _uncertain_kde_upper_probability = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _uncertain_kde_sum_prob = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _uncertain_max_density_activation = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _uncertain_max_density_label = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        
        for _activation in _uncertain_simulation_result_kde:
            for _c in _classes:
                
                if _norm:
                    _uncertain_kde_pdfs[_activation][_c] = _scaler.fit_transform(_uncertain_kde_pdfs[_activation][_c].reshape(-1, 1)).reshape(-1) # TODO
                
                _uncertain_kde_density_peak_indices[_activation][_c] = scipy.signal.find_peaks(_uncertain_kde_pdfs[_activation][_c])[0]

                # if max peak value not in list, append to peak history
                _max_pdf = _uncertain_kde_pdfs[_activation][_c]
                
                if (np.argmax(_max_pdf) not in _uncertain_kde_density_peak_indices[_activation][_c]):
                    _uncertain_kde_density_peak_indices[_activation][_c] = np.append(_uncertain_kde_density_peak_indices[_activation][_c], np.argmax(_max_pdf))
                
                _uncertain_kde_density_peak_pdf[_activation][_c] = [_uncertain_kde_pdfs[_activation][_c][i] for i in _uncertain_kde_density_peak_indices[_activation][_c]]
    
                _uncertain_kde_stats[_activation][_c] = {int(_uncertain_kde_density_peak_indices[_activation][_c][i]) : _uncertain_kde_density_peak_pdf[_activation][_c][i] for i in range(len(_uncertain_kde_density_peak_indices[_activation][_c]))}
        
                # kde integral for percentages under the curve
                _uncertain_kde_lower_probability[_activation][_c] = round(_uncertain_simulation_result_kde[_activation][_c].integrate_box_1d(float("-inf"), 0.5), 8)
                _uncertain_kde_upper_probability[_activation][_c] = round(_uncertain_simulation_result_kde[_activation][_c].integrate_box_1d(0.5, float("inf")), 8)
                _uncertain_kde_sum_prob[_activation][_c] = round(_uncertain_kde_lower_probability[_activation][_c] + _uncertain_kde_upper_probability[_activation][_c], 2)
                
            
                _uncertain_max_density_activation[_activation][_c] = max(_uncertain_kde_stats[_activation][_c], key=_uncertain_kde_stats[_activation][_c].get) / len(_x_axis)
            _uncertain_max_density_label[_activation] = np.argmax(list(_uncertain_max_density_activation[_activation].values()))
        
        
        
        
        """
            #step 5.2.b: row-wise predictions on original samples
            -----> Simulation procedure for true original kde induced simulation frames
        """
        
        # predictions and labels
        _y_simulation_original_hat = model.predict(_X_ORIGINAL_INPUT, verbose=0)


        _y_simulation_original_hat_labels = {
            "softmax_labels" : np.argmax(_y_simulation_original_hat["softmax"], axis=1),
            "sigmoid_labels" : np.argmax(_y_simulation_original_hat["sigmoid"], axis=1)    
            }
        
        
        # simulation parametric statistics
        _y_simulation_original_hat_mean = {
            "softmax_mean" :  _y_simulation_original_hat["softmax"].mean(axis=0),
            "sigmoid_mean" : _y_simulation_original_hat["sigmoid"].mean(axis=0)
            }
        
        
        _y_simulation_original_hat_mean_label = {
            "softmax_mean_label" : np.argmax(_y_simulation_original_hat_mean["softmax_mean"]),
            "sigmoid_mean_label" : np.argmax(_y_simulation_original_hat_mean["sigmoid_mean"])
            }


        _y_simulation_original_hat_std = {
            "softmax_std" : _y_simulation_original_hat["softmax"].std(axis=0),
            "sigmoid_std" : _y_simulation_original_hat["sigmoid"].std(axis=0)
            }


        # simulation non-parametric statistics
        _original_simulation_result_kde = {
            "softmax_class" : [],
            "sigmoid_class" : []
            }
        
        
        # calculate pdf of each class of each prediction
        for _activation in _original_simulation_result_kde:
            
            _temp_original_kde = []
            
            for _preds in _y_simulation_original_hat[_activation.replace("_class", "")].transpose():
                _temp_original_kde.append(stats.gaussian_kde(_preds, bw_method=_PRED_BANDWIDTH, weights=utils.adjust_edgeweight(_preds, _PRED_BANDWIDTH)))

            # to convert lists to dictionary
            _original_simulation_result_kde[_activation] = {_classes[i] : _temp_original_kde[i] for i in range(_unique_outcomes)}


        """
        _temp_uncertain_kde = []
        for i in _y_simulation_uncertain_hat["sigmoid"].transpose():
             _temp_uncertain_kde.append(stats.gaussian_kde(i, bw_method=_PRED_BANDWIDTH, weights=adjust_edgeweight(i)))

        
        _uncertain_simulation_result_kde["sigmoid_class"] = {_classes[i] : _temp_uncertain_kde[i] for i in range(_unique_outcomes)}
        """


        _original_kde_pdfs = {
            "softmax_class" : {i : _original_simulation_result_kde["softmax_class"][i].pdf(_x_axis) for i in _classes},
            "sigmoid_class" : {i : _original_simulation_result_kde["sigmoid_class"][i].pdf(_x_axis) for i in _classes}
            }
        
        _original_kde_density_peak_indices = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _original_kde_density_peak_pdf = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _original_kde_stats = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _original_kde_lower_probability = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _original_kde_upper_probability = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _original_kde_sum_prob = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _original_max_density_activation = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        _original_max_density_label = {
            "softmax_class" : {i : 0 for i in _classes},
            "sigmoid_class" : {i : 0 for i in _classes}
            }
        
        
        for _activation in _original_simulation_result_kde:
            for _c in _classes:
                
                if _norm:
                    _original_kde_pdfs[_activation][_c] = _scaler.fit_transform(_original_kde_pdfs[_activation][_c].reshape(-1, 1)).reshape(-1) # TODO
                
                _original_kde_density_peak_indices[_activation][_c] = scipy.signal.find_peaks(_original_kde_pdfs[_activation][_c])[0]

                # if max peak value not in list, append to peak history
                _max_pdf = _original_kde_pdfs[_activation][_c]
                
                if (np.argmax(_max_pdf) not in _original_kde_density_peak_indices[_activation][_c]):
                    _original_kde_density_peak_indices[_activation][_c] = np.append(_original_kde_density_peak_indices[_activation][_c], np.argmax(_max_pdf))
                
                _original_kde_density_peak_pdf[_activation][_c] = [_original_kde_pdfs[_activation][_c][i] for i in _original_kde_density_peak_indices[_activation][_c]]
    
                _original_kde_stats[_activation][_c] = {int(_original_kde_density_peak_indices[_activation][_c][i]) : _original_kde_density_peak_pdf[_activation][_c][i] for i in range(len(_original_kde_density_peak_indices[_activation][_c]))}
        
                # kde integral for percentages under the curve
                _original_kde_lower_probability[_activation][_c] = round(_original_simulation_result_kde[_activation][_c].integrate_box_1d(float("-inf"), 0.5), 8)
                _original_kde_upper_probability[_activation][_c] = round(_original_simulation_result_kde[_activation][_c].integrate_box_1d(0.5, float("inf")), 8)
                _original_kde_sum_prob[_activation][_c] = round(_original_kde_lower_probability[_activation][_c] + _original_kde_upper_probability[_activation][_c], 2)
                
            
                _original_max_density_activation[_activation][_c] = max(_original_kde_stats[_activation][_c], key=_original_kde_stats[_activation][_c].get) / len(_x_axis)
            _original_max_density_label[_activation] = np.argmax(list(_original_max_density_activation[_activation].values()))



        for _activation in ["sigmoid", "softmax"]:
            
            _fig, _axs = plt.subplots(_unique_outcomes, 1, figsize=(17, 11))
            for _plot, _c in enumerate(_classes):
    
                # visualize predictions with hist plots
                sns.histplot(data={f"{_c} {_activation.capitalize()}. Activations"  : pd.DataFrame(_y_simulation_uncertain_hat[_activation]).iloc[:, _plot]}, 
                             bins=15, 
                             binrange=(0, 1),
                             alpha=0.3,
                             stat="count", 
                             kde=False, 
                             kde_kws={"cut":0},
                             ax=_axs[_plot]).set_title(label=f'Row: {_row} Class: {_plot} Uncertain {_activation.capitalize()} Activation. Plot - Miss-Rate: {_MISS_RATE} - Sim.-Length: {_SIMULATION_LENGTH}')
            
                _axs[_plot].axvline(x=pd.DataFrame(y_original_hat[_activation]).iloc[_row, _plot], linewidth=4, alpha=1, linestyle = "--", color = "red", label="Model Class Prediction")
            
                _axs[_plot].axvline(x=pd.Series(_y_simulation_uncertain_hat_mean[_activation + "_mean"]).loc[_plot], linewidth=4, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value")
                
                _axs[_plot].axvline(x=_uncertain_max_density_activation[_activation + "_class"][_c], color="black", linestyle = "-.", linewidth=4, label="Uncert. KDE Max Density") 

                _axs[_plot].axvline(x=pd.Series(y_original_categorical[_row]).loc[_plot], linewidth=8, linestyle = "-", color = "green", label="Original Class")
            
                _axs[_plot].legend(["Model Class Prediction", "Uncert. Sim. Mean Value", "Uncert. KDE Max Density", "Original Class Asign.", _activation.capitalize() + "Activations"])
            plt.show()
            
            
            
            fig, _axs = plt.subplots(_unique_outcomes, 1, figsize=(17, 11))
            for _plot, _c in enumerate(_classes):
    
                # visualize predictions with hist plots
                sns.histplot(data={f"{_c} {_activation.capitalize()} Activations"  : pd.DataFrame(_y_simulation_original_hat[_activation]).iloc[:, _plot]}, 
                             bins=15, 
                             binrange=(0, 1),
                             alpha=0.3,
                             stat="count", 
                             kde=False, 
                             kde_kws={"cut":0},
                             ax=_axs[_plot]).set_title(label=f'Row: {_row} Class: {_plot} Original {_activation.capitalize()} Activation. Plot - Miss-Rate: {_MISS_RATE} - Sim.-Length: {_SIMULATION_LENGTH}')
            
                _axs[_plot].axvline(x=pd.DataFrame(y_original_hat[_activation]).iloc[_row, _plot], linewidth=4, alpha=1, linestyle = "--", color = "red", label="Predicted Model Label")
            
                _axs[_plot].axvline(x=pd.Series(_y_simulation_uncertain_hat_mean[_activation + "_mean"]).loc[_plot], linewidth=4, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value")
                
                _axs[_plot].axvline(x=_original_max_density_activation[_activation + "_class"][_c], color="black", linestyle = "-.", linewidth=4, label="Original. KDE Max Density") 
                
                _axs[_plot].axvline(x=pd.Series(y_original_categorical[_row]).loc[_plot], linewidth=8, linestyle = "-", color = "green", label="Original Class")
                _axs[_plot].legend(["Model Class Prediction", "Uncert. Sim. Mean Value", "Uncert. KDE Max Density", "Original Class Asign.", _activation.capitalize() + "Activations"])
            plt.show()



        """
            #append simulation row results
        """
        
        SIMULATION_ROW_RESULTS.append({
            "0_Overall Row Data" : {
                "0.1_row_id" : _row,
                "0.2_dataset" : _dataset,
                "0.3_miss_rate" : _MISS_RATE,
                "0.4_miss_rate_%" : round(_DATAFRAME_SIMULATE_ROW.isnull().sum().sum() * 100 / len(_DATAFRAME_SIMULATE_ROW[:-1]),2),
                "0.5_Simulation_length" : _SIMULATION_LENGTH,
                "0.6_Simulated_row" : _DATAFRAME_SIMULATE_ROW,
                "0.7_uncertain_attributes" : _uncertain_attributes,
                },
            "1_Uncertain Simulation Collection" : {
                #"1.00_x_input_rmse" : _uncertain_sim_input_rmse,
                #"1.01_x_input_stats" : _X_simulation_uncertain.describe(),
                "1.02_y_simulation_hat" : _y_simulation_uncertain_hat,
                "1.03_y_simulation_hat_labels" : _y_simulation_uncertain_hat_labels,
                "1.04_label_frequency" : {"softmax_labels" : pd.Series(_y_simulation_uncertain_hat_labels["softmax_labels"]).value_counts(),
                                          "sigmoid_lables" : pd.Series(_y_simulation_uncertain_hat_labels["sigmoid_lables"]).value_counts()},
                "1.05_simulation_mean" : _y_simulation_uncertain_hat_mean,
                "1.06_simulation_std" : _y_simulation_uncertain_hat_std,
                "1.07_kde_pdfs" : _uncertain_kde_pdfs,
                "1.08_kde_peaks_and_indices" : _uncertain_kde_stats,
                "1.09_kde_max_density_activation" : _uncertain_max_density_activation,
                "1.10_kde_lower_bound_probability" : _uncertain_kde_lower_probability,
                "1.11_kde_upper_bound_probability" : _uncertain_kde_upper_probability,
                "1.12_kde_combined_probability" : _uncertain_kde_sum_prob
                },
            "2_Original Simulation Collection" : {
                #"0.00_x_input_rmse" : _original_sim_input_rmse,
                #"2.01_x_input_stats" : _X_simulation_original.describe(),
                "2.02_y_simulation_hat" : _y_simulation_original_hat,
                "2.03_y_simulation_hat_labels" : _y_simulation_original_hat_labels,
                "2.04_label_frequency" : {"softmax_labels" : pd.Series(_y_simulation_original_hat_labels["softmax_labels"]).value_counts(),
                                          "sigmoid_labels" : pd.Series(_y_simulation_original_hat_labels["sigmoid_labels"]).value_counts()},
                "2.05_simulation_mean" : _y_simulation_original_hat_mean,
                "2.06_simulation_std" : _y_simulation_original_hat_std,
                "2.07_kde_pdfs" : _original_kde_pdfs,
                "2.08_kde_peaks_and_indices" : _original_kde_stats,
                "2.09_kde_max_density_activation" : _original_max_density_activation,
                "2.10_kde_lower_bound_probability" : _original_kde_lower_probability,
                "2.11_kde_upper_bound_probability" : _original_kde_upper_probability,
                "2.12_kde_combined_probability" : _original_kde_sum_prob
                },
            })


        # simulation history appendix
        #SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.0_Input_RMSE"].append(_uncertain_sim_input_rmse)
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.1_Means"].append(_y_simulation_uncertain_hat_mean["softmax_mean"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.2_Mean_Labels"].append(_y_simulation_uncertain_hat_mean_label["softmax_mean_label"])       
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.4_Stds"].append(_y_simulation_uncertain_hat_std["softmax_std"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.5_Max_Density_Activation"].append(np.array(list(_uncertain_max_density_activation["softmax_class"].values())))
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.6_Max_Density_Act_Label"].append(_uncertain_max_density_label["softmax_class"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.8_Lower_Bound_Probability"].append(_uncertain_kde_lower_probability["softmax_class"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.9_Upper_Bound_Probability"].append(_uncertain_kde_upper_probability["softmax_class"])
        
        #SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.0_Input_RMSE"].append(_uncertain_sim_input_rmse)
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.1_Means"].append(_y_simulation_uncertain_hat_mean["sigmoid_mean"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.2_Mean_Labels"].append(_y_simulation_uncertain_hat_mean_label["sigmoid_mean_label"])       
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.4_Stds"].append(_y_simulation_uncertain_hat_std["sigmoid_std"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.5_Max_Density_Activation"].append(np.array(list(_uncertain_max_density_activation["sigmoid_class"].values())))
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.6_Max_Density_Act_Label"].append(_uncertain_max_density_label["sigmoid_class"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.8_Lower_Bound_Probability"].append(_uncertain_kde_lower_probability["sigmoid_class"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.9_Upper_Bound_Probability"].append(_uncertain_kde_upper_probability["sigmoid_class"])
        
        
        # simulation history appendix
        #SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.0_Input_RMSE"].append(_original_sim_input_rmse)
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.1_Means"].append(_y_simulation_original_hat_mean["softmax_mean"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.2_Mean_Labels"].append(_y_simulation_original_hat_mean_label["softmax_mean_label"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.4_Stds"].append(_y_simulation_original_hat_std["softmax_std"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.5_Max_Density_Activation"].append(np.array(list(_original_max_density_activation["softmax_class"].values())))
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.6_Max_Density_Act_Label"].append(_original_max_density_label["softmax_class"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.8_Lower_Bound_Probability"].append(_original_kde_lower_probability)
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.9_Upper_Bound_Probability"].append(_original_kde_upper_probability)

        #SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.0_Input_RMSE"].append(_original_sim_input_rmse)
        SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.1_Means"].append(_y_simulation_original_hat_mean["sigmoid_mean"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.2_Mean_Labels"].append(_y_simulation_original_hat_mean_label["sigmoid_mean_label"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.4_Stds"].append(_y_simulation_original_hat_std["sigmoid_std"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.5_Max_Density_Activation"].append(np.array(list(_original_max_density_activation["sigmoid_class"].values())))
        SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.6_Max_Density_Act_Label"].append(_original_max_density_label["sigmoid_class"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.8_Lower_Bound_Probability"].append(_original_kde_lower_probability["sigmoid_class"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.9_Upper_Bound_Probability"].append(_original_kde_upper_probability["sigmoid_class"])

    # some accessory time metrics for comparison 
    _elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - _sim_start_time))    
        
    SIMULATION_COLLECTION["0_Simulation_Info"]["0.8_elapsed_sim_time"] = str(_elapsed_time)
    

    SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.3_Mean_Label_Frequenzy"] = pd.Series(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.2_Mean_Labels"]).value_counts()
    SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.3_Mean_Label_Frequenzy"] = pd.Series(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.2_Mean_Labels"]).value_counts()
    
    SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.7_Max_Density_Act_Label_Frequency"] = pd.Series(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.6_Max_Density_Act_Label"]).value_counts()
    SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.7_Max_Density_Act_Label_Frequency"] = pd.Series(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.6_Max_Density_Act_Label"]).value_counts()
    
    
    SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.3_Mean_Label_Frequenzy"] = pd.Series(SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.2_Mean_Labels"]).value_counts()
    SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.3_Mean_Label_Frequenzy"] = pd.Series(SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.2_Mean_Labels"]).value_counts()
    
    SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.7_Max_Density_Act_Label_Frequency"] = pd.Series(SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.6_Max_Density_Act_Label"]).value_counts()
    SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.7_Max_Density_Act_Label_Frequency"] = pd.Series(SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.6_Max_Density_Act_Label"]).value_counts()   
    
    

    SIMULATION_COLLECTION["3_Uncert_vs_Orig_KDE"] = {
        "Softmax" : {
            "3.1.1_Explanation" : "Analysing the differences between Uncertain and Original KDE Simulations",
            "3.1.2_Sim_Mean_dist" : mse(SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.1_Means"], SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.1_Means"], squared=False),
            "3.1.4_Sim_Stds_dist" : mse(SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.4_Stds"], SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.4_Stds"], squared=False),
            "3.1.6_Sim_Max_Density_Sigmoid_dist" : mse(SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.5_Max_Density_Activation"], SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.5_Max_Density_Activation"], squared=False),
            },
        "Sigmoid" : {
            "3.2.1_Explanation" : "Analysing the differences between Uncertain and Original KDE Simulations",
            "3.2.2_Sim_Mean_dist" : mse(SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.1_Means"], SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.1_Means"], squared=False),
            "3.2.4_Sim_Stds_dist" : mse(SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.4_Stds"], SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.4_Stds"], squared=False),
            "3.2.6_Sim_Max_Density_Sigmoid_dist" : mse(SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.5_Max_Density_Activation"], SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.5_Max_Density_Activation"], squared=False),
            }
        }
        
  
 
    print('\n\nSimulation execution time:', _elapsed_time)






sys.exit()






if _save_simulated_results:
    
    # save results and collections to folder
    # file name contains "dataset", "miss rate" and "simulation range" as identifier
    
    _results_id = len(os.listdir(_results_path))
    _results_file_name = os.path.join(_results_path, str(_results_id) + "_" + _dataset + "_" + str(_MISS_RATE) + "_" + str(_SIMULATION_RANGE))
    
    """
    if _IMPUTE == True and _SIMULATE == True:
        pickle.dump({"SIMULATION_COLLECTION": SIMULATION_COLLECTION,
                     "SIMULATION_ROW_RESULTS" : SIMULATION_ROW_RESULTS,
                     "DATAFRAME_COMBINED_RESULTS" : DATAFRAME_COMBINED_RESULTS,
                     "DATAFRAME_COMBINED_DISTANCES" : DATAFRAME_COMBINED_DISTANCES,
                     "DATAFRAME_COMBINED_LABELS" : DATAFRAME_COMBINED_LABELS,
                     "DATAFRAME_COMBINED_LABELS_ANALYSIS" : DATAFRAME_COMBINED_LABELS_ANALYSIS
                     }, open(_results_file_name, "wb"))
    else:

    if True:
        pickle.dump({"SIMULATION_COLLECTION": SIMULATION_COLLECTION,
                     "SIMULATION_ROW_RESULTS" : SIMULATION_ROW_RESULTS,
                     }, open(_results_file_name, "wb"))
    """



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















