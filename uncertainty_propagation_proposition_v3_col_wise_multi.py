# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:24:32 2023

@author: Selii
"""

import os
import sys
import pickle

from tqdm import tqdm
import time

import utils
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer



import scipy
import scipy.stats as stats



##########################################################################################################################
# set important paths
##########################################################################################################################


# set path to different folders
_dataset_path = os.path.join(os.getcwd(), 'datasets')
#image_path = os.path.join(os.getcwd(), 'images')
_model_path = os.path.join(os.getcwd(), 'models')
_results_path = os.path.join(os.getcwd(), 'sim_results')




##########################################################################################################################
"""
information about the datasets:
    -[1] wdbc - all attributes are considered continious - outcome is binary 
    -[2] climate_simulation - 
    -[3] australian - 
    -[4] predict+students+dropout+and+academic+success - three outcomes
    
following all the different settings for this simulation run can be found
    -dataset = "choose dataset"
    -standardize_dataset = "used for standardizing the dataset -- values between 0 and 1 -- minmax"
"""
##########################################################################################################################

#choose working dataset: "australian" or "climate_simulation", "wdbc" -> Breast Cancer Wisconsin (Diagnostic)
_dataset = "predict+students+dropout+and+academic+success"


# set random state          
_RANDOM_STATE = 24

# other constants
_INIT_DATA_BANDWIDTH = None
_PRED_BANDWIDTH = None # --> if None (default) "scott" is used
#_KDE_WEIGHTS = None


# further dataset settings
_standardize_data = True


# settings for visualization
_visiualize_data = True
_visualize_original_predictions = True
_visualize_imputed_predictions = True


# train or load model
_train_model = False
_save_new_model = True
_load_model = True


# prediction metrics
_get_original_prediction_metrics = True
_get_imputed_prediction_metrics = True
_get_simulated_prediction_metrics = False


# DATAFRAME_MISS settings - Introduction to missing values in the choosen Dataframe
# load DataFrame_MISS // if True, an already created one will be loaded, else a new one will be created
_load_dataframe_miss = True

_MISS_RATE=0.3


#KDE_VALUES OF EACH COLUMN - affected frames are DATAFRAME_SIMULATE -> Uncertain and DATAFRAME -> Certain/True
_compare_col_kde_distributions = False
_compare_col_kde_mode = "combined"    # choose between "single", "combined", "both"


# modes for deterministic/stochastic experiments on missing dataframes
# choose between kde_imputer, mean, median, most_frequent, KNNImputer
_IMPUTE = True
_IMPUTE_METHOD = "mean"

_SIMULATE = True
_SIMULATION_LENGTH = 100
#_SIMULATION_RANGE = None
_SIMULATION_RANGE = range(0, 1, 1)
_simulation_visualizations = True

_save_simulated_results = False

_load_simulated_results = False
_load_results_id = 0


##########################################################################################################################
"""
    # load original datasets with full data
"""
##########################################################################################################################

    
# load dataset
if _dataset == "predict+students+dropout+and+academic+success":
    
    with open(os.path.join(_dataset_path, _dataset + ".csv"), 'rb') as DATAFRAME_ORIGINAL:
        DATAFRAME_ORIGINAL = pd.read_csv(DATAFRAME_ORIGINAL, sep=";", engine="python")

    # change target names to numerical value
    DATAFRAME_ORIGINAL.iloc[:,-1].replace(['Dropout', 'Enrolled', "Graduate"], [0, 1, 2], inplace=True)


else:
    print("No valid dataset found!")
    
  
    
  
"""
    change all column names to standardized attribute names
"""  
    
_column_names = ["Attribute: " + str(i) for i in range(len(DATAFRAME_ORIGINAL.columns))]
_column_names[-1] = "Outcome"
DATAFRAME_ORIGINAL.columns = _column_names    

 

##########################################################################################################################
"""
    # standardization of values for better performance
"""
##########################################################################################################################
    

if _standardize_data:
    
    # use data scaler to norm the data (scaler used = MinM_axsclaer, values between 0 and 1)
    _scaler = MinMaxScaler()
    
    # steps for multi-label dataframe scaling
    # 1. drop outcome (labels shoud not be scaled)  
    y_original = DATAFRAME_ORIGINAL.iloc[:,-1].copy()
    DATAFRAME_ORIGINAL = DATAFRAME_ORIGINAL.iloc[:,:-1].copy()
    
    # 2. scale rest of dataframe
    DATAFRAME_ORIGINAL = pd.DataFrame(_scaler.fit_transform(DATAFRAME_ORIGINAL))
    
    # 3. add unscaled outcome back to scaled dataframe && and column names
    DATAFRAME_ORIGINAL = DATAFRAME_ORIGINAL.merge(y_original, left_index=True, right_index=True)
    DATAFRAME_ORIGINAL.columns = _column_names


DATAFRAME_ORIGINAL_STATS = DATAFRAME_ORIGINAL.describe()




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



# y labels have to be changed to categorical data - num classes is equal to len unique values in DATAFRAME
y_original_categorical = keras.utils.to_categorical(y_original, num_classes=len(DATAFRAME_ORIGINAL.Outcome.unique()))

_X_original_train, _X_original_test, _y_original_train, _y_original_test = train_test_split(X_original, 
                                                                                        y_original_categorical, 
                                                                                        test_size=0.25,
                                                                                        random_state=_RANDOM_STATE)




##########################################################################################################################
"""
    # create standard vanilla feed forward feural network
"""
##########################################################################################################################


if _train_model:
    
    # layers of the network
    _inputs = keras.Input(shape=(X_original.shape[1]))
    _x = layers.Dense(32, activation='relu')(_inputs)
    _x = layers.Dense(16, activation='relu')(_x)  
    #_x = layers.Dense(16, activation='relu')(_x)
    
    """
        --> Multivariate Model 
    """
    
    _sigmoid = layers.Dense(len(DATAFRAME_ORIGINAL.Outcome.unique()), activation=None)(_x)

    
    # model with two output heads
    #   1. Head: outputs the sigmoid without any activation function
    #   2. Head: outputs a default softmax layer for classification
    model = keras.Model(inputs=_inputs, outputs={"sigmoid" : tf.nn.sigmoid(_sigmoid), #_sigmoid,
                                                "predictions" : tf.nn.softmax(_sigmoid)})
    
    
    # compile model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss={"sigmoid": keras.losses.BinaryCrossentropy(),#lambda y_true, y_pred: 0.0,
                        "predictions": keras.losses.CategoricalCrossentropy()},
                  metrics=["accuracy"])
    
    
    
    # fit model        
    model_history = model.fit(_X_original_train, 
                              {'sigmoid': _y_original_train, 
                               'predictions': _y_original_train}, 
                              validation_data=[_X_original_test, _y_original_test], 
                              batch_size=15,
                              epochs=50,
                              verbose=0)
    
    if _save_new_model:
        # save new model
        model.save(os.path.join(_model_path, _dataset + "_multi_model.keras"))


    # plot model
    utils.plot_history(model_history)
    
    


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
                        "predictions": keras.losses.CategoricalCrossentropy()},
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

y_original_hat_labels_soft = np.argmax(y_original_hat["predictions"], axis=1)
y_original_hat_labels_sig = np.argmax(y_original_hat["sigmoid"], axis=1)

y_original_hat_label_soft_freq = pd.Series(y_original_hat_labels_soft).value_counts()
y_original_hat_label_sig_freq = pd.Series(y_original_hat_labels_soft).value_counts()

if _visualize_original_predictions:
    
    
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=y_original_hat_labels_soft, 
                 bins=10, 
                 stat="count")
    plt.xlabel('Softmax Activations')
    plt.ylabel('Frequency')
    plt.title('True Combined Output Hist Plot - Softmax')
    plt.tight_layout()
    plt.show()
    
    
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=y_original_hat_labels_sig, 
                 bins=10, 
                 stat="count")
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Frequency')
    plt.title('True Combined Output Hist Plot - Sigmoid')
    plt.tight_layout()
    plt.show()


if _get_original_prediction_metrics:
    
    """
    utils.create_metrics(y_original, y_original_hat_labels)
    plt.show()
    """


##########################################################################################################################
# introduce missing data - aka. aleatoric uncertainty
"""
    Here in this step a new DATAFRAME is introduced. This contains missing data with a specific missing rate in each row
"""
##########################################################################################################################


# second part is a statement to check if a dataframe really exists and if not, a new one will be created even if load is true
if _load_dataframe_miss and Path(os.path.join(_dataset_path, "miss_frames", _dataset, _dataset + "_miss_rate_" + str(_MISS_RATE) + ".dat")).exists():
  
    """
        already created DATAFRAME_MISS will be loaded
    """

    DATAFRAME_MISS = pd.read_pickle(os.path.join(_dataset_path, "miss_frames", _dataset, _dataset + "_miss_rate_" + str(_MISS_RATE) + ".dat"))    
    
else:
    
    """
        a new DATAFRAME_MISS will be created and saved
    """
    
    # if dataset folder does not exist, create a new one
    if Path(os.path.join(_dataset_path, "miss_frames", _dataset)).exists() == False:
        os.mkdir(os.path.join(_dataset_path, "miss_frames", _dataset)) 
    
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME_ORIGINAL.iloc[:, :-1], miss_rate=_MISS_RATE, random_seed=_RANDOM_STATE) 
    DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME_ORIGINAL.iloc[:,-1], left_index=True, right_index=True)

    # save DATAFRAME_MISS to pickle.dat 
    DATAFRAME_MISS.to_pickle(os.path.join(_dataset_path, "miss_frames", _dataset, _dataset + "_miss_rate_" + str(_MISS_RATE) + ".dat"))
    


# get statistics of DATAFRAME_MISS
DATAFRAME_MISS_STATS = DATAFRAME_MISS.describe()




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
    
    
if _IMPUTE and _IMPUTE_METHOD == "mean":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.copy()
    
    _simp_imp = SimpleImputer(strategy="mean")
    DATAFRAME_IMPUTE = pd.DataFrame(_simp_imp.fit_transform(DATAFRAME_IMPUTE), columns=_column_names)
    
    
elif _IMPUTE and _IMPUTE_METHOD == "median":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.copy()
    
    _simp_imp = SimpleImputer(strategy="median")
    DATAFRAME_IMPUTE = pd.DataFrame(_simp_imp.fit_transform(DATAFRAME_IMPUTE), columns=_column_names)


elif _IMPUTE and _IMPUTE_METHOD == "most_frequent":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.copy()
    
    _simp_imp = SimpleImputer(strategy="most_frequent")
    DATAFRAME_IMPUTE = pd.DataFrame(_simp_imp.fit_transform(DATAFRAME_IMPUTE), columns=_column_names)
    
    
elif _IMPUTE and _IMPUTE_METHOD == "KNNImputer":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.iloc[:,:-1].copy()
    
    _knn_imp = KNNImputer(n_neighbors=5)
    DATAFRAME_IMPUTE = pd.DataFrame(_knn_imp.fit_transform(DATAFRAME_IMPUTE), columns=_column_names)
    
    
if _IMPUTE:
    DATAFRAME_IMPUTE_STATS = DATAFRAME_IMPUTE.describe()
    
    
if _SIMULATE:
    _DATAFRAME_SIMULATE = DATAFRAME_MISS.copy()
    _SIMULATE_METHOD = "KDE_Simulation"


# exit if statement if no further simulations will be made
if _IMPUTE == False and _SIMULATE == False:
    sys.exit()




##########################################################################################################################
# experiments modul 1 - with imputation --> full data --> get_predictions
##########################################################################################################################

if _IMPUTE:
    
    print("\nPredictions for dataset with uncertainties and imputed values:")
    
    X_impute = DATAFRAME_IMPUTE.iloc[:, 0:-1]
    

    y_impute_hat = model.predict(X_impute)
    
    y_impute_hat_labels_soft = np.argmax(y_impute_hat["predictions"], axis=1)
    y_impute_hat_labels_sig = np.argmax(y_impute_hat["sigmoid"], axis=1)
    
    y_impute_hat_label_soft_freq = pd.Series(y_impute_hat_labels_soft).value_counts()
    y_impute_hat_label_sig_freq = pd.Series(y_impute_hat_labels_sig).value_counts()
    
    
    if _visualize_imputed_predictions:
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=y_impute_hat_labels_soft, bins=10, stat="count", kde=False, kde_kws={"cut":0})
        plt.xlabel('Softmax Activations')
        plt.ylabel('Frequency')
        plt.title(f'Uncertain (deter.) Combined Output Hist Plot - Miss-Rate: {_MISS_RATE} - Impute-Method: {_IMPUTE_METHOD} - Softmax')
        plt.tight_layout()
        plt.show()


        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=y_impute_hat_labels_soft, bins=10, stat="count", kde=False, kde_kws={"cut":0})
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Frequency')
        plt.title(f'Uncertain (deter.) Combined Output Hist Plot - Miss-Rate: {_MISS_RATE} - Impute-Method: {_IMPUTE_METHOD} - Sigmoid')
        plt.tight_layout()
        plt.show()

    
    if _get_imputed_prediction_metrics:
        
        """   
        utils.create_metrics(y_original, y_impute_hat_labels)
        plt.show()
        """



##########################################################################################################################
# experiments module 2 -- col wise simulations ----------> get kde values of dataframe
##########################################################################################################################

"""
    DISCLAIMER: DATAFRAME_SIMULATE is equal to DATAFRAME_MISS (including missing values) - naming because
"""


if _SIMULATE:
    
    """
        KDE COLLECTION -- ORIGINAL 
        --> is equal to the true distribution of the underlying data of the specific dataset
        --> to be able to get the true distribution we will use the original dataset with full certain data and no missing values
    """


    kde_collection_original = []
    
    for _column in _column_names:
        
        # get the kde of all values inside a column of the dataset
        _column_values = DATAFRAME_ORIGINAL[_column].values
        
        kde = stats.gaussian_kde(_column_values, bw_method=_INIT_DATA_BANDWIDTH)   
        kde_collection_original.append(kde)
        
        
    
    
    """
        KDE COLLECTION -- UNCERTAIN 
        --> is equal to the uncertain distribution of the underlying data of the specific dataset with missing values
        --> for the uncertain distribution we will use the dataset including missing data (=uncertain data) 
        --> for computing, all of the missing data has to be dropped first, to retrieve the uncertain distribution of the rest
    """
    
    
    kde_collection_uncertain = []
    
    for _column in _column_names:
        
        # drop all missing values and get kde of remaining values inside a column
        _column_values = _DATAFRAME_SIMULATE[_column].dropna().values
        
        kde = stats.gaussian_kde(_column_values, bw_method=_INIT_DATA_BANDWIDTH)   
        kde_collection_uncertain.append(kde)
        
        


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



    """
        main collection of kde distributions 
    """

    # to convert lists to dictionary
    kde_collection_original = {_column_names[i]: kde_collection_original[i] for i in range(len(_column_names))}
        
    # to convert lists to dictionary
    kde_collection_uncertain = {_column_names[i]: kde_collection_uncertain[i] for i in range(len(_column_names))}




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
    
    # Weighting and clipping
    # Amount of density below 0 & above 1
    def adjust_edgeweight(y_hat):
        
        # @https://andrewpwheeler.com/2021/06/07/kde-plots-for-predicted-probabilities-in-python/
        
        # if chosen kde bandwidth is not a number, reuturn weights 0 and compute default values
        if type(_PRED_BANDWIDTH) not in [int, float]:
            edgeweight = None
            return edgeweight
        
        below_0 = stats.norm.cdf(x=0, loc=y_hat, scale=_PRED_BANDWIDTH)
        above_1 = 1 - stats.norm.cdf(x=1, loc=y_hat, scale=_PRED_BANDWIDTH)
        
        edgeweight = 1 / (1 - below_0 - above_1)
        
        return edgeweight
    
    
    #x-axis ranges from 0 and 1 with .001 steps -- is also used for sigmoid accuracy
    # x-axis can be interpreted as sigmoid values between 0 and 1 with above mentioned steps (accuracy)
    if _SIMULATION_LENGTH <= 10.000:
        _x_axis = np.arange(0.0, 1.0, 0.000005)
    else:
        _x_axis = np.arange(0.0, 1.0, 0.001)
    
    """
    # simulation collection is holding all summarized information
    SIMULATION_COLLECTION = {
        "0_Simulation_Info" : {
            "0.1_random_state" : _RANDOM_STATE,
            "0.2_dataset" : _dataset,
            "0.3_dataset_size" : DATAFRAME_ORIGINAL.size,
            "0.4_miss_rate" : _MISS_RATE,
            "0.5_num_missing" : DATAFRAME_MISS.isnull().sum().sum(),
            "0.6_simulation_length" : _SIMULATION_LENGTH,
            "0.7_elapsed_sim_time" : "",
            "0.8_simulated_rows" : len(_SIMULATION_RANGE)
            },
        "1_Uncertain_Simulation" : {
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
    """ 
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
        
        _uncertain_sample_collection = []
        _original_sample_collection = [] 
        
        for _key in _uncertain_attributes:
            
            _uncertain_sample = kde_collection_uncertain[_key].resample(_SIMULATION_LENGTH, seed=_RANDOM_STATE).flatten()
            _original_sample = kde_collection_original[_key].resample(_SIMULATION_LENGTH, seed=_RANDOM_STATE).flatten()


            # if standardize is true and values x are x < 0 or x > 1, then set x respectively to 0 or 1
            if _standardize_data:
                
                _uncertain_sample[(_uncertain_sample < 0)] = 0
                _uncertain_sample[(_uncertain_sample > 1)] = 1
                
                _original_sample[(_original_sample < 0)] = 0
                _original_sample[(_original_sample > 1)] = 1


            _uncertain_sample_collection.append(_uncertain_sample)
            _original_sample_collection.append(_original_sample)
        
    
        _uncertain_sample_collection = pd.DataFrame(_uncertain_sample_collection).transpose()
        _uncertain_sample_collection.columns = _uncertain_attributes
    
        _original_sample_collection = pd.DataFrame(_original_sample_collection).transpose()
        _original_sample_collection.columns = _uncertain_attributes
    
    
        """
            # step 4: create DATAFRAME for faster simulation (basis input) and replace missing values with sampled ones   
            # index length of DATAFRAME_MISS_ROW is now equal to number of simulations
        """
        
        _DATAFRAME_MC_FOUNDATION = _DATAFRAME_SIMULATE_ROW.copy().transpose()
        _DATAFRAME_MC_FOUNDATION = pd.concat([_DATAFRAME_MC_FOUNDATION] * _SIMULATION_LENGTH, ignore_index=True)
        
        
        # basis dataframe used for uncertain kde simulation
        _DATAFRAME_MC_UNCERTAIN_KDE_SIMULATION = _DATAFRAME_MC_FOUNDATION.copy()
        
        
        # basis dataframe used for original (true) kde simulation
        _DATAFRAME_MC_ORIGINAL_KDE_SIMULATION = _DATAFRAME_MC_FOUNDATION.copy()
        
        
        # replace the missing values of DATAFRAME_MISS_ROW/ (now MC_SIMULATION) with the created samples 
        for _col in _uncertain_attributes:
            
            _DATAFRAME_MC_UNCERTAIN_KDE_SIMULATION[_col] = _uncertain_sample_collection[_col]
            _DATAFRAME_MC_ORIGINAL_KDE_SIMULATION[_col] = _original_sample_collection[_col]
        
        
        """
            step 5: main predictions on collected samples/data
        """
        
        _X_simulation_uncertain = _DATAFRAME_MC_UNCERTAIN_KDE_SIMULATION.iloc[:, 0:-1]
        
        _X_simulation_original = _DATAFRAME_MC_ORIGINAL_KDE_SIMULATION.iloc[:, 0:-1]
            
        

        """
        #step 5.2.a: row-wise predictions on uncertain samples
            -----> Simulation procedure for uncertain kde induced simulation frames
        """
        
        # predictions and labels
        y_uncertain_simulation_hat = model.predict(_X_simulation_uncertain, verbose=0)
        
        y_uncertain_simulation_hat_labels = np.argmax(y_uncertain_simulation_hat["predictions"], axis=1)     
        y_uncertain_simulation_hat_labels = np.argmax(y_uncertain_simulation_hat["sigmoid"], axis=1)    
        
        # simulation outcome and statistics
        y_uncertain_simulation_hat_softmax_mean = y_uncertain_simulation_hat["predictions"].mean(axis=0)
        y_uncertain_simulation_hat_sigmoid_mean = y_uncertain_simulation_hat["sigmoid"].mean(axis=0)
        
        y_uncertain_simulation_hat_softmax_std = y_uncertain_simulation_hat["predictions"].std(axis=0)
        y_uncertain_simulation_hat_sigmoid_std = y_uncertain_simulation_hat["sigmoid"].std(axis=0)
        
        y_uncertain_simulation_hat_label_soft_freq = pd.Series(y_uncertain_simulation_hat_labels).value_counts()
        y_uncertain_simulation_hat_label_sig_frequency = pd.Series(y_uncertain_simulation_hat_labels).value_counts()
        
        """
            #step 5.2.b: row-wise predictions on original samples
            -----> Simulation procedure for true original kde induced simulation frames
        """
        
        # predictions and labels
        y_original_simulation_hat = model.predict(_X_simulation_original, verbose=0)
        
        y_original_simulation_hat_labels = np.argmax(y_original_simulation_hat["predictions"], axis=1)    
        y_original_simulation_hat_labels = np.argmax(y_original_simulation_hat["sigmoid"], axis=1)    
        
        # simulation outcome and statistics
        y_original_simulation_hat_softmax_mean = y_original_simulation_hat["predictions"].mean(axis=0)
        y_original_simulation_hat_sigmoid_mean = y_original_simulation_hat["sigmoid"].mean(axis=0)
        
        y_original_simulation_hat_softmax_std = y_original_simulation_hat["predictions"].std(axis=0)
        y_original_simulation_hat_sigmoid_std = y_original_simulation_hat["sigmoid"].std(axis=0)
        
        y_original_simulation_hat_label_soft_freq = pd.Series(y_original_simulation_hat_labels).value_counts()
        y_original_simulation_hat_label_sig_freq = pd.Series(y_original_simulation_hat_labels).value_counts()



        # single
        sns.kdeplot(y_original_simulation_hat["sigmoid"], 
                    bw_method=_PRED_BANDWIDTH)  
        plt.title(f"Original Row - {_row} // Logit Plot")
        plt.show()
        
        sns.kdeplot(y_original_simulation_hat["predictions"], 
                    bw_method=_PRED_BANDWIDTH)  
        plt.title(f"Original Row - {_row} // Prediction Softmax Plot")
        plt.show()
        
        
        test = pd.DataFrame(y_original_simulation_hat["sigmoid"], columns=["Label: " + str(i) for i in range(len(DATAFRAME_ORIGINAL.Outcome.unique()))])
        

        from sklearn.neighbors import KernelDensity
        import matplotlib.gridspec as grid_spec
        
        gs = grid_spec.GridSpec(len(test.columns),1)
        fig = plt.figure(figsize=(16,9))
        
        i = 0
        
        ax_objs = []
        for label in test.columns:
            
            data = test[label].values
            
            x_d = _x_axis
        
            kde = stats.gaussian_kde(data).logpdf(x_d)
        
            # creating new axes object
            ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        
            # plotting the distribution
            ax_objs[-1].plot(x_d, kde, color="#f0f0f0",lw=1)
            ax_objs[-1].fill_between(x_d, kde, alpha=1)
        
        
            # setting uniform x and y lims
            ax_objs[-1].set_xlim(0,1)
            ax_objs[-1].set_ylim(0,2.5)
        
            # make background transparent
            rect = ax_objs[-1].patch
            rect.set_alpha(0)
        
            # remove borders, axis ticks, and labels
            ax_objs[-1].set_yticklabels([])
        
            if i == len(test.columns)-1:
                ax_objs[-1].set_xlabel("Test Score", fontsize=16,fontweight="bold")
            else:
                ax_objs[-1].set_xticklabels([])
        
            spines = ["top","right","left","bottom"]
            for s in spines:
                ax_objs[-1].spines[s].set_visible(False)
        
            #adj_country = country.replace(" ","\n")
            #ax_objs[-1].text(-0.02,0,adj_country,fontweight="bold",fontsize=14,ha="right")
        
        
            i += 1
        
        gs.update(hspace=-0.7)
        
        fig.text(0.07,0.85,"Distribution of Aptitude Test Results from 18 – 24 year-olds",fontsize=20)
        
        plt.tight_layout()
        plt.show()
        
        
        """
        # combined
        sns.kdeplot(y_original_simulation_hat["sigmoid"].flatten(), 
                    bw_method=__PRED_BANDWIDTH)  
        plt.title(f"Original Row - {_row} // Logit Plot")
        plt.show()
        
        sns.kdeplot(y_original_simulation_hat["predictions"].flatten(), 
                    bw_method=__PRED_BANDWIDTH)  
        plt.title(f"Original Row - {_row} // Prediction Softmax Plot")
        plt.show()
        """
            
    """    
    # some accessory time metrics for comparison 
    _elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - _sim_start_time  ))    
        
    SIMULATION_COLLECTION["0_Simulation_Info"]["0.7_elapsed_sim_time"] = str(_elapsed_time)
    
    SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.3_Mean_Label_Frequenzy"] = pd.Series(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.2_Mean_Labels"]).value_counts()
    SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.7_Max_Density_Sig_Label_Frequency"] = pd.Series(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.6_Max_Density_Sig_Label"]).value_counts()
    SIMULATION_COLLECTION["2_Original_Simulation"]["2.3_Mean_Label_Frequenzy"] = pd.Series(SIMULATION_COLLECTION["2_Original_Simulation"]["2.2_Mean_Labels"]).value_counts()
    SIMULATION_COLLECTION["2_Original_Simulation"]["2.7_Max_Density_Sig_Label_Frequency"] = pd.Series(SIMULATION_COLLECTION["2_Original_Simulation"]["2.6_Max_Density_Sig_Label"]).value_counts()
            
    
    SIMULATION_COLLECTION["3_Uncert_vs_Orig_KDE"] = {
        "3.1_Explanation" : "Analysing the differences between Uncertain and Original KDE Simulations",
        "3.2_Sim_Mean_dist" : np.abs(np.array(SIMULATION_COLLECTION["2_Original_Simulation"]["2.1_Means"]) - np.array(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.1_Means"])),
        "3.3_Sim_Mean_dist_avg" : None,
        "3.4_Sim_Stds_dist" : np.abs(np.array(SIMULATION_COLLECTION["2_Original_Simulation"]["2.4_Stds"]) - np.array(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.4_Stds"])),
        "3.5_Sim_Stds_dist_avg" : None,
        "3.6_Sim_Max_Density_Sigmoid_dist" : np.abs(np.array(SIMULATION_COLLECTION["2_Original_Simulation"]["2.5_Max_Density_Sigmoid"]) - np.array(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.5_Max_Density_Sigmoid"])),
        "3.7_Sim_Max_Density_Sigmoid_dist_avg" : None
        }
        
    SIMULATION_COLLECTION["3_Uncert_vs_Orig_KDE"]["3.3_Sim_Mean_dist_avg"] = SIMULATION_COLLECTION["3_Uncert_vs_Orig_KDE"]["3.2_Sim_Mean_dist"].mean()
    SIMULATION_COLLECTION["3_Uncert_vs_Orig_KDE"]["3.5_Sim_Stds_dist_avg"] = SIMULATION_COLLECTION["3_Uncert_vs_Orig_KDE"]["3.4_Sim_Stds_dist"].mean()
    SIMULATION_COLLECTION["3_Uncert_vs_Orig_KDE"]["3.7_Sim_Max_Density_Sigmoid_dist_avg"] = SIMULATION_COLLECTION["3_Uncert_vs_Orig_KDE"]["3.6_Sim_Max_Density_Sigmoid_dist"].mean()
    

    print('\n\nSimulation execution time:', _elapsed_time)
    """
    """
            --------------------------------> simulations process end <--------------------------------
            --------------------------------> simulations process end <--------------------------------
            --------------------------------> simulations process end <--------------------------------
            --------------------------------> simulations process end <--------------------------------
            --------------------------------> simulations process end <--------------------------------
            --------------------------------> simulations process end <--------------------------------
            --------------------------------> simulations process end <--------------------------------
            --------------------------------> simulations process end <--------------------------------
    """
    

    
    """
        Below: combined Comparisons between the prediction results of Uncertain and Certain KDE simulations
    """
    







"""
    ---> Comparison of everything - Creation of extended dataframe containing all results
    
    Explanation of DATAFRAME_COMBINED_RESULTS:
        - Original Label is equal to the Label which is found originally in the dataset
        - 0: is the shortcut for a prediction with a trained model on full data without uncertainties
            -> only uncertainties found here are model uncertainties 
        - 1: is the shortcut for predictions with imputed values
        
        - 2: simulation results - metric mean 
        
"""




"""
 # TODO
if _IMPUTE == True and _SIMULATE == True:
        
        _min_idx = min(_SIMULATION_RANGE)
        _max_idx = max(_SIMULATION_RANGE) + 1


        DATAFRAME_COMBINED_RESULTS = pd.DataFrame(data={"Original_Label" : y_original[_min_idx:_max_idx],
                                                        "0_Prediction" : y_original_hat[_min_idx:_max_idx],
                                                        "0_Predicted_Label" : y_original_hat_labels[_min_idx:_max_idx],
                                                        #"0_Prediction_Result" : (y_original[_min_idx:_max_idx] == y_original_hat_labels[_min_idx:_max_idx]),
                                                        "1_Imputation" : y_impute_hat[_min_idx:_max_idx],
                                                        "1_Imputation_Label" : y_impute_hat_labels[_min_idx:_max_idx],
                                                        #"1_Results_vs_Prediction_Label" : (y_original_hat_labels[_min_idx:_max_idx] == y_impute_hat_labels[_min_idx:_max_idx]),
                                                        "2_Orig_Sim_Mean" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.1_Means"],
                                                        "2_Orig_Sim_Label" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.2_Mean_Labels"],
                                                        "2_Orig_Sim_Std" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.4_Stds"],
                                                        "2_Orig_Sim_Max_Density" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.5_Max_Density_Sigmoid"],
                                                        "2_Orig_Sim_Max_Density_Label" : SIMULATION_COLLECTION["2_Original_Simulation"]["2.6_Max_Density_Sig_Label"],
                                                        
                                                        "3_Uncert_Sim_Mean" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.1_Means"],
                                                        "3_Uncert_Sim_Label" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.2_Mean_Labels"],
                                                        "3_Uncert_Sim_Std" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.4_Stds"],
                                                        "3_Uncert_Sim_Max_Density" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.5_Max_Density_Sigmoid"],
                                                        "3_Uncert_Sim_Max_Density_Label" : SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.6_Max_Density_Sig_Label"],
                                                        }).transpose()
                                                  
        
        DATAFRAME_COMBINED_DISTANCES = pd.DataFrame(data={"0_Prediction" : DATAFRAME_COMBINED_RESULTS.loc["0_Prediction"],
                                                          "1_Imputation_distance_to_Prediction" : np.abs(np.array(DATAFRAME_COMBINED_RESULTS.loc["0_Prediction"]) - np.array(DATAFRAME_COMBINED_RESULTS.loc["1_Imputation"])),
                                                          "2_Orig_Sim_Mean_distance_to_Prediction" : np.abs(np.array(DATAFRAME_COMBINED_RESULTS.loc["0_Prediction"]) - np.array(DATAFRAME_COMBINED_RESULTS.loc["2_Orig_Sim_Mean"])),
                                                          "2_Orig_Sim_Max_Density_distance_to_Prediction" : np.abs(np.array(DATAFRAME_COMBINED_RESULTS.loc["0_Prediction"]) - np.array(DATAFRAME_COMBINED_RESULTS.loc["2_Orig_Sim_Max_Density"])),
                                                          "3_Uncert_Sim_Mean_distance_to_Prediction" : np.abs(np.array(DATAFRAME_COMBINED_RESULTS.loc["0_Prediction"]) - np.array(DATAFRAME_COMBINED_RESULTS.loc["3_Uncert_Sim_Mean"])),
                                                          "3_Uncert_Sim_Max_Density_distance_to_Prediction" : np.abs(np.array(DATAFRAME_COMBINED_RESULTS.loc["0_Prediction"]) - np.array(DATAFRAME_COMBINED_RESULTS.loc["3_Uncert_Sim_Max_Density"])),
                                                          }).transpose()
                                                          
                                                          
        
        DATAFRAME_COMBINED_LABELS = pd.DataFrame(data={"Original_Label" : DATAFRAME_COMBINED_RESULTS.loc["Original_Label"],
                                                       "0_Predicted_Label" : DATAFRAME_COMBINED_RESULTS.loc["0_Predicted_Label"],
                                                       "0_Predicted_Label_Corr_Asign" : (DATAFRAME_COMBINED_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_RESULTS.loc["0_Predicted_Label"]),
                                                       
                                                       "1_Imputation_Label" : DATAFRAME_COMBINED_RESULTS.loc["1_Imputation_Label"],
                                                       "1_Imputation_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_RESULTS.loc["1_Imputation_Label"]),
                                                       "1_Imputation_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_RESULTS.loc["1_Imputation_Label"]),
                                                       
                                                       "2_Orig_Sim_Label" : DATAFRAME_COMBINED_RESULTS.loc["2_Orig_Sim_Label"],
                                                       "2_Orig_Sim_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_RESULTS.loc["2_Orig_Sim_Label"]),
                                                       "2_Orig_Sim_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_RESULTS.loc["2_Orig_Sim_Label"]),
                                                       
                                                       "2_Orig_Sim_Max_Density_Label" : DATAFRAME_COMBINED_RESULTS.loc["2_Orig_Sim_Max_Density_Label"],
                                                       "2_Orig_Sim_Max_Density_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_RESULTS.loc["2_Orig_Sim_Max_Density_Label"]),
                                                       "2_Orig_Sim_Max_Density_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_RESULTS.loc["2_Orig_Sim_Max_Density_Label"]),
                                                       
                                                       "3_Uncert_Sim_Label" : DATAFRAME_COMBINED_RESULTS.loc["3_Uncert_Sim_Label"],
                                                       "3_Uncert_Sim_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_RESULTS.loc["3_Uncert_Sim_Label"]),
                                                       "3_Uncert_Sim_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_RESULTS.loc["3_Uncert_Sim_Label"]),
                                                       
                                                       "3_Uncert_Sim_Max_Density_Label" : DATAFRAME_COMBINED_RESULTS.loc["3_Uncert_Sim_Max_Density_Label"],
                                                       "3_Uncert_Sim_Max_Density_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_RESULTS.loc["3_Uncert_Sim_Max_Density_Label"]),
                                                       "3_Uncert_Sim_Max_Density_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_RESULTS.loc["3_Uncert_Sim_Max_Density_Label"]),
                                                       }).transpose()
        
        

    
        DATAFRAME_COMBINED_LABELS_ANALYSIS = pd.Series(data={"Correct Orig. labels assigned by 0_Predicted_Label": DATAFRAME_COMBINED_LABELS.loc["0_Predicted_Label_Corr_Asign"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 1_Imputation_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imputation_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 1_Imputation_Label": DATAFRAME_COMBINED_LABELS.loc["1_Imputation_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 2_Orig_Sim_Label": DATAFRAME_COMBINED_LABELS.loc["2_Orig_Sim_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 2_Orig_Sim_Label": DATAFRAME_COMBINED_LABELS.loc["2_Orig_Sim_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 2_Orig_Sim_Max_Density_Label": DATAFRAME_COMBINED_LABELS.loc["2_Orig_Sim_Max_Density_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 2_Orig_Sim_Max_Density_Label": DATAFRAME_COMBINED_LABELS.loc["2_Orig_Sim_Max_Density_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 3_Uncert_Sim_Label": DATAFRAME_COMBINED_LABELS.loc["3_Uncert_Sim_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 3_Uncert_Sim_Label": DATAFRAME_COMBINED_LABELS.loc["3_Uncert_Sim_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             
                                                             "Correct Orig. labels assigned by 3_Uncert_Sim_Max_Density_Label": DATAFRAME_COMBINED_LABELS.loc["3_Uncert_Sim_Max_Density_Label_Corr_Asign_Original"].value_counts(True)[0],
                                                             "Correct Pred. labels assigned by 3_Uncert_Sim_Max_Density_Label": DATAFRAME_COMBINED_LABELS.loc["3_Uncert_Sim_Max_Density_Label_Corr_Asign_Predicted"].value_counts(True)[0],
                                                             })
    
"""


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





# general comparison of distributions (original, impute, sim-uncert, sim-orig)
"""         ### general comparison
kde_collection_impute = []

for _column in _column_names:
    
    # drop all missing values and get kde of remaining values inside a column
    _column_values = DATAFRAME_IMPUTE[_column].values
    
    kde = stats.gaussian_kde(_column_values, bw_method=_INIT_DATA_BANDWIDTH)   
    kde_collection_impute.append(kde)

kde_collection_impute = {_column_names[i]: kde_collection_impute[i] for i in range(len(_column_names))}

gen_original_sample = []
gen_impute_sample = []
gen_uncertain_sample = [] 

for _key in _column_names[:-1]:
    
    _original_sample = kde_collection_original[_key].resample(_SIMULATION_LENGTH, seed=_RANDOM_STATE).flatten()
    _impute_sample = kde_collection_impute[_key].resample(_SIMULATION_LENGTH, seed=_RANDOM_STATE).flatten()
    _uncertain_sample = kde_collection_uncertain[_key].resample(_SIMULATION_LENGTH, seed=_RANDOM_STATE).flatten()


    # if standardize is true and values x are x < 0 or x > 1, then set x respectively to 0 or 1
    if _standardize_data:
        
        _uncertain_sample[(_uncertain_sample < 0)] = 0
        _uncertain_sample[(_uncertain_sample > 1)] = 1
        
        _original_sample[(_original_sample < 0)] = 0
        _original_sample[(_original_sample > 1)] = 1
        
        _impute_sample[(_impute_sample < 0)] = 0
        _impute_sample[(_impute_sample > 1)] = 1


    gen_original_sample.append(_uncertain_sample)
    gen_uncertain_sample.append(_original_sample)
    gen_impute_sample.append(_impute_sample)


gen_original_sample = pd.DataFrame(gen_original_sample).transpose()
gen_original_sample.columns = _column_names[:-1]

gen_impute_sample = pd.DataFrame(gen_impute_sample).transpose()
gen_impute_sample.columns = _column_names[:-1]

gen_uncertain_sample = pd.DataFrame(gen_uncertain_sample).transpose()
gen_uncertain_sample.columns = _column_names[:-1]



gen_original_pred = model.predict(gen_original_sample).flatten()
gen_impute_pred = model.predict(gen_impute_sample).flatten()
gen_uncertain_pred = model.predict(gen_uncertain_sample).flatten()


sns.kdeplot({"original":gen_original_pred,
             "impute":gen_impute_pred,
             "uncertain":gen_uncertain_pred},
            fill=False)
"""


# ade evaluationen


# 3. Prediction with neural networks:
    # 3.1 Certain Uncertainty predictions with neural networks (point predictions)
        # predictions with neural networks - consequenzes of these predictions -> transition to 3.2
        # 3.1.1 binary
        # 3.1.2 softmax multi-variate predictions
    # 3.2 Uncertain Predictions, deterministic solution (standard approach)
        # showcasing predictions with neural neutworks and introduced data uncertainty
    # 3.3 Uncertain Predictions, stochastic solution (simulationn procedure)

# 4. Showcase of results und discussion of above three
    # binary results == 2
    
    # multivariate results unique_outputs >= 3

# Summary and discussion

# Limitations and model uncertainty predictions (bayes &&&)



#test = SIMULATION_ROW_RESULTS[0]["1_Uncertain Simulation Collection"]["1.3_y_simulation_hat"]

