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
_dataset = "wdbc"


# set random state          
_RANDOM_STATE = 24

# other constants
_INIT_DATA_BANDWIDTH = None
_PRED_BANDWIDTH = 0.2 # --> if None (default) "scott" is used
_KDE_WEIGHTS = None


# further dataset settings
_standardize_data = True


# settings for visualization
_visiualize_data = True
_visualize_original_predictions = True
_visualize_imputed_predictions = True


# train or load model
_train_model = True
_save_new_model = False
_load_model = False


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
_SIMULATION_LENGTH = 100000
#_SIMULATION_RANGE = None
_SIMULATION_RANGE = range(566, 569, 1)
_simulation_visualizations = True

_save_simulated_results = False

_load_simulated_results = False
_load_results_id = 0


##########################################################################################################################
"""
    # load original datasets with full data
"""
##########################################################################################################################

    
# load data for climate modal simulation crashes dataset
if _dataset == "wdbc":
    
    with open(os.path.join(_dataset_path, _dataset + ".dat"), 'rb') as DATAFRAME_ORIGINAL:
        DATAFRAME_ORIGINAL = pd.read_table(DATAFRAME_ORIGINAL, sep=",", engine='python', header = None)
    
    # drop the first column (contains ids) and move the orig. second colum (contains outcomes) to the end
    y_original, DATAFRAME_ORIGINAL = [DATAFRAME_ORIGINAL.iloc[:,1].copy(), DATAFRAME_ORIGINAL.iloc[:, 2:].copy()]
    DATAFRAME_ORIGINAL = DATAFRAME_ORIGINAL.merge(y_original, left_index=True, right_index=True)
    
    # change string outcome values to type int
    DATAFRAME_ORIGINAL.iloc[:,-1].replace(['B', 'M'], [0, 1], inplace=True)
 
    
elif _dataset == "climate_simulation":
    
    with open(os.path.join(_dataset_path, _dataset + ".dat"), 'rb') as DATAFRAME_ORIGINAL:
        DATAFRAME_ORIGINAL = pd.read_table(DATAFRAME_ORIGINAL, sep="\s+", engine='python', header = 0)
    
    # drop the first two elements of the dataset -> not relevant
    DATAFRAME_ORIGINAL = DATAFRAME_ORIGINAL.iloc[:, 2:]
    

elif _dataset == "australian":
    
    with open(os.path.join(_dataset_path, _dataset + ".dat"), 'rb') as DATAFRAME_ORIGINAL:
        DATAFRAME_ORIGINAL = pd.read_table(DATAFRAME_ORIGINAL, sep=" ", engine="python", header=None)    
    

elif _dataset == "predict+students+dropout+and+academic+success":
    
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
    
    
"""
    variable unique_outcomes decides which kind of simulation has to be choosen,
    dependend on the dataset - counts possible classes of outcomes
"""
    
_unique_outcomes = len(DATAFRAME_ORIGINAL.Outcome.unique())

 

##########################################################################################################################
"""
    # standardization of values for better performance
"""
##########################################################################################################################
    

if _standardize_data:
    
    # use data scaler to norm the data (scaler used = MinM_axsclaer, values between 0 and 1)
    _scaler = MinMaxScaler()
    
    
    if _unique_outcomes == 2:
        
        DATAFRAME_ORIGINAL = pd.DataFrame(_scaler.fit_transform(DATAFRAME_ORIGINAL))
        DATAFRAME_ORIGINAL.columns = _column_names
        
        
    elif _unique_outcomes >= 3:
        
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
    _hist = DATAFRAME_ORIGINAL.hist(column=_column_names, bins=10, figsize=(20, 12), density=False, sharey=False, sharex=True)
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

if _unique_outcomes == 2:
    _X_original_train, _X_original_test, _y_original_train,  _y_original_test = train_test_split(X_original, 
                                                                                             y_original, 
                                                                                             test_size=0.25,
                                                                                             random_state=_RANDOM_STATE)

elif _unique_outcomes >= 3:
    
    # y labels have to be changed to categorical data
    y_original_categorical = keras.utils.to_categorical(y_original, num_classes=_unique_outcomes)
    
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
    #_x = layers.Dense(32, activation='relu')(_inputs)
    _x = layers.Dense(16, activation='relu')(_inputs)  
    #_x = layers.Dense(16, activation='relu')(_x)
    
    

    if _unique_outcomes == 2:
        
        """
            --> Binary Model 
        """
        
        _outputs = layers.Dense(1, activation="sigmoid")(_x)              
        # build model
        model = keras.Model(inputs=_inputs, outputs=_outputs)
        
        # compile model
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=["accuracy"])
        
        # fit model
        model_history = model.fit(_X_original_train, 
                                  _y_original_train, 
                                  validation_data=[_X_original_test, _y_original_test], 
                                  batch_size=15, 
                                  epochs=50, 
                                  verbose=0)
        
        if _save_new_model:
            # save new model
            model.save(os.path.join(_model_path, _dataset + "_binary_model.keras"))
        
    
    
    
    elif _unique_outcomes >= 3:
        
        """
            --> Multivariate Model 
        """
        
        _logits = layers.Dense(_unique_outcomes, activation=None)(_x)

        
        # model with two output heads
        #   1. Head: outputs the logits without any activation function
        #   2. Head: outputs a default softmax layer for classification
        model = keras.Model(inputs=_inputs, outputs={"logits" : _logits, #_logits,
                                                    "predictions" : tf.nn.softmax(_logits)})
        
        
        # compile model
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss={"logits": lambda y_true, y_pred: 0.0,
                            "predictions": keras.losses.CategoricalCrossentropy()},
                      metrics=["accuracy"])
        
        
        
        # fit model        
        model_history = model.fit(_X_original_train, 
                                  {'logits': _y_original_train, 
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
    
    
    if _unique_outcomes == 2:
        
        # loading and compiling saved model structure
        model = keras.models.load_model(os.path.join(_model_path, _dataset + "_binary_model.keras"))
        
        
    elif _unique_outcomes >= 3:
        
        # load model, but do not compile (because of custom layer). 
        # Compiling in a second step 
        model = keras.models.load_model(os.path.join(_model_path, _dataset + "_multi_model.keras"), compile=False)
        
        # compile model
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss={"logits": lambda y_true, y_pred: 0.0,
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

if _unique_outcomes == 2:
    
    """
        #   RESULTS // Original Precictions
    """
    y_original_hat = model.predict(X_original).flatten()
    y_original_hat_labels = (y_original_hat>0.5).astype("int32")
    y_original_hat_label_frequency = pd.Series(y_original_hat_labels).value_counts()
    
    
    if _visualize_original_predictions:

        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data={"sigmoid" : y_original_hat, "label" : y_original_hat_labels}, 
                     x="sigmoid", 
                     hue="label", 
                     bins=15, 
                     #binrange=(0, 1), 
                     stat="count", 
                     kde=False, 
                     kde_kws={"cut":0})
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Frequency')
        plt.title('Original (True) dataset combined output')
        plt.tight_layout()
        plt.show()
    
    
    
    
elif _unique_outcomes >= 3:
    
    y_original_hat = model.predict(X_original)
    y_original_hat_labels = np.argmax(y_original_hat["predictions"], axis=1)
    y_original_hat_label_frequency = pd.Series(y_original_hat_labels).value_counts()

    if _visualize_original_predictions:
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=y_original_hat_labels, 
                     bins=10, 
                     stat="count")
        plt.xlabel('Softmax Activations')
        plt.ylabel('Frequency')
        plt.title('True Combined Output Hist Plot')
        plt.tight_layout()
        plt.show()




if _get_original_prediction_metrics:
    
    if _unique_outcomes == 2:
        
        utils.create_metrics(y_original, y_original_hat_labels)
        plt.show()
    """
    elif _unique_outcomes >= 3:
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
                        sharex=True)
    plt.title('Input without missing data')
    plt.tight_layout()
    plt.show()
    
    
    # comparison of original and uncertain DATAFRAME    
    plt.figure(figsize=(12, 6))
    sns.histplot(data={"DATAFRAME_ORIGINAL" : np.array(DATAFRAME_ORIGINAL).flatten(), 
                       "DATAFRAME_MISS" : np.array(DATAFRAME_MISS).flatten()})
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
    
    if _unique_outcomes == 2:
        
        y_impute_hat = model.predict(X_impute).flatten()
        y_impute_hat_labels = (y_impute_hat>0.5).astype("int32")
        y_impute_hat_label_frequency = pd.Series(y_impute_hat_labels).value_counts()
        
        if _visualize_imputed_predictions:
            
            # visualize predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(data={"sigmoid" : y_impute_hat, "label" : y_impute_hat_labels}, 
                         x="sigmoid", 
                         hue="label", 
                     bins=15, 
                     binrange=(0, 1), 
                     stat="count", 
                     kde=False, 
                     kde_kws={"cut":0})
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Frequency')
            plt.title(f'Impute (uncertain) dataset combined output - Miss-Rate: {_MISS_RATE} - Impute-Method: {_IMPUTE_METHOD}')
            plt.tight_layout()
            plt.show()
            
            
        
        
    elif _unique_outcomes >= 3:
        
        y_impute_hat = model.predict(X_impute)
        y_impute_hat_labels = np.argmax(y_impute_hat["predictions"], axis=1)
        y_impute_hat_label_frequency = pd.Series(y_impute_hat_labels).value_counts()
            
        if _visualize_imputed_predictions:
            
            # visualize predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(data=y_impute_hat_labels, bins=10, stat="count", kde=False, kde_kws={"cut":0})
            plt.xlabel('Softmax Activations')
            plt.ylabel('Frequency')
            plt.title(f'Uncertain (deter.) Combined Output Hist Plot - Miss-Rate: {_MISS_RATE} - Impute-Method: {_IMPUTE_METHOD}')
            plt.tight_layout()
            plt.show()



    
    if _get_imputed_prediction_metrics:
        
        if _unique_outcomes == 2:
            
            utils.create_metrics(y_original, y_impute_hat_labels)
            plt.show()
        """   
        elif _unique_outcomes >= 3:
            
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
            
        
        
        
        if _unique_outcomes == 2:

            
            """
            #step 5.1.a: row-wise predictions on uncertain samples
                -----> Simulation procedure for uncertain kde induced simulation frames
            """

            # predictions and labels
            _y_simulation_uncertain_hat = model.predict(_X_simulation_uncertain, verbose=0).flatten()
            _y_simulation_uncertain_hat_labels = (_y_simulation_uncertain_hat>0.5).astype("int32")
            
            
            # simulation parametric statistics
            _y_simulation_uncertain_hat_mean = _y_simulation_uncertain_hat.mean()
            _y_simulation_uncertain_hat_mean_label = (_y_simulation_uncertain_hat.mean()>0.5).astype("int32")
            _y_simulation_uncertain_hat_std = _y_simulation_uncertain_hat.std()
            
            #### simulation non-parametric statistics
            _uncertain_simulation_result_kde = stats.gaussian_kde(_y_simulation_uncertain_hat, 
                                                     bw_method=_PRED_BANDWIDTH, 
                                                     weights=adjust_edgeweight(_y_simulation_uncertain_hat))
            
            _uncertain_kde_pdfs = _uncertain_simulation_result_kde.pdf(_x_axis)
            _uncertain_kde_density_peak_indices = scipy.signal.find_peaks(_uncertain_kde_pdfs)[0] 
            
            # if max peak value not in list, append to peak history
            if (np.argmax(_uncertain_kde_pdfs) not in _uncertain_kde_density_peak_indices):
                _uncertain_kde_density_peak_indices = np.append(_uncertain_kde_density_peak_indices, np.argmax(_uncertain_kde_pdfs))
                
            _uncertain_kde_density_peak_pdf = [_uncertain_kde_pdfs[i] for i in _uncertain_kde_density_peak_indices]
            _uncertain_kde_stats = {int(_uncertain_kde_density_peak_indices[i]) : _uncertain_kde_density_peak_pdf[i] for i in range(len(_uncertain_kde_density_peak_indices))}
            
            # kde integral for percentages under the curve
            #_uncertain_kde_lower_probability = round(_uncertain_simulation_result_kde.integrate_box_1d(float("-inf"), 0.5), 8)
            #_uncertain_kde_upper_probability = round(_uncertain_simulation_result_kde.integrate_box_1d(0.5, float("inf")), 8)
            _uncertain_kde_lower_probability = round(_uncertain_simulation_result_kde.integrate_box_1d(0, 0.5), 8)
            _uncertain_kde_upper_probability = round(_uncertain_simulation_result_kde.integrate_box_1d(0.5, 1), 8)
            _uncertain_kde_sum_prob = round(_uncertain_kde_lower_probability + _uncertain_kde_upper_probability, 2)
            
            _uncertain_max_density_sigmoid = max(_uncertain_kde_stats, key=_uncertain_kde_stats.get) / len(_x_axis)
            _uncertain_max_density_label = int(_uncertain_max_density_sigmoid>0.5)
            
            
            """
                #step 5.1.b: row-wise predictions on original samples
                -----> Simulation procedure for true original kde induced simulation frames
            """
            
            # predictions and labels
            _y_simulation_original_hat = model.predict(_X_simulation_original, verbose=0).flatten()
            _y_simulation_original_hat_labels = (_y_simulation_original_hat>0.5).astype("int32")
            
            # simulation parametric statistics
            _y_simulation_original_hat_mean = _y_simulation_original_hat.mean()
            _y_simulation_original_hat_mean_label = (_y_simulation_original_hat.mean()>0.5).astype("int32")
            _y_simulation_original_hat_std = _y_simulation_original_hat.std()
            
            
            #### simulation non-parametric statistics
            _original_simulation_result_kde = stats.gaussian_kde(_y_simulation_original_hat, 
                                                    bw_method=_PRED_BANDWIDTH, 
                                                    weights=adjust_edgeweight(_y_simulation_original_hat))
            
            _original_kde_pdfs = _original_simulation_result_kde.pdf(_x_axis) 
            _original_kde_density_peak_indices = scipy.signal.find_peaks(_original_kde_pdfs)[0]
            
            # if max peak value not in list, append to peak history
            if (np.argmax(_original_kde_pdfs) not in _original_kde_density_peak_indices):
                _original_kde_density_peak_indices = np.append(_original_kde_density_peak_indices, np.argmax(_original_kde_pdfs))
                
            _original_kde_density_peak_pdf = [_original_kde_pdfs[i] for i in _original_kde_density_peak_indices]
            _original_kde_stats = {int(_original_kde_density_peak_indices[i]) : _original_kde_density_peak_pdf[i] for i in range(len(_original_kde_density_peak_indices))}
            
            
            # kde integral for percentages under the curve
            #_original_kde_lower_probability = round(_original_simulation_result_kde.integrate_box_1d(float("-inf"), 0.5), 8)
            #_original_kde_upper_probability = round(_original_simulation_result_kde.integrate_box_1d(0.5, float("inf")), 8)
            _original_kde_lower_probability = round(_original_simulation_result_kde.integrate_box_1d(0, 0.5), 8)
            _original_kde_upper_probability = round(_original_simulation_result_kde.integrate_box_1d(0.5, 1), 8)
            _original_kde_sum_prob = round(_original_kde_lower_probability + _original_kde_upper_probability, 2)
            
            _original_max_density_sigmoid = max(_original_kde_stats, key=_original_kde_stats.get) / len(_x_axis)
            _original_max_density_label = int(_original_max_density_sigmoid>0.5)


            """
            test = sns.kdeplot(data={"uncert":_y_simulation_uncertain_hat,
                              "orig":_y_simulation_original_hat}, cumulative=True, bw_method=_PRED_BANDWIDTH)
            """
            """
            original_cdf = np.cumsum(_original_kde_pdfs / sum(_original_kde_pdfs))
            uncertain_cdf = np.cumsum(_uncertain_kde_pdfs / sum(_uncertain_kde_pdfs))
            
            plt.plot(_x_axis, original_cdf, label="original", color="orange")
            plt.plot(_x_axis, uncertain_cdf, label="uncertain", color="blue")
            """


            #kl_div_ff = np.sum(scipy.special.kl_div(_original_kde_pdfs, _uncertain_kde_pdfs))
            #kl_div_bw = np.sum(scipy.special.kl_div(_uncertain_kde_pdfs, _original_kde_pdfs))
 

            """
                #append simulation row results
            """
            
            SIMULATION_ROW_RESULTS.append({
                "0_Overall Row Data" : {
                    "0.1_row_id" : _row,
                    "0.2_dataset" : _dataset,
                    "0.3_miss_rate" : _MISS_RATE,
                    "0.4_Simulation_length" : _SIMULATION_LENGTH,
                    "0.5_Simulated_row" : _DATAFRAME_SIMULATE_ROW,
                    "0.6_uncertain_attributes" : _uncertain_attributes,
                    },
                "1_Uncertain Simulation Collection" : {
                    "1.01_x_input_stats" : _X_simulation_uncertain.describe(),
                    "1.02_y_simulation_hat" : _y_simulation_uncertain_hat,
                    "1.03_y_simulation_hat_labels" : _y_simulation_uncertain_hat_labels,
                    "1.04_label_frequency" : pd.Series(_y_simulation_uncertain_hat_labels).value_counts(),
                    "1.05_simulation_mean" : _y_simulation_uncertain_hat_mean,
                    "1.06_simulation_std" : _y_simulation_uncertain_hat_std,
                    "1.07_kde_pdfs" : _uncertain_kde_pdfs,
                    "1.08_kde_peaks_and_indices" : _uncertain_kde_stats,
                    "1.09_kde_max_density_sigmoid" : _uncertain_max_density_sigmoid,
                    "1.10_kde_lower_bound_probability" : _uncertain_kde_lower_probability,
                    "1.11_kde_upper_bound_probability" : _uncertain_kde_upper_probability,
                    "1.12_kde_combined_probability" : _uncertain_kde_sum_prob
                    },
                "2_Original Simulation Collection" : {
                    "2.01_x_input_stats" : _X_simulation_original.describe(),
                    "2.02_y_simulation_hat" : _y_simulation_original_hat,
                    "2.03_y_simulation_hat_labels" : _y_simulation_original_hat_labels,
                    "2.04_label_frequency" : pd.Series(_y_simulation_original_hat_labels).value_counts(),
                    "2.05_simulation_mean" : _y_simulation_original_hat_mean,
                    "2.06_simulation_std" : _y_simulation_original_hat_std,
                    "2.07_kde_pdfs" : _original_kde_pdfs,
                    "2.08_kde_peaks_and_indices" : _original_kde_stats,
                    "2.09_kde_max_density_sigmoid" : _original_max_density_sigmoid,
                    "2.10_kde_lower_bound_probability" : _original_kde_lower_probability,
                    "2.11_kde_upper_bound_probability" : _original_kde_upper_probability,
                    "2.12_kde_combined_probability" : _original_kde_sum_prob
                    },
                "3_Row_analysis" : {
                    "Explanation" : "Row differences of the uncertain kde simulation from the original kde simulation",
                    "Input_Differences" : _X_simulation_original.describe() - _X_simulation_uncertain.describe(),
                    "Label_Frequ_diff" : pd.Series(_y_simulation_original_hat_labels).value_counts() - pd.Series(_y_simulation_uncertain_hat_labels).value_counts(),
                    "Sim_Mean_dist" : np.abs(_y_simulation_original_hat_mean - _y_simulation_uncertain_hat_mean),
                    "Sim_Std_dist" : np.abs(_y_simulation_original_hat_std - _y_simulation_uncertain_hat_std),
                    "Sim_Max_Density_Sigmoid_dist" : np.abs(_original_max_density_sigmoid - _uncertain_max_density_sigmoid),
                    #"KS-Test" : stats.ks_2samp(_original_kde_pdfs, _uncertain_kde_pdfs)
                    }
                })

                        
            # simulation history appendix
            SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.1_Means"].append(_y_simulation_uncertain_hat_mean)
            SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.2_Mean_Labels"].append(_y_simulation_uncertain_hat_mean_label)       
            SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.4_Stds"].append(_y_simulation_uncertain_hat_std)
            SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.5_Max_Density_Sigmoid"].append(_uncertain_max_density_sigmoid)
            SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.6_Max_Density_Sig_Label"].append(_uncertain_max_density_label)
            SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.8_Lower_Bound_Probability"].append(_uncertain_kde_lower_probability)
            SIMULATION_COLLECTION["1_Uncertain_Simulation"]["1.9_Upper_Bound_Probability"].append(_uncertain_kde_upper_probability)
            
            # simulation history appendix
            SIMULATION_COLLECTION["2_Original_Simulation"]["2.1_Means"].append(_y_simulation_original_hat_mean)
            SIMULATION_COLLECTION["2_Original_Simulation"]["2.2_Mean_Labels"].append(_y_simulation_original_hat_mean_label)
            SIMULATION_COLLECTION["2_Original_Simulation"]["2.4_Stds"].append(_y_simulation_original_hat_std)
            SIMULATION_COLLECTION["2_Original_Simulation"]["2.5_Max_Density_Sigmoid"].append(_original_max_density_sigmoid)
            SIMULATION_COLLECTION["2_Original_Simulation"]["2.6_Max_Density_Sig_Label"].append(_original_max_density_label)
            SIMULATION_COLLECTION["2_Original_Simulation"]["2.8_Lower_Bound_Probability"].append(_original_kde_lower_probability)
            SIMULATION_COLLECTION["2_Original_Simulation"]["2.9_Upper_Bound_Probability"].append(_original_kde_upper_probability)



            # visualizations for binary simulation // comparison plots
            if _simulation_visualizations:
    
                _fig, _axs = plt.subplots(2, 1, figsize=(17, 11))
    
                """
                    Plot_5.1.a: Histogam which shows the uncertain kde simulated row sigmoid results with hue 
                """
                # visualize predictions with hist plots
                sns.histplot(data={"Sigmoid Activations" : _y_simulation_uncertain_hat,
                                   "label" : _y_simulation_uncertain_hat_labels}, 
                             x="Sigmoid Activations", 
                             hue="label", 
                             hue_order=[0, 1], 
                             bins=15, 
                             binrange=(0, 1), 
                             stat="count", 
                             kde=False, 
                             kde_kws={"cut":0},
                             ax=_axs[0]).set_title(label=f'Row: {_row} Uncertain KDE Sim. Output Hist. Plot - Miss-Rate: {_MISS_RATE} - Sim.-Length: {_SIMULATION_LENGTH}')
                
                _axs[0].axvline(x=y_original[_row], linewidth=4, linestyle = "-", color = "green", label="Original Label")
                _axs[0].axvline(x=y_original_hat_labels[_row], linewidth=2, alpha=1, linestyle = "--", color = "red", label="Predicted Model Label")
                
                if _IMPUTE:
                    _axs[0].axvline(x=y_impute_hat[_row], linewidth=2, linestyle = "--", color = "purple", label="Impute Prediction" + " (" + _IMPUTE_METHOD.capitalize() + ")") # impute prediction
                
                #plt.axvline(x=_y_simulation_original_hat_mean, linewidth=2, linestyle = "-.", color = "black", label="Orig. Mean Sim. Value") # orig. simulation prediction mean
                #plt.axvline(x=_y_simulation_uncertain_hat_mean, linewidth=2, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # uncert. simulation prediction mean
                
                # Max Density Vertical Lines
                _axs[0].axvline(x=_uncertain_max_density_sigmoid, color="grey", linestyle = "-.", linewidth=2, label="Uncert. KDE Max Density") 
                _axs[0].axvline(x=_original_max_density_sigmoid, color="black", linestyle = "-.", linewidth=2, label="Orig. KDE Max Density")
                
                _axs[0].legend(["Original Label", "Predicted Model Label", "Impute Prediction" + " (" + _IMPUTE_METHOD.capitalize() + ")", "Orig. Mean Sim. Value", "Uncert. Mean Sim. Value", "Label 1", "Label 0"])


                """
                    Plot_5.1.b: Histogam which shows the original kde simulated row sigmoid results with hue 
                """
                # visualize predictions with hist plots
                sns.histplot(data={"Sigmoid Activations" : _y_simulation_original_hat,
                                   "label" : _y_simulation_original_hat_labels}, 
                             x="Sigmoid Activations", 
                             hue="label", 
                             hue_order=[0, 1], 
                             bins=15, 
                             binrange=(0, 1), 
                             stat="count", 
                             kde=False, 
                             kde_kws={"cut":0},
                             ax=_axs[1])
                
                _axs[1].axvline(x=y_original[_row], linewidth=4, linestyle = "-", color = "green", label="Original Label")
                _axs[1].axvline(x=y_original_hat_labels[_row], linewidth=2, alpha=1, linestyle = "--", color = "red", label="Predicted Model Label")
                
                if _IMPUTE:
                    _axs[1].axvline(x=y_impute_hat[_row], linewidth=2, linestyle = "--", color = "purple", label="Impute Prediction" + " (" + _IMPUTE_METHOD.capitalize() + ")") # impute prediction
                
                # Max Density Vertical Lines
                _axs[1].axvline(x=_uncertain_max_density_sigmoid, color="grey", linestyle = "-.", linewidth=2, label="Uncert. KDE Max Density") 
                _axs[1].axvline(x=_original_max_density_sigmoid, color="black", linestyle = "-.", linewidth=2, label="Orig. KDE Max Density")
                
                #plt.axvline(x=_y_simulation_original_hat_mean, linewidth=2, linestyle = "-.", color = "black", label="Orig. Mean Sim. Value") # orig. simulation prediction mean
                #plt.axvline(x=_y_simulation_uncertain_hat_mean, linewidth=2, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # uncert. simulation prediction mean
                
                plt.title(f'Row: {_row} Original KDE Sim. Output Hist. Plot - Miss-Rate: {_MISS_RATE} - Sim.-Length: {_SIMULATION_LENGTH}')
                _axs[1].legend(["Original Label", "Predicted Model Label", "Impute Prediction" + " (" + _IMPUTE_METHOD.capitalize() + ")", "Orig. Mean Sim. Value", "Uncert. Mean Sim. Value", "Label 1", "Label 0"])
                plt.show()
                
                
                
                """
                    Plot_combined_output: KDE PLOT of Uncerlying uncertainty
                """

                # KDE Distributions
                plt.plot(_x_axis, _uncertain_kde_pdfs, 
                         label="Uncertain KDE Distribution", 
                         color="grey", linestyle = "--")
                
                plt.plot(_x_axis, _original_kde_pdfs, 
                         label="Original KDE Distribution", 
                         color="black", linestyle = "--")
                
                #plot normal distribution with mean and std of simulated values
                #plt.plot(_x_axis, stats.norm.pdf(_x_axis, _y_simulation_original_hat_mean, _y_simulation_original_hat_std), label="Orig. Sim. Distribution", color="black", linestyle = "-")
                #plt.plot(_x_axis, stats.norm.pdf(_x_axis, _y_simulation_uncertain_hat_mean, _y_simulation_uncertain_hat_std), label="Uncert. Sim. Distribution", color="grey", linestyle = "--")

                #plt.axvline(x=_y_simulation_uncertain_hat_mean, linewidth=1, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # mean uncertain kde prediction
                #plt.axvline(x=_y_simulation_original_hat_mean, linewidth=1, linestyle = "-.", color = "black", label="Orig. Mean Sim. Value") # mean original kde prediction
  

                # Max Density Vertical Lines
                plt.vlines(x=_uncertain_max_density_sigmoid, 
                           ymin = 0, ymax = max(_uncertain_kde_density_peak_pdf), 
                           color="grey", linestyle = "-.", linewidth=0.9, 
                           label="Uncert. KDE Max Density") 
                
                plt.vlines(x=_original_max_density_sigmoid, 
                           ymin = 0, ymax = max(_original_kde_density_peak_pdf), 
                           color="black", linestyle = "-.", linewidth=0.9, 
                           label="Orig. KDE Max Density")
                
                # Predicted and Original Label
                plt.axvline(x=y_original[_row], 
                            linewidth=4, linestyle = "-", 
                            color = "green", label="Original Label")
                
                plt.axvline(x=y_original_hat_labels[_row], 
                            linewidth=2, alpha=1, linestyle = "--", 
                            color = "red", label="Predicted Model Label")
                
                # Impute Vertical Line
                if _IMPUTE:
                    plt.axvline(x=y_impute_hat[_row], 
                                linewidth=1, linestyle = "--", color = "purple", 
                                label="Impute Prediction" + " (" + _IMPUTE_METHOD.capitalize() + ")")
                
                plt.title(f'Row: {_row} Underlying Uncertainty of the Simulation - Miss-Rate: {_MISS_RATE} - Sim.-Length: {_SIMULATION_LENGTH}')
                plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
                plt.xlabel('Sigmoid Activation')
                plt.ylabel('Density')
                plt.ylim([0, max(max(_uncertain_kde_pdfs), max(_original_kde_pdfs)) + 0.1])
                plt.show()
    
    
    

    
        if _unique_outcomes >= 3:
            
            
            """
            #step 5.2.a: row-wise predictions on uncertain samples
                -----> Simulation procedure for uncertain kde induced simulation frames
            """
            
            # predictions and labels
            y_uncertain_simulation_hat = model.predict(_X_simulation_uncertain, verbose=0)
            y_uncertain_simulation_hat_labels = np.argmax(y_uncertain_simulation_hat["predictions"], axis=1)     
            
            # simulation outcome and statistics
            y_uncertain_simulation_hat_softmax_mean = y_uncertain_simulation_hat["predictions"].mean(axis=0)
            y_uncertain_simulation_hat_logits_mean = y_uncertain_simulation_hat["logits"].mean(axis=0)
            
            y_uncertain_simulation_hat_softmax_std = y_uncertain_simulation_hat["predictions"].std(axis=0)
            y_uncertain_simulation_hat_logits_std = y_uncertain_simulation_hat["logits"].std(axis=0)
            
            y_uncertain_simulation_hat_label_frequency = pd.Series(y_uncertain_simulation_hat_labels).value_counts()
            
            
            """
                #step 5.2.b: row-wise predictions on original samples
                -----> Simulation procedure for true original kde induced simulation frames
            """
            
            # predictions and labels
            y_original_simulation_hat = model.predict(_X_simulation_original, verbose=0)
            y_original_simulation_hat_labels = np.argmax(y_original_simulation_hat["predictions"], axis=1)    
            
            # simulation outcome and statistics
            y_original_simulation_hat_softmax_mean = y_original_simulation_hat["predictions"].mean(axis=0)
            y_original_simulation_hat_logits_mean = y_original_simulation_hat["logits"].mean(axis=0)
            
            y_original_simulation_hat_softmax_std = y_original_simulation_hat["predictions"].std(axis=0)
            y_original_simulation_hat_logits_std = y_original_simulation_hat["logits"].std(axis=0)
            
            y_original_simulation_hat_label_frequency = pd.Series(y_original_simulation_hat_labels).value_counts()
    



            # single
            sns.kdeplot(y_original_simulation_hat["logits"], 
                        bw_method=_PRED_BANDWIDTH)  
            plt.title(f"Original Row - {_row} // Logit Plot")
            plt.show()
            
            sns.kdeplot(y_original_simulation_hat["predictions"], 
                        bw_method=_PRED_BANDWIDTH)  
            plt.title(f"Original Row - {_row} // Prediction Softmax Plot")
            plt.show()
            """
            # combined
            sns.kdeplot(y_original_simulation_hat["logits"].flatten(), 
                        bw_method=__PRED_BANDWIDTH)  
            plt.title(f"Original Row - {_row} // Logit Plot")
            plt.show()
            
            sns.kdeplot(y_original_simulation_hat["predictions"].flatten(), 
                        bw_method=__PRED_BANDWIDTH)  
            plt.title(f"Original Row - {_row} // Prediction Softmax Plot")
            plt.show()
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
    
    
    if _save_simulated_results:
        
        # save results and collections to folder
        # file name contains "dataset", "miss rate" and "simulation range" as identifier
        
        _results_id = len(os.listdir(_results_path))
        _results_file_name = os.path.join(_results_path, str(_results_id) + "_" + _dataset + "_" + str(_MISS_RATE) + "_" + str(_SIMULATION_RANGE))
        
        pickle.dump({"SIMULATION_COLLECTION": SIMULATION_COLLECTION,
                     "SIMULATION_ROW_RESULTS" : SIMULATION_ROW_RESULTS}, 
                    open(_results_file_name, "wb"))
        
        
    
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
    
    if _unique_outcomes == 2:       
        
        
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
        if _get_simulated_prediction_metrics:
            
            utils.create_metrics(y_original, uncertain_simulation_history_mean_labels)
            plt.show()
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

    







"""
    ---> Comparison of everything - Creation of extended dataframe containing all results
    
    Explanation of DATAFRAME_COMBINED_RESULTS:
        - Original Label is equal to the Label which is found originally in the dataset
        - 0: is the shortcut for a prediction with a trained model on full data without uncertainties
            -> only uncertainties found here are model uncertainties 
        - 1: is the shortcut for predictions with imputed values
        
        - 2: simulation results - metric mean 
        
"""

 # TODO
if _IMPUTE == True and _SIMULATE == True:
        
        min_idx = min(_SIMULATION_RANGE)
        max_idx = max(_SIMULATION_RANGE) + 1
        
        

        DATAFRAME_COMBINED_RESULTS = pd.DataFrame(data={"Original_Label" : y_original[min_idx:max_idx],
                                                        "0_Prediction" : y_original_hat[min_idx:max_idx],
                                                        "0_Predicted_Label" : y_original_hat_labels[min_idx:max_idx],
                                                        #"0_Prediction_Result" : (y_original[min_idx:max_idx] == y_original_hat_labels[min_idx:max_idx]),
                                                        "1_Imputation" : y_impute_hat[min_idx:max_idx],
                                                        "1_Imputation_Label" : y_impute_hat_labels[min_idx:max_idx],
                                                        #"1_Results_vs_Prediction_Label" : (y_original_hat_labels[min_idx:max_idx] == y_impute_hat_labels[min_idx:max_idx]),
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
                                                  
                                                  
                                                  
        DATAFRAME_COMBINED_ANALYSIS = pd.Series(data={"Correct labels assigned by model": DATAFRAME_COMBINED_RESULTS["0_Prediction_Result"].value_counts(True)[0],
                                                      "Correct labels assigned by imputation": DATAFRAME_COMBINED_RESULTS["1_Results_vs_Prediction_Label"].value_counts(True)[0],
                                                      "Correct labels assigned by simulation_unc_kde": DATAFRAME_COMBINED_RESULTS["2_U_Simulation_vs_Prediction_Label"].value_counts(True)[0],
                                                      "Correct labels assigned by imputation_ori_kde": DATAFRAME_COMBINED_RESULTS["3_O_Simulation_vs_Prediction_Label"].value_counts(True)[0]})
    












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

