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

from sklearn.metrics import mean_squared_error as mse
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
_train_model = True
_save_new_model = True
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
_SIMULATION_LENGTH = 100000
#_SIMULATION_RANGE = None
_SIMULATION_RANGE = range(0, 2, 1)
_simulation_visualizations = True
_norm= True
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

_unique_outcomes = len(DATAFRAME_ORIGINAL.Outcome.unique())

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
    utils.plot_history(model_history, model_type="multi")
    



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


# second part is a statement to check if a dataframe really exists and if not, a new one will be created even if load is true
if _load_dataframe_miss and Path(os.path.join(_dataset_path, "miss_frames", _dataset, _dataset + "_miss_rate_" + str(_MISS_RATE) + ".dat")).exists() and _create_dataframe_miss==False:
  
    """
        already created DATAFRAME_MISS will be loaded
    """

    DATAFRAME_MISS = pd.read_pickle(os.path.join(_dataset_path, "miss_frames", _dataset, _dataset + "_miss_rate_" + str(_MISS_RATE) + ".dat"))    
    
    
elif _create_dataframe_miss:
    
    """
        a new DATAFRAME_MISS will be created and saved
    """
    
    # if dataset folder does not exist, create a new one
    if Path(os.path.join(_dataset_path, "miss_frames", _dataset)).exists() == False:
        os.mkdir(os.path.join(_dataset_path, "miss_frames", _dataset)) 
    
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME_ORIGINAL, delete_mode =_DELETE_MODE, miss_rate=_MISS_RATE, random_seed=_RANDOM_STATE) 

    # save DATAFRAME_MISS to pickle.dat 
    DATAFRAME_MISS.to_pickle(os.path.join(_dataset_path, "miss_frames", _dataset, _dataset + "_miss_rate_" + str(_MISS_RATE) + ".dat"))
     


print("\nDataframe Statistics:")
print("Size of Original Dataframe:", DATAFRAME_ORIGINAL.size)
print(f"Deletion Settings: Mode={_DELETE_MODE} and Rate={_MISS_RATE}")
print("Deleted:", round(DATAFRAME_MISS.isnull().sum().sum()), "Values from Original")
# missing values per column
#DATAFRAME_MISS.isnull().mean() * 100
print("Missing data: ~" + str(round(DATAFRAME_MISS.isnull().sum().sum() * 100 / DATAFRAME_ORIGINAL.size, 2)), "%\n")



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

    
    DATAFRAME_IMPUTE_COLLECTION = {"MEAN_IMPUTE" : _DATAFRAME_MEAN_IMPUTE,
                               "MEDIAN_IMPUTE" : _DATAFRAME_MEDIAN_IMPUTE,
                               "MODE_IMPUTE" : _DATAFRAME_MODE_IMPUTE,
                               "KNN_IMPUTE" : _DATAFRAME_KNN_IMPUTE}
    
    
    DATAFRAME_IMPUTE_STATS = {"MEAN_IMPUTE" : _DATAFRAME_MEAN_IMPUTE.describe(),
                              "MEDIAN_IMPUTE" : _DATAFRAME_MEDIAN_IMPUTE.describe(),
                              "MODE_IMPUTE" : _DATAFRAME_MODE_IMPUTE.describe(),
                              "KNN_IMPUTE" : _DATAFRAME_KNN_IMPUTE.describe()}
    
    
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
    
    DATAFRAME_IMPUTE_RESULTS_COLLECTION = {}
    
    for _frame_key in DATAFRAME_IMPUTE_COLLECTION:
        
        print(f"Calculating results for dataframe: {_frame_key}")
        
        # create input frame for model predictions
        _X_impute = DATAFRAME_IMPUTE_COLLECTION[_frame_key].iloc[:, 0:-1]
         
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
    
    def kde_latin_hypercube_sampler(kde_collection, sim_length, random_state):
        
        """
            # @https://github.com/scipy/scipy/blob/v1.3.3/scipy/stats/kde.py#L439
            # function has been rewritten to hande latin hypercube sampling
        """
        
        # sample in 1 dimension with specific simulation length
        lhs_sampler = stats.qmc.LatinHypercube(1, seed=random_state)
        lhs_sample = lhs_sampler.random(n=sim_length)
        
        # get the underlying dataset of kde_collection
        kde_dataset = kde_collection.dataset.flatten()
    
        # scale the created lhs samples // mul with 100 for index
        _sample_scaled = stats.qmc.scale(lhs_sample, min(kde_dataset), max(kde_dataset)).flatten()
        #_uncertain_sample_scaled.sort()
        #uncertain_kde = kde_collection_uncertain[_key](_uncertain_sample_scaled)
        #plt.plot(np.linspace(0, 1, _SIMULATION_LENGTH), uncertain_kde)
    
        norm = np.transpose(np.random.multivariate_normal(np.zeros((kde_collection.d,), float),
                                             kde_collection.covariance, size=_SIMULATION_LENGTH))
        
        # original function index picker
        #indices = random.choice(kde_collection_uncertain[_key].n, size=10, p=kde_collection_uncertain[_key].weights)
        
        # new index picker with lhs functionality
        indices = np.array(_sample_scaled * 100, dtype="int32")
        
        means = kde_collection.dataset[:, indices]
        
        return means + norm
    
    
    
    
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
        
        # sample from uncertain and original kde for input imputation
        for _key in _uncertain_attributes:

            if _monte_carlo:
                
                # resample randomly a new dataset of the underlying kde
                uncertain_sample = kde_collection_uncertain[_key].resample(_SIMULATION_LENGTH, seed=_RANDOM_STATE).flatten()
                original_sample = kde_collection_original[_key].resample(_SIMULATION_LENGTH, seed=_RANDOM_STATE).flatten()
    
        
            if _latin_hypercube:
                
                _uncertain_sample = kde_latin_hypercube_sampler(kde_collection_uncertain[_key], _SIMULATION_LENGTH, _RANDOM_STATE).flatten()
                _original_sample = kde_latin_hypercube_sampler(kde_collection_original[_key], _SIMULATION_LENGTH, _RANDOM_STATE).flatten()
            
            
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


        # calculate Input RMSE for original and uncertain Monte Carlo Input
        
        #create an equivalent representation of a original row in monte carlo style of the original dataframe for comparison
        _original_df_mc_row_input = pd.DataFrame(X_original.loc[_row]).copy().transpose()
        _original_df_mc_row_input = pd.concat([_original_df_mc_row_input] * _SIMULATION_LENGTH, ignore_index=True)
        
        _uncertain_sim_input_rmse = mse(_original_df_mc_row_input, _X_simulation_uncertain, squared=False)
        _original_sim_input_rmse = mse(_original_df_mc_row_input, _X_simulation_original, squared=False)


        """
        #step 5.2.a: row-wise predictions on uncertain samples
            -----> Simulation procedure for uncertain kde induced simulation frames
        """
        
        # predictions and labels
        _y_simulation_uncertain_hat = model.predict(_X_simulation_uncertain, verbose=0)
        
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
                _temp_uncertain_kde.append(stats.gaussian_kde(_preds, bw_method=_PRED_BANDWIDTH, weights=adjust_edgeweight(_preds)))

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
        _y_simulation_original_hat = model.predict(_X_simulation_original, verbose=0)


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
                _temp_original_kde.append(stats.gaussian_kde(_preds, bw_method=_PRED_BANDWIDTH, weights=adjust_edgeweight(_preds)))

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
                "1.00_x_input_rmse" : _uncertain_sim_input_rmse,
                "1.01_x_input_stats" : _X_simulation_uncertain.describe(),
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
                "0.00_x_input_rmse" : _original_sim_input_rmse,
                "2.01_x_input_stats" : _X_simulation_original.describe(),
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
            "3_Row_analysis" : {
                "Explanation" : "Row differences of the uncertain kde simulation from the original kde simulation",
                "Input_Differences" : _X_simulation_original.describe() - _X_simulation_uncertain.describe(),
                #"Label_Frequ_diff" : pd.Series(_y_simulation_original_hat_labels).value_counts() - pd.Series(_y_simulation_uncertain_hat_labels).value_counts(),
                "Sim_Mean_dist" : {"softmax_mean_dist" : np.abs(_y_simulation_original_hat_mean["softmax_mean"] - _y_simulation_uncertain_hat_mean["softmax_mean"]),
                                   "sigmoid_mean_dist" : np.abs(_y_simulation_original_hat_mean["sigmoid_mean"] - _y_simulation_uncertain_hat_mean["sigmoid_mean"])},
                "Sim_Std_dist" : {"softmax_std_dist" : np.abs(_y_simulation_original_hat_std["softmax_std"] - _y_simulation_uncertain_hat_std["softmax_std"]),
                                   "sigmoid_std_dist" : np.abs(_y_simulation_original_hat_std["sigmoid_std"] - _y_simulation_uncertain_hat_std["sigmoid_std"])},
                "Sim_Max_Density_acivation_dist" : {"softmax_std_dist" : np.abs(np.array(list(_original_max_density_activation["softmax_class"].values())) - np.array(list(_uncertain_max_density_activation["softmax_class"].values()))),
                                                  "sigmoid_std_dist" : np.abs(np.array(list(_original_max_density_activation["sigmoid_class"].values())) - np.array(list(_uncertain_max_density_activation["sigmoid_class"].values())))},
                }
            })


        # simulation history appendix
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.0_Input_RMSE"].append(_uncertain_sim_input_rmse)
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.1_Means"].append(_y_simulation_uncertain_hat_mean["softmax_mean"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.2_Mean_Labels"].append(_y_simulation_uncertain_hat_mean_label["softmax_mean_label"])       
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.4_Stds"].append(_y_simulation_uncertain_hat_std["softmax_std"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.5_Max_Density_Activation"].append(np.array(list(_uncertain_max_density_activation["softmax_class"].values())))
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.6_Max_Density_Act_Label"].append(_uncertain_max_density_label["softmax_class"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.8_Lower_Bound_Probability"].append(_uncertain_kde_lower_probability["softmax_class"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Softmax"]["1.1.9_Upper_Bound_Probability"].append(_uncertain_kde_upper_probability["softmax_class"])
        
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.0_Input_RMSE"].append(_uncertain_sim_input_rmse)
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.1_Means"].append(_y_simulation_uncertain_hat_mean["sigmoid_mean"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.2_Mean_Labels"].append(_y_simulation_uncertain_hat_mean_label["sigmoid_mean_label"])       
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.4_Stds"].append(_y_simulation_uncertain_hat_std["sigmoid_std"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.5_Max_Density_Activation"].append(np.array(list(_uncertain_max_density_activation["sigmoid_class"].values())))
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.6_Max_Density_Act_Label"].append(_uncertain_max_density_label["sigmoid_class"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.8_Lower_Bound_Probability"].append(_uncertain_kde_lower_probability["sigmoid_class"])
        SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.9_Upper_Bound_Probability"].append(_uncertain_kde_upper_probability["sigmoid_class"])
        
        
        # simulation history appendix
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.0_Input_RMSE"].append(_original_sim_input_rmse)
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.1_Means"].append(_y_simulation_original_hat_mean["softmax_mean"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.2_Mean_Labels"].append(_y_simulation_original_hat_mean_label["softmax_mean_label"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.4_Stds"].append(_y_simulation_original_hat_std["softmax_std"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.5_Max_Density_Activation"].append(np.array(list(_original_max_density_activation["softmax_class"].values())))
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.6_Max_Density_Act_Label"].append(_original_max_density_label["softmax_class"])
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.8_Lower_Bound_Probability"].append(_original_kde_lower_probability)
        SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.9_Upper_Bound_Probability"].append(_original_kde_upper_probability)

        SIMULATION_COLLECTION["2_Original_Simulation"]["Sigmoid"]["2.2.0_Input_RMSE"].append(_original_sim_input_rmse)
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









# TODO
sys.exit()

 # TODO
if _IMPUTE == True and _SIMULATE == True:
        
        _min_idx = min(_SIMULATION_RANGE)
        _max_idx = max(_SIMULATION_RANGE) + 1


        metric="sigmoid"

        DATAFRAME_COMBINED_ROW_RESULTS = pd.DataFrame(data={"Original_Label" : y_original[_min_idx:_max_idx],
                                                        "0_Prediction" : y_original_hat[metric][_min_idx:_max_idx],
                                                        "0_Predicted_Label" : y_original_hat_label[metric][_min_idx:_max_idx],
                                                        
                                                        "1_Imputation-Mean" : DATAFRAME_IMPUTE_RESULTS_COLLECTION["MEAN_IMPUTE"][metric]["y_impute_hat"][_min_idx:_max_idx],
                                                        "1_Imputation-Mean_Label" : DATAFRAME_IMPUTE_RESULTS_COLLECTION["MEAN_IMPUTE"][metric]["y_impute_hat_labels"][_min_idx:_max_idx],    
                                                        "1_Imputation-Mode" : DATAFRAME_IMPUTE_RESULTS_COLLECTION["MODE_IMPUTE"][metric]["y_impute_hat"][_min_idx:_max_idx],
                                                        "1_Imputation-Mode_Label" : DATAFRAME_IMPUTE_RESULTS_COLLECTION["MODE_IMPUTE"][metric]["y_impute_hat_labels"][_min_idx:_max_idx],
                                                        "1_Imputation-Median" : DATAFRAME_IMPUTE_RESULTS_COLLECTION["MEDIAN_IMPUTE"][metric]["y_impute_hat"][_min_idx:_max_idx],
                                                        "1_Imputation-Median_Label" : DATAFRAME_IMPUTE_RESULTS_COLLECTION["MEDIAN_IMPUTE"][metric]["y_impute_hat_labels"][_min_idx:_max_idx],
                                                        "1_Imputation-KNNImp" : DATAFRAME_IMPUTE_RESULTS_COLLECTION["KNN_IMPUTE"][metric]["y_impute_hat"][_min_idx:_max_idx],
                                                        "1_Imputation-KNNIMP_Label" : DATAFRAME_IMPUTE_RESULTS_COLLECTION["KNN_IMPUTE"][metric]["y_impute_hat_labels"][_min_idx:_max_idx],
                                                        
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
                                                       
                                                       "1_Imp-KNNIMP_Label" : DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-Mean_Label"],
                                                       "1_Imp-KNNIMP_Label_Corr_Asign_Original" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["Original_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-KNNIMP_Label"]),
                                                       "1_Imp-KNNIMP_Label_Corr_Asign_Predicted" : (DATAFRAME_COMBINED_ROW_RESULTS.loc["0_Predicted_Label"] == DATAFRAME_COMBINED_ROW_RESULTS.loc["1_Imputation-KNNIMP_Label"]),
                                                       
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
                                                   "Uncertain_Mean_Sim_Input_RMSE" : np.mean(SIMULATION_COLLECTION["1_Uncertain_Simulation"]["Sigmoid"]["1.2.0_Input_RMSE"]),
                                                   "Original_Mean_Sim_Input_RMSE" : np.mean(SIMULATION_COLLECTION["2_Original_Simulation"]["Softmax"]["2.1.0_Input_RMSE"])
                                                   }, name="RMSE")


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















