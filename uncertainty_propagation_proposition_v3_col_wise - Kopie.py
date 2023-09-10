# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:24:32 2023

@author: Selii
"""

import os
import sys
#import pickle
#import random
from tqdm import tqdm

import utils
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#import seaborn.objects as so

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
#from sklearn.neighbors import KernelDensity

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
#from sklearn.impute import IterativeImputer

#from scipy.stats import gaussian_kde
import scipy
import scipy.stats as stats
#from sklearn.neighbors import KernelDensity


#import chaospy as cp


##########################################################################################################################
# set important paths
##########################################################################################################################


# set path to different folders
_dataset_path = os.path.join(os.getcwd(), 'datasets')
#image_path = os.path.join(os.getcwd(), 'images')
_model_path = os.path.join(os.getcwd(), 'models')




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


# set random state          # TODO fix RANOMSTATE and check if tensorflow random seed works
_RANDOM_STATE = 24
tf.random.set_seed(_RANDOM_STATE)
"""
np.random.seed(RANDOM_STATE)
np.random.RandomState(RANDOM_STATE)
"""



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
_get_original_prediction_metrics = True
_get_imputed_prediction_metrics = True
_get_simulated_prediction_metrics = True


# DATAFRAME_MISS settings - Introduction to missing values in the choosen Dataframe
# load DataFrame_MISS // if True, an already created one will be loaded, else a new one will be created
_load_dataframe_miss = True

_MISS_RATE=0.3


#KDE_VALUES OF EACH COLUMN - affected frames are DATAFRAME_SIMULATE -> Uncertain and DATAFRAME -> Certain/True
_compare_col_kde_distributions = True
_compare_col_kde_mode = "combined"    # choose between "single", "combined", "both"


# modes for deterministic/stochastic experiments on missing dataframes
# choose between kde_imputer, SimpleImputer//mean, SimpleImputer//median, SimpleImputer//most_frequent, KNNImputer
_IMPUTE = True
_IMPUTE_METHOD = "SimpleImputer//mean"

_SIMULATE = True
_SIMULATION_LENGTH = 100000
#_SIMULATION_RANGE = None
_SIMULATION_RANGE = range(5, 6, 1)
_simulation_visualizations = True




##########################################################################################################################
"""
    # load original datasets with full data
"""
##########################################################################################################################

    
# load data for climate modal simulation crashes dataset
if _dataset == "wdbc":
    
    with open(os.path.join(_dataset_path, _dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep=",", engine='python', header = None)
    
    # drop the first column (contains ids) and move the orig. second colum (contains outcomes) to the end
    y_complete, DATAFRAME = [DATAFRAME.iloc[:,1].copy(), DATAFRAME.iloc[:, 2:].copy()]
    DATAFRAME = DATAFRAME.merge(y_complete, left_index=True, right_index=True)
    
    # change string outcome values to type int
    DATAFRAME.iloc[:,-1].replace(['B', 'M'], [0, 1], inplace=True)
 
    
elif _dataset == "climate_simulation":
    
    with open(os.path.join(_dataset_path, _dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep="\s+", engine='python', header = 0)
    
    # drop the first two elements of the dataset -> not relevant
    DATAFRAME = DATAFRAME.iloc[:, 2:]
    

elif _dataset == "australian":
    
    with open(os.path.join(_dataset_path, _dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep=" ", engine="python", header=None)    
    

elif _dataset == "predict+students+dropout+and+academic+success":
    
    with open(os.path.join(_dataset_path, _dataset + ".csv"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_csv(DATAFRAME, sep=";", engine="python")

    # change target names to numerical value
    DATAFRAME.iloc[:,-1].replace(['Dropout', 'Enrolled', "Graduate"], [0, 1, 2], inplace=True)


else:
    print("No valid dataset found!")
    
  
    
  
"""
    change all column names to standardized attribute names
"""  
    
column_names = ["Attribute: " + str(i) for i in range(len(DATAFRAME.columns))]
column_names[-1] = "Outcome"
DATAFRAME.columns = column_names    
    
    
"""
    variable unique_outcomes decides which kind of simulation has to be choosen,
    dependend on the dataset - counts possible classes of outcomes
"""
    
_unique_outcomes = len(DATAFRAME.Outcome.unique())

 

##########################################################################################################################
"""
    # standardization of values for better performance
"""
##########################################################################################################################
    

if _standardize_data:
    
    # use data scaler to norm the data (scaler used = MinMaxSclaer, values between 0 and 1)
    _scaler = MinMaxScaler()
    
    
    if _unique_outcomes == 2:
        
        DATAFRAME = pd.DataFrame(_scaler.fit_transform(DATAFRAME))
        DATAFRAME.columns = column_names
        
        
    elif _unique_outcomes >= 3:
        
        # steps for multi-label dataframe scaling
        # 1. drop outcome (labels shoud not be scaled)  
        y_complete = DATAFRAME.iloc[:,-1].copy()
        DATAFRAME = DATAFRAME.iloc[:,:-1].copy()
        
        # 2. scale rest of dataframe
        DATAFRAME = pd.DataFrame(_scaler.fit_transform(DATAFRAME))
        
        # 3. add unscaled outcome back to scaled dataframe && and column names
        DATAFRAME = DATAFRAME.merge(y_complete, left_index=True, right_index=True)
        DATAFRAME.columns = column_names


DATAFRAME_SUMMARY = DATAFRAME.describe()




##########################################################################################################################
"""
    # visiualize true underlying data of Dataframe 
"""
##########################################################################################################################


if _visiualize_data:
    
    # Plotting combined distribution using histograms
    _hist = DATAFRAME.hist(column=column_names, bins=10, figsize=(20, 12), density=False, sharey=False, sharex=True)
    plt.title('Input without missing data')
    plt.tight_layout()
    plt.show()

    
    """
    # Visualizing correlation between variables using a heatmap
    corr_matrix = DATAFRAME.iloc[:,:-1].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
    plt.tight_layout()
    plt.show()
    """

    # is this relevant ?
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=np.array(DATAFRAME).flatten(), fill=True)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Original combined dataset KDE')
    plt.tight_layout()
    plt.show()
    
    


##########################################################################################################################
"""
    # choose frame mode and perform train - test - split
"""
##########################################################################################################################

    
X_complete = DATAFRAME.iloc[:, 0:-1]
y_complete = DATAFRAME[column_names[-1]]

if _unique_outcomes == 2:
    X_complete_train, X_complete_test, y_complete_train,  y_complete_test = train_test_split(X_complete, 
                                                                                             y_complete, 
                                                                                             test_size=0.25)

elif _unique_outcomes >= 3:
    
    # y labels have to be changed to categorical data
    y_complete_categorical = keras.utils.to_categorical(y_complete, num_classes=_unique_outcomes)
    
    X_complete_train, X_complete_test, y_complete_train, y_complete_test = train_test_split(X_complete, 
                                                                                            y_complete_categorical, 
                                                                                            test_size=0.25)




##########################################################################################################################
"""
    # create standard vanilla feed forward feural network
"""
##########################################################################################################################


if _train_model:
    
    # layers of the network
    _inputs = keras.Input(shape=(X_complete.shape[1]))
    _x = layers.Dense(32, activation='relu')(_inputs)
    _x = layers.Dense(16, activation='relu')(_x)
    
    

    if _unique_outcomes == 2:
        
        """
            --> Binary Model 
        """
        
        _outputs = layers.Dense(1, activation='sigmoid')(_x)
        
        # build model
        model = keras.Model(inputs=_inputs, outputs=_outputs)
        
        # compile model
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=["accuracy"])
        
        # fit model
        model_history = model.fit(X_complete_train, 
                                  y_complete_train, 
                                  validation_data=[X_complete_test, y_complete_test], 
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
        model = keras.Model(inputs=_inputs, outputs={"logits" : _logits,
                                                    "predictions" : tf.nn.softmax(_logits)})
        
        
        # compile model
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss={"logits": lambda y_true, y_pred: 0.0,
                            "predictions": keras.losses.CategoricalCrossentropy()},
                      metrics=["accuracy"])
        
        
        
        # fit model        
        model_history = model.fit(X_complete_train, 
                                  {'logits': y_complete_train, 
                                   'predictions': y_complete_train}, 
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
    y_complete_hat = model.predict(X_complete).flatten()
    y_complete_hat_labels = (y_complete_hat>0.5).astype("int32")
    y_complete_hat_label_frequency = pd.Series(y_complete_hat_labels).value_counts()
    
    
    if _visualize_original_predictions:
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data={"sigmoid" : y_complete_hat, "label" : y_complete_hat_labels}, 
                     x="sigmoid", 
                     hue="label", 
                     bins=10, 
                     stat="density", 
                     kde=False,
                     kde_kws={"cut":0})
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Frequency')
        plt.title('True Combined Output Hist Plot')
        plt.tight_layout()
        plt.show()
        
        
        """ --> redundant
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data={"sigmoid" : y_complete_hat, "label" : y_complete_hat_labels}, 
                    x="sigmoid", 
                    hue="label", 
                    common_grid=True, 
                    cut=0)
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Density')
        plt.title('True Combined Output Density Plot')
        plt.tight_layout()
        plt.show()
        """
    
    
    
elif _unique_outcomes >= 3:
    
    y_complete_hat = model.predict(X_complete)
    y_complete_hat_labels = np.argmax(y_complete_hat["predictions"], axis=1)
    y_complete_hat_label_frequency = pd.Series(y_complete_hat_labels).value_counts()

    if _visualize_original_predictions:
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=y_complete_hat_labels, 
                     bins=10, 
                     stat="count")
        plt.xlabel('Softmax Activations')
        plt.ylabel('Frequency')
        plt.title('True Combined Output Hist Plot')
        plt.tight_layout()
        plt.show()




if _get_original_prediction_metrics:
    
    if _unique_outcomes == 2:
        
        utils.create_metrics(y_complete, y_complete_hat_labels)
        plt.show()
    """
    elif _unique_outcomes >= 3:
        utils.create_metrics(y_complete, y_complete_hat_labels)
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
    
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=_MISS_RATE, random_seed=_RANDOM_STATE) 
    DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)

    # save DATAFRAME_MISS to pickle.dat 
    DATAFRAME_MISS.to_pickle(os.path.join(_dataset_path, "miss_frames", _dataset, _dataset + "_miss_rate_" + str(_MISS_RATE) + ".dat"))
    
#TODO
sys.exit()

# get statistics of DATAFRAME_MISS
DATAFRAME_MISS_SUMMARY = DATAFRAME_MISS.describe()




if _visiualize_data:
    
    # Plotting combined distribution using histograms
    DATAFRAME_MISS.hist(column=column_names, 
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
    sns.kdeplot(data={"DATAFRAME_KDE" : np.array(DATAFRAME).flatten(), 
                      "DATAFRAME_MISS_KDE" : np.array(DATAFRAME_MISS).flatten()}, 
                fill=False, 
                common_grid=True)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Original & Uncertain Dataset KDE comparison')
    plt.tight_layout()
    plt.show()
    

    
    
##########################################################################################################################
# Count missing data
##########################################################################################################################


count_missing = DATAFRAME_MISS.isnull().sum().sum()
print("\nCount of missing values:", count_missing, "\n")



    
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
    
    
if _IMPUTE and _IMPUTE_METHOD == "SimpleImputer//mean":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.copy()
    
    _simp_imp = SimpleImputer(strategy="mean")
    DATAFRAME_IMPUTE = pd.DataFrame(_simp_imp.fit_transform(DATAFRAME_IMPUTE), columns=column_names)
    
    
elif _IMPUTE and _IMPUTE_METHOD == "SimpleImputer//median":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.copy()
    
    _simp_imp = SimpleImputer(strategy="median")
    DATAFRAME_IMPUTE = pd.DataFrame(_simp_imp.fit_transform(DATAFRAME_IMPUTE), columns=column_names)


elif _IMPUTE and _IMPUTE_METHOD == "SimpleImputer//most_frequent":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.copy()
    
    _simp_imp = SimpleImputer(strategy="most_frequent")
    DATAFRAME_IMPUTE = pd.DataFrame(_simp_imp.fit_transform(DATAFRAME_IMPUTE), columns=column_names)
    
    
elif _IMPUTE and _IMPUTE_METHOD == "KNNImputer":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.iloc[:,:-1].copy()
    
    _knn_imp = KNNImputer(n_neighbors=5)
    DATAFRAME_IMPUTE = pd.DataFrame(_knn_imp.fit_transform(DATAFRAME_IMPUTE), columns=column_names)
    
    
    
if _SIMULATE:
    DATAFRAME_SIMULATE = DATAFRAME_MISS.copy()
    _SIMULATE_METHOD = "KDE_Simulation"


# exit if statement if no further simulations will be made
if _IMPUTE == False and _SIMULATE == False:
    sys.exit()




##########################################################################################################################
# experiments modul 1 - with imputation --> full data --> get_predictions
##########################################################################################################################

if _IMPUTE:
    
    print("\nPredictions for uncertain Dataset with uncertainties and imputed values:")
    
    X_impute = DATAFRAME_IMPUTE.iloc[:, 0:-1]
    
    if _unique_outcomes == 2:
        
        y_impute_hat = model.predict(X_impute).flatten()
        y_impute_hat_labels = (y_impute_hat>0.5).astype("int32")
        y_impute_hat_label_frequency = pd.Series(y_impute_hat_labels).value_counts()
        
        if _visualize_imputed_predictions:
            
            # visualize predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(data={"sigmoid" : y_impute_hat, "label" : y_impute_hat_labels}, 
                         x="sigmoid", hue="label", 
                         bins=10, 
                         stat="density", 
                         kde=False, 
                         kde_kws={"cut":0})
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Frequency')
            plt.title(f'Uncertain (deter.) Combined Output Hist Plot - Miss-Rate: {_MISS_RATE} - Impute-Method: {_IMPUTE_METHOD}')
            plt.tight_layout()
            plt.show()
            
            
            # visualize predictions
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data={"sigmoid" : y_impute_hat, "label" : y_impute_hat_labels}, 
                        x="sigmoid", 
                        hue="label", 
                        common_grid=True, 
                        cut=0)
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Density')
            plt.title(f'Uncertain Combined Output Density Plot - Miss-Rate: {_MISS_RATE} - Impute-Method: {_IMPUTE_METHOD}')
            plt.tight_layout()
            plt.show()

            """
            # compare imputation method against true distribution
            y_compare_joint = pd.concat([y_complete_joint, y_impute_joint], axis=1, ignore_index=True, sort=False)
            y_compare_joint.columns = ["True_Sigmoid", "True_Label", "Imputed_Sigmoid", "Imputed_Label"]
            y_compare_sigs = pd.DataFrame(data=[y_compare_joint["True_Sigmoid"], y_compare_joint["Imputed_Sigmoid"]]).transpose()

            
            plt.figure(figsize=(10, 6))
            #sns.kdeplot(data=y_compare_sigs, common_grid=True, cut=0)
            sns.histplot(data={"True_Sigmoid" : y_complete_hat["predictions"]}, bins=15)
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Density')
            plt.title(f'True/Uncertain(deter.) Sigmoid Comparison Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {IMPUTE_METHOD}')
            plt.tight_layout()
            plt.show()
            """
        
        
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
            
            utils.create_metrics(y_complete, y_impute_hat_labels)
            plt.show()
        """   
        elif _unique_outcomes >= 3:
            
            utils.create_metrics(y_complete, y_impute_hat_labels)
            plt.show()
        """




##########################################################################################################################
# Full Comparison # TODO
##########################################################################################################################

# comparisons over attributes without outcome values

"""
    These are mean values over the mean of all attributes of the datasets
"""

DATAFRAME_COMPARISON_SUMMARY = {"DATAFRAME mean count." : DATAFRAME_SUMMARY.loc["count"].mean(),
                                "DATAFRAME mean" : DATAFRAME_SUMMARY.loc["mean"].mean(), 
                                "DATAFRAME std" : DATAFRAME_SUMMARY.loc["std"].mean(),
                                "DATAFRAME_MISS mean count." : DATAFRAME_MISS_SUMMARY.loc["count"].mean(),
                                "DATAFRAME_MISS mean" : DATAFRAME_MISS_SUMMARY.loc["mean"].mean(),
                                "DATAFRAME_MISS std" : DATAFRAME_MISS_SUMMARY.loc["std"].mean()}


##########################################################################################################################
# experiments module 2 -- col wise simulations ----------> get kde values of dataframe
##########################################################################################################################


if _SIMULATE:
    
    """
        KDE COLLECTION -- ORIGINAL 
        --> is equal to the true distribution of the underlying data of the specific dataset
        --> to be able to get the true distribution we will use the original dataset with full certain data and no missing values
    """


    kde_collection_original = []
    
    for _column in column_names:
        values = DATAFRAME[_column].values
        
        kde = stats.gaussian_kde(values)   
        kde_collection_original.append(kde)
        
        
    
    
    """
        KDE COLLECTION -- UNCERTAIN 
        --> is equal to the uncertain distribution of the underlying data of the specific dataset with missing values
        --> for the uncertain distribution we will use the dataset including missing data (=uncertain data) 
        --> for computing, all of the missing data has to be dropped first, to retrieve the uncertain distribution of the rest
    """
    
    
    kde_collection_uncertain = []
    
    for _column in column_names:
        values = DATAFRAME_SIMULATE[_column].dropna().values
        
        kde = stats.gaussian_kde(values)   
        kde_collection_uncertain.append(kde)
        
        


    """
        Comperative Visualization of Certain (True) and Uncertain Column Distribution
        --> good for analyzing the differences between the two distribtions
    """    
    
    
    if _compare_col_kde_distributions: 
    
        
        if _compare_col_kde_mode == "single" or _compare_col_kde_mode == "both":
            
            for _column in column_names:
                
                # KDE Plot of column without missing data
                plt.figure(figsize=(8, 4))
                sns.kdeplot(data={"Certain Distribution // DATAFRAME":DATAFRAME[_column], 
                                  "Uncertain Distribution // DATAFRAME_SIMULATE":DATAFRAME_SIMULATE[_column]}, common_grid=True)
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
                    
                    if _column_count >= len(column_names) or column_names[_column_count] == column_names[-1]:
                        continue
                    
                    _ax = plt.subplot2grid((_hist.shape[0],_hist.shape[1]), (_i,_j))
                    sns.kdeplot(data={"Certain Distribution // DATAFRAME":DATAFRAME[column_names[_column_count]], 
                                      "Uncertain Distribution // DATAFRAME_SIMULATE":DATAFRAME_SIMULATE[column_names[_column_count]]}, 
                                common_grid=True, 
                                legend = False)
                    plt.title(column_names[_column_count])
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
    kde_collection_original = {column_names[i]: kde_collection_original[i] for i in range(len(column_names))}
        
    # to convert lists to dictionary
    kde_collection_uncertain = {column_names[i]: kde_collection_uncertain[i] for i in range(len(column_names))}
    

    
    
"""
"""         # test for the equality of kde samples from attributes with different missing rates of the different kde's
"""    
    
   ##-----------------------------------> just some tests
    
first_attr_orig = kde_collection_original[0]

first_attr_uncert = kde_collection_uncertain[0]
    
    
first_attr_orig_value = first_attr_orig.pdf(0.3)
    
first_attr_uncert_value = first_attr_uncert.pdf(0.3)   
    

print("Original", first_attr_orig_value)
print("Uncertain", first_attr_uncert_value)
    
    
first_attr_orig_sample = first_attr_orig.resample(100000)
    
first_attr_uncert_sample = first_attr_uncert.resample(100000)
    

first_attr_orig_stats = stats.describe(first_attr_orig_sample.transpose())
print(first_attr_orig_stats)

first_attr_uncert_stats = stats.describe(first_attr_uncert_sample.transpose())
print(first_attr_uncert_stats)

sns.histplot(first_attr_orig_sample.transpose())
plt.show()
sns.histplot(first_attr_uncert_sample.transpose())
plt.show()

"""  
"""
"""


#TODO
##########################################################################################################################
# experiments modul 2 - with simulation --> missing data (row wise) --> useage of kde of columns to simulate outcome
##########################################################################################################################

# impute == false is equal to stochastic simulation approach
if _SIMULATE == True:
    
    print("\nPredictions for uncertain Dataset with uncertainties and simulated values:")
    
    # step 0 --> first for loop for getting row and simulating this specific row
    
    uncertain_simulation_history_mean = []
    uncertain_simulation_history_mean_labels = []    
    
    
    original_simulation_history_mean = []
    original_simulation_history_mean_labels = []  
    
    
    """
        in the next step (for-loop) the main simulation part is carried out
            - the for loop will itterate through all rows inside a given dataset
            - in each cycle two main simulation predictions will be creaded
                - one given a true distribution (kde)
                - and one given a uncertain distribution (kde)
            
    """
    
    
    if _SIMULATION_RANGE == None:
        _SIMULATION_RANGE = range(len(DATAFRAME_SIMULATE))
    
    
    for i in tqdm(_SIMULATION_RANGE):
        
        """
            # step 1: get current row to perform simulation with
        """
        DATAFRAME_SIMULATE_ROW = pd.DataFrame(DATAFRAME_SIMULATE.loc[i])
        
        
        """
            # step 2: find all the attributes with nan values and save to variable as keys for the kde-dictionary
        """
        
        inds_to_key = np.where(DATAFRAME_SIMULATE_ROW.isna().all(axis=1))[0]
        inds_to_key = [column_names[i] for i in inds_to_key]
    
        """
            # step 3: sample a value from the specific kde of the missing value - aka. beginning of MonteCarlo Simulation
            # --> and safe sampled values for this row in a history
        """
        
        uncertain_sample_collection = []
        original_sample_collection = [] 
        
        for key in inds_to_key:
            
            _uncertain_sample = kde_collection_uncertain[key].resample(_SIMULATION_LENGTH)
            _original_sample = kde_collection_original[key].resample(_SIMULATION_LENGTH)
          
            uncertain_sample_collection.append(_uncertain_sample.flatten())
            original_sample_collection.append(_original_sample.flatten())
        
    
        uncertain_sample_collection = pd.DataFrame(uncertain_sample_collection).transpose()
        uncertain_sample_collection.columns = inds_to_key
    
        original_sample_collection = pd.DataFrame(original_sample_collection).transpose()
        original_sample_collection.columns = inds_to_key
    
    
        """
            # step 4: create DATAFRAME for faster simulation (basis input) and replace missing values with sampled ones   
            # index length of DATAFRAME_MISS_ROW is now equal to number of simulations
        """
        
        DATAFRAME_MC_SIMULATION = DATAFRAME_SIMULATE_ROW.copy().transpose()
        DATAFRAME_MC_SIMULATION = pd.concat([DATAFRAME_MC_SIMULATION] * _SIMULATION_LENGTH, ignore_index=True)
        
        
        # basis dataframe used for uncertain kde simulation
        DATAFRAME_MC_UNCERTAIN_KDE_SIMULATION = DATAFRAME_MC_SIMULATION.copy()
        
        
        # basis dataframe used for original (true) kde simulation
        DATAFRAME_MC_ORIGINAL_KDE_SIMULATION = DATAFRAME_MC_SIMULATION.copy()
        
        
        # replace the missing values of DATAFRAME_MISS_ROW/ (now MC_SIMULATION) with the created samples 
        for _col in inds_to_key:
            
            DATAFRAME_MC_UNCERTAIN_KDE_SIMULATION[_col] = uncertain_sample_collection[_col]
            DATAFRAME_MC_ORIGINAL_KDE_SIMULATION[_col] = original_sample_collection[_col]
        
        
        """
            step 5: main predictions on collected samples/data
        """
        
        X_uncertain_simulation = DATAFRAME_MC_UNCERTAIN_KDE_SIMULATION.iloc[:, 0:-1]
        
        X_original_simulation = DATAFRAME_MC_ORIGINAL_KDE_SIMULATION.iloc[:, 0:-1]
            
        
        
        
        if _unique_outcomes == 2:
            
            
            """
            #step 5.1.a: row-wise predictions on uncertain samples
                -----> Simulation procedure for uncertain kde induced simulation frames
            """

            # predictions and labels
            y_uncertain_simulation_hat = model.predict(X_uncertain_simulation, verbose=0).flatten()
            y_uncertain_simulation_hat_labels = (y_uncertain_simulation_hat>0.5).astype("int32")
            
            # simulation outcome and statistics
            y_uncertain_simulation_hat_mean = y_uncertain_simulation_hat.mean()
            y_uncertain_simulation_hat_std = y_uncertain_simulation_hat.std()
            
            
            y_uncertain_simulation_joint = np.stack([y_uncertain_simulation_hat, y_uncertain_simulation_hat_labels], 1)
            y_uncertain_simulation_joint = pd.DataFrame(y_uncertain_simulation_joint, columns=["sigmoid", "label"])
        
            # simulation history appendix
            uncertain_simulation_history_mean.append(y_uncertain_simulation_hat_mean)
            uncertain_simulation_history_mean_labels.append((y_uncertain_simulation_hat_mean>0.5).astype("int32"))
            
            
            """
                #step 5.1.b: row-wise predictions on original samples
                -----> Simulation procedure for true original kde induced simulation frames
            """
            
            # predictions and labels
            y_original_simulation_hat = model.predict(X_original_simulation, verbose=0).flatten()
            y_original_simulation_hat_labels = (y_original_simulation_hat>0.5).astype("int32")
            
            # simulation outcome and statistics
            y_original_simulation_hat_mean = y_original_simulation_hat.mean()
            y_original_simulation_hat_std = y_original_simulation_hat.std()
            
            y_original_simulation_joint = np.stack([y_original_simulation_hat, y_original_simulation_hat_labels], 1)
            y_original_simulation_joint = pd.DataFrame(y_original_simulation_joint, columns=["sigmoid", "label"])
        
            # simulation history appendix
            original_simulation_history_mean.append(y_original_simulation_hat_mean)
            original_simulation_history_mean_labels.append((y_original_simulation_hat_mean>0.5).astype("int32"))
    


            """
                combined statistics/metrics of the kde curve found inside of the simulated predictions
            """
            
            #x-axis ranges from 0 and 1 with .001 steps
            x_axis = np.arange(0.0, 1.0, 0.001)
            
            #### uncertain kde curve
            uncertain_kde_pdfs = stats.gaussian_kde(y_uncertain_simulation_hat).pdf(x_axis) 

            uncertain_kde_density_peak_indices = scipy.signal.find_peaks(uncertain_kde_pdfs)
            uncertain_kde_density_peak_pdf = [uncertain_kde_pdfs[i] for i in uncertain_kde_density_peak_indices[0]]
            
            uncertain_kde_stats = {int(uncertain_kde_density_peak_indices[0][i]) : uncertain_kde_density_peak_pdf[i] for i in range(len(uncertain_kde_density_peak_indices[0]))}
            
            
            #### origninal kde curve
            original_kde_pdfs = stats.gaussian_kde(y_original_simulation_hat).pdf(x_axis) 
            
            original_kde_density_peak_indices = scipy.signal.find_peaks(original_kde_pdfs)
            original_kde_density_peak_pdf = [original_kde_pdfs[i] for i in original_kde_density_peak_indices[0]]
            
            origninal_kde_stats = {int(original_kde_density_peak_indices[0][i]) : original_kde_density_peak_pdf[i] for i in range(len(original_kde_density_peak_indices[0]))}
            

            # visualizations for binary simulation // comparison plots
            if _simulation_visualizations:
    
                """
                    Plot_5.1.a: Histogam which shows the simulated row sigmoid results with hue 
                """
                # visualize predictions with hist plots
                plt.figure(figsize=(10, 6))
                sns.histplot(data=y_uncertain_simulation_joint, x="sigmoid", hue="label", hue_order=[0, 1], bins=15, binrange=(0, 1), stat="count", kde=False, kde_kws={"cut":0})
                
                plt.axvline(x=y_complete[i], linewidth=4, linestyle = "-", color = "green", label="Original Label")
                plt.axvline(x=y_complete_hat_labels[i], linewidth=4, linestyle = "-", color = "red", label="Predicted Model Label")
                
                if _IMPUTE:
                    plt.axvline(x=y_impute_hat[i], linewidth=2, linestyle = "--", color = "purple", label="Imputated Prediction") # impute prediction
                
                plt.axvline(x=y_original_simulation_hat_mean, linewidth=2, linestyle = "-.", color = "black", label="Orig. Mean Sim. Value") # orig. simulation prediction mean
                plt.axvline(x=y_uncertain_simulation_hat_mean, linewidth=2, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # uncert. simulation prediction mean
                
                plt.title(f'Row: {i} Uncertain KDE Sim. Output Hist Plot - Miss-Rate: {_MISS_RATE}')
                plt.legend(["Original Label", "Predicted Model Label", "Imputated Prediction", "Orig. Mean Sim. Value", "Uncert. Mean Sim. Value", "Label 1", "Label 0"])
                plt.xlabel('Sigmoid Activations')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.show()
        
        
                """
                    Plot_5.1.b: Histogam which shows the simulated row sigmoid results with hue 
                """
                # visualize predictions with hist plots
                plt.figure(figsize=(10, 6))
                sns.histplot(data=y_original_simulation_joint, x="sigmoid", hue="label", hue_order=[0, 1], bins=15, binrange=(0, 1), stat="count", kde=False, kde_kws={"cut":0})
                
                plt.axvline(x=y_complete[i], linewidth=4, linestyle = "-", color = "green", label="Original Label")
                plt.axvline(x=y_complete_hat_labels[i], linewidth=4, linestyle = "-", color = "red", label="Predicted Model Label")
                
                if _IMPUTE:
                    plt.axvline(x=y_impute_hat[i], linewidth=2, linestyle = "--", color = "purple", label="Imputated Prediction") # impute prediction
                
                plt.axvline(x=y_original_simulation_hat_mean, linewidth=2, linestyle = "-.", color = "black", label="Orig. Mean Sim. Value") # orig. simulation prediction mean
                plt.axvline(x=y_uncertain_simulation_hat_mean, linewidth=2, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # uncert. simulation prediction mean
                
                plt.title(f'Row: {i} Original KDE Sim. Output Hist Plot - Miss-Rate: {_MISS_RATE}')
                plt.legend(["Original Label", "Predicted Model Label", "Imputated Prediction", "Orig. Mean Sim. Value", "Uncert. Mean Sim. Value", "Label 1", "Label 0"])
                plt.xlabel('Sigmoid Activations')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.show()
                
                
                """
                    Plot_combined_output: KDE PLOT of Uncerlying uncertainty
                """

                #plot normal distribution with mean and std of simulated values
                plt.plot(x_axis, stats.norm.pdf(x_axis, y_original_simulation_hat_mean, y_original_simulation_hat_std), label="Orig. Sim. Distribution", color="black", linestyle = "-")
                plt.axvline(x=y_original_simulation_hat_mean, linewidth=1, linestyle = "-.", color = "black", label="Orig. Mean Sim. Value") # mean original kde prediction
                
                plt.plot(x_axis, uncertain_kde_pdfs, label="Uncertain. Sim. Distribution // KDE", color="pink", linestyle = "--")
                plt.plot(x_axis, original_kde_pdfs, label="Original. Sim. Distribution // KDE", color="green", linestyle = "--")
                
                plt.plot(x_axis, stats.norm.pdf(x_axis, y_uncertain_simulation_hat_mean, y_uncertain_simulation_hat_std), label="Uncert. Sim. Distribution", color="grey", linestyle = "--")
                plt.axvline(x=y_uncertain_simulation_hat_mean, linewidth=1, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # mean uncertain kde prediction
                
                plt.axvline(x=y_complete[i], linewidth=3, linestyle = "-", color = "green", label="Original Label")
                plt.axvline(x=y_complete_hat_labels[i], linewidth=2, linestyle = "-", color = "red", label="Predicted Model Label")
                
                if _IMPUTE:
                    plt.axvline(x=y_impute_hat[i], linewidth=1, linestyle = "--", color = "purple", label="Imputated Prediction") # impute prediction
                
                plt.title(f'Row: {i} Underlying Uncertainty of the Simulation - Miss-Rate: {_MISS_RATE} - Impute-Method: {_SIMULATE_METHOD}')
                plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
                #plt.legend()
                plt.xlabel('Sigmoid Activations')
                plt.ylabel('Density of Sigmoid Activations')
                #plt.tight_layout()
                plt.show()
    
    
    
    
    
        if _unique_outcomes >= 3:
            
            
            """
            #step 5.2.a: row-wise predictions on uncertain samples
                -----> Simulation procedure for uncertain kde induced simulation frames
            """
            
            # predictions and labels
            y_uncertain_simulation_hat = model.predict(X_uncertain_simulation, verbose=0)
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
            y_original_simulation_hat = model.predict(X_original_simulation, verbose=0)
            y_original_simulation_hat_labels = np.argmax(y_original_simulation_hat["predictions"], axis=1)    
            
            # simulation outcome and statistics
            y_original_simulation_hat_softmax_mean = y_original_simulation_hat["predictions"].mean(axis=0)
            y_original_simulation_hat_logits_mean = y_original_simulation_hat["logits"].mean(axis=0)
            
            y_original_simulation_hat_softmax_std = y_original_simulation_hat["predictions"].std(axis=0)
            y_original_simulation_hat_logits_std = y_original_simulation_hat["logits"].std(axis=0)
            
            y_original_simulation_hat_label_frequency = pd.Series(y_original_simulation_hat_labels).value_counts()
    



            # single
            sns.kdeplot(y_original_simulation_hat["logits"])  
            plt.title(f"Original Row - {i} // Logit Plot")
            plt.show()
            
            sns.kdeplot(y_original_simulation_hat["predictions"])  
            plt.title(f"Original Row - {i} // Prediction Softmax Plot")
            plt.show()
            """
            # combined
            sns.kdeplot(y_original_simulation_hat["logits"].flatten())  
            plt.title(f"Original Row - {i} // Logit Plot")
            plt.show()
            
            sns.kdeplot(y_original_simulation_hat["predictions"].flatten())  
            plt.title(f"Original Row - {i} // Prediction Softmax Plot")
            plt.show()
            """
            

            

    
    
    
    
    
    
    
    
    """
            ----------------> simulations process end ---> further analysis below
    """
    
    """
        Below: Comparisons between the prediction results of Uncertain and Certain KDE simulations
    """
    
    if _unique_outcomes == 2:       # TODO try to stay away from mean prediction metric
        
        y_uncertain_simulation_history_joint = np.stack([uncertain_simulation_history_mean, uncertain_simulation_history_mean_labels], 1)
        y_uncertain_simulation_history_joint = pd.DataFrame(y_uncertain_simulation_history_joint, columns=["sigmoid", "label"])
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=y_uncertain_simulation_history_joint, x="sigmoid", hue="label", hue_order=[0, 1], bins=15, binrange=(0, 1), stat="count", kde=False, kde_kws={"cut":0})
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Frequency')
        plt.title(f'Uncertain (Unc. Stoch.) Combined Output Hist Plot - Miss-Rate: {_MISS_RATE} - Impute-Method: {_SIMULATE_METHOD}')
        plt.tight_layout()
        plt.show()
        
        
        y_original_simulation_history_joint = np.stack([original_simulation_history_mean, original_simulation_history_mean_labels], 1)
        y_original_simulation_history_joint = pd.DataFrame(y_original_simulation_history_joint, columns=["sigmoid", "label"])
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=y_original_simulation_history_joint, x="sigmoid", hue="label", hue_order=[0, 1], bins=15, binrange=(0, 1), stat="count", kde=False, kde_kws={"cut":0})
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Frequency')
        plt.title(f'Uncertain (Ori. Stoch.) Combined Output Hist Plot - Miss-Rate: {_MISS_RATE} - Impute-Method: {_SIMULATE_METHOD}')
        plt.tight_layout()
        plt.show()
        
        """
        if _get_simulated_prediction_metrics:
            
            utils.create_metrics(y_complete, uncertain_simulation_history_mean_labels)
            plt.show()
        """            
    











"""
    ---> Comparison of everything
    
    Explanation of DATAFRAME_COMBINED_RESULTS:
        - Original Label is equal to the Label which is found originally in the dataset
        - 0: is the shortcut for a prediction with a trained model on full data without uncertainties
            -> only uncertainties found here are model uncertainties 
        - 1: is the shortcut for predictions with imputed values
        
        - 2: simulation results - metric mean 
        
"""
if _IMPUTE == True and _SIMULATE == True:
    
    if len(_SIMULATION_RANGE) == len(DATAFRAME_SIMULATE):
        
        DATAFRAME_COMBINED_RESULTS = np.stack([y_complete, 
                                               y_complete_hat, 
                                               y_complete_hat_labels, 
                                               (y_complete == y_complete_hat_labels),
                                               y_impute_hat,
                                               y_impute_hat_labels,
                                               (y_complete_hat_labels == y_impute_hat_labels),
                                               uncertain_simulation_history_mean,
                                               uncertain_simulation_history_mean_labels,
                                               (y_complete_hat_labels == uncertain_simulation_history_mean_labels),
                                               original_simulation_history_mean,
                                               original_simulation_history_mean_labels,
                                               (y_complete_hat_labels == original_simulation_history_mean_labels)], 1)
        
        DATAFRAME_COMBINED_RESULTS = pd.DataFrame(data=DATAFRAME_COMBINED_RESULTS, columns=["Original_Label", 
                                                                                            "0_Prediction", 
                                                                                            "0_Predicted_Label", 
                                                                                            "0_Prediction_Result",
                                                                                            "1_Imputation",
                                                                                            "1_Imputation_Label",
                                                                                            "1_Results_vs_Prediction_Label",
                                                                                            "2_U_Simulation_Mean",
                                                                                            "2_U_Simulation_Label",
                                                                                            "2_U_Simulation_vs_Prediction_Label",
                                                                                            "3_O_Simulation_Mean",
                                                                                            "3_O_Simulation_Label",
                                                                                            "3_O_Simulation_vs_Prediction_Label"])
        
        DATAFRAME_COMBINED_RESULTS["0_Prediction_Result"] = DATAFRAME_COMBINED_RESULTS["0_Prediction_Result"].astype(bool)
        DATAFRAME_COMBINED_RESULTS["1_Results_vs_Prediction_Label"] = DATAFRAME_COMBINED_RESULTS["1_Results_vs_Prediction_Label"].astype(bool)
        DATAFRAME_COMBINED_RESULTS["2_U_Simulation_vs_Prediction_Label"] = DATAFRAME_COMBINED_RESULTS["2_U_Simulation_vs_Prediction_Label"].astype(bool)
        DATAFRAME_COMBINED_RESULTS["3_O_Simulation_vs_Prediction_Label"] = DATAFRAME_COMBINED_RESULTS["3_O_Simulation_vs_Prediction_Label"].astype(bool)
    
    
    
        DATAFRAME_COMBINED_ANALYSIS = pd.Series(data={"Correct label assigned by model": DATAFRAME_COMBINED_RESULTS["0_Prediction_Result"].value_counts(True)[0],
                                                      "Correct label assigned by imputation": DATAFRAME_COMBINED_RESULTS["1_Results_vs_Prediction_Label"].value_counts(True)[0],
                                                      "Correct label assigned by simulation_unc_kde": DATAFRAME_COMBINED_RESULTS["2_U_Simulation_vs_Prediction_Label"].value_counts(True)[0],
                                                      "Correct label assigned by imputation_ori_kde": DATAFRAME_COMBINED_RESULTS["3_O_Simulation_vs_Prediction_Label"].value_counts(True)[0]})
    



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




