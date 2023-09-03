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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#import seaborn.objects as so

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
import scipy.stats as stats
#from sklearn.neighbors import KernelDensity


#import chaospy as cp


##########################################################################################################################
# set important paths
##########################################################################################################################


# set path to different folders
dataset_path = os.path.join(os.getcwd(), 'datasets')
#image_path = os.path.join(os.getcwd(), 'images')
model_path = os.path.join(os.getcwd(), 'models')




##########################################################################################################################
"""
information about the datasets:
    -wdbc - all attributes are considered continious - outcome is binary 
    -climate_simulation - 
    -australian - 
    - predict+students+dropout+and+academic+success - three outcomes
    
following all the different settings for this simulation run can be found
    -dataset = "choose dataset"
    -standardize_dataset = "used for standardizing the dataset -- values between 0 and 1 -- minmax"
"""
##########################################################################################################################

#choose working dataset: "australian" or "climate_simulation", "wdbc" -> Breast Cancer Wisconsin (Diagnostic)
dataset = "predict+students+dropout+and+academic+success" 

"""
# set random state
RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)
np.random.RandomState(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
"""

# further dataset settings
standardize_data = True


# settings for visualization
visiualize_data = False
visualize_original_predictions = True
visualize_imputed_predictions = True


# train or load model
train_model = False
load_model = True

# prediction metrics
get_original_prediction_metrics = False
get_imputed_prediction_metrics = False
get_simulated_prediction_metrics = False


# DATAFRAME_MISS settings - Introduction to missing values in the choosen Dataframe
# load DataFrame_MISS // if True, an already created one will be loaded, else a new one will be created
load_dataframe_miss = True

MISS_RATE=0.1


#KDE_VALUES OF EACH COLUMN - affected frames are DATAFRAME_SIMULATE -> Uncertain and DATAFRAME -> Certain/True
compare_col_kde_distributions = False


# modes for deterministic/stochastic experiments on missing dataframes
# choose between kde_imputer, SimpleImputer//mean, SimpleImputer//median, SimpleImputer//most_frequent, KNNImputer
IMPUTE = True
IMPUTE_METHOD = "SimpleImputer//mean"

SIMULATE = False
SIMULATION_LENGTH = 100
#SIMULATION_RANGE = None
SIMULATION_RANGE = range(0, 10, 1)
simulation_visualizations = True




##########################################################################################################################
# load original datasets with full data
##########################################################################################################################

    
# load data for climate modal simulation crashes dataset
if dataset == "wdbc":
    
    with open(os.path.join(dataset_path, dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep=",", engine='python', header = None)
    
    # drop the first 
    y_complete = DATAFRAME.iloc[:,1].copy()
    DATAFRAME = DATAFRAME.iloc[:, 2:].copy()
    
    DATAFRAME = DATAFRAME.merge(y_complete, left_index=True, right_index=True)
    
    DATAFRAME.iloc[:,-1].replace(['B', 'M'], [0, 1], inplace=True)
    
    column_names = ["Attribute: " + str(i) for i in range(len(DATAFRAME.columns))]
    column_names[-1] = "Outcome"
    DATAFRAME.columns = column_names
    
    
    
    
# load data for climate modal simulation crashes dataset
if dataset == "climate_simulation":
    
    with open(os.path.join(dataset_path, dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep="\s+", engine='python', header = 0)
    
    # drop the first two elements of the dataset -> not relevant
    DATAFRAME = DATAFRAME.iloc[:, 2:]

    column_names = DATAFRAME.columns.to_list()




# load data for australian credit card approval dataset
if dataset == "australian":
    
    with open(os.path.join(dataset_path, dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep=" ", engine="python", header=None)
    
    # rename columns   
    column_names=["Sex", "Age", "Mean time at adresses", "Home status", "Current occupation",
              "Current job status", "Mean time with employers", "Other investments",
              "Bank account", "Time with bank", "Liability reference", "Account reference",
              "Monthly housing expense", "Savings account balance", "0 - Reject / 1 - Accept"]
    
    
    original_col_names = ["0", "1", "2", "3", "4", "5" , "6", "7", "8", "9", "10", "11", "12", "13", "14"]
    
    # rename columns
    DATAFRAME.columns = column_names
    
    
    

# load data for predict+students+dropout+and+academic+success dataset
if dataset == "predict+students+dropout+and+academic+success":
    
    with open(os.path.join(dataset_path, dataset + ".csv"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_csv(DATAFRAME, sep=";", engine="python")

    # change target names to numerical value
    DATAFRAME.iloc[:,-1].replace(['Dropout', 'Enrolled', "Graduate"], [0, 1, 2], inplace=True)

    # rename columns   
    column_names_original = DATAFRAME.columns
    
    
    column_names = ["Attribute: " + str(i) for i in range(len(DATAFRAME.columns))]
    column_names[-1] = "Outcome"
    DATAFRAME.columns = column_names
    
    


unique_outcomes = len(DATAFRAME.Outcome.unique())



##########################################################################################################################
# standardization of values for better performance
##########################################################################################################################
    

if standardize_data:
    # use data scaler to norm the data
    scaler = MinMaxScaler()
    
    if unique_outcomes == 2:
        
        # change to dataframe
        DATAFRAME = pd.DataFrame(scaler.fit_transform(DATAFRAME))
        DATAFRAME.columns = column_names
        
    elif unique_outcomes >= 3:
        
        # change to dataframe
        # drop outcome --> scale rest of dataframe --> add unscaled outcome back to normal
        y_complete = DATAFRAME.iloc[:,-1].copy()
        DATAFRAME = DATAFRAME.iloc[:,:-1].copy()
        
        DATAFRAME = pd.DataFrame(scaler.fit_transform(DATAFRAME))
        
        DATAFRAME = DATAFRAME.merge(y_complete, left_index=True, right_index=True)
        
        DATAFRAME.columns = column_names


DATAFRAME_describe = DATAFRAME.describe()




##########################################################################################################################
# visiualize true underlying data of Dataframe 
##########################################################################################################################


if visiualize_data:
    
    # Plotting combined distribution using histograms
    DATAFRAME.hist(column=column_names, bins=15, figsize=(12, 10), density=False, sharey=False, sharex=True)
    #plt.xlabel('Sigmoid Activations')
    #plt.ylabel('Density')
    plt.title('Input without missing data')
    plt.tight_layout()
    plt.show()

    
    """
    # Visualizing correlation between variables using a heatmap
    corr_matrix = DATAFRAME.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
    plt.tight_layout()
    plt.show()
    """
    
    """ --> redundant
    # Create a KDE plot for each column
    for column in column_names[:-1]:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(data=DATAFRAME[column], fill=True, color='skyblue', alpha=0.5)
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.title(f'KDE Plot of {column}')
        plt.tight_layout()
        plt.show()
    """    
        
    # Create a combined KDE plot
    plt.figure(figsize=(12, 6))
    for column in column_names[:-1]:
        sns.kdeplot(data=DATAFRAME[column], fill=True, label=column)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Combined KDE Plot')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    


##########################################################################################################################
# choose frame mode and perform train - test - split
##########################################################################################################################

    
X_complete = DATAFRAME.iloc[:, 0:-1]
y_complete = DATAFRAME[column_names[-1]]

if unique_outcomes >= 3:
    y_complete_categorical = keras.utils.to_categorical(y_complete, num_classes=unique_outcomes)


if unique_outcomes == 2:
    X_complete_train, X_complete_test, y_complete_train,  y_complete_test = train_test_split(X_complete, y_complete, test_size=0.25)

elif unique_outcomes >= 3:
    X_complete_train, X_complete_test, y_complete_train,  y_complete_test = train_test_split(X_complete, y_complete_categorical, test_size=0.25)


##########################################################################################################################
# create standard vanilla feed forward feural network
##########################################################################################################################


if train_model:
    
    # layers of the network
    inputs = keras.Input(shape=(X_complete.shape[1]))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    
    # binary model
    if unique_outcomes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
    # multivariate model
    elif unique_outcomes >= 3: 
        outputs = layers.Dense(unique_outcomes, activation='softmax')(x)
        
        
    # build model
    model = keras.Model(inputs=inputs, outputs=outputs)


    # binary model
    if unique_outcomes == 2:
        # compile model
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=["accuracy"])
    # multivariate model
    elif unique_outcomes >= 3: 
        # compile model
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=["accuracy"])    
    
    # fit model
    model_history = model.fit(X_complete_train, y_complete_train, validation_data=[X_complete_test, y_complete_test], batch_size=15, epochs=50, verbose=0)
    
    
    # plot model
    utils.plot_history(model_history)


    # save new model
    model.save(os.path.join(model_path, dataset + "_model"))
    
    


##########################################################################################################################
# load model without training
##########################################################################################################################


if load_model:
    
    model = keras.models.load_model(os.path.join(model_path, dataset + "_model"))
    model.summary()




##########################################################################################################################
# singe prediction metrics with a perfectly trained model - no uncertainties -- deterministic as usual
##########################################################################################################################

print("\nPredictions for complete Dataset without uncertainties:")

if unique_outcomes == 2:
    
    y_complete_hat = model.predict(X_complete).flatten()
    y_complete_hat_labels = (y_complete_hat>0.5).astype("int32")
    y_complete_joint = np.stack([y_complete_hat, y_complete_hat_labels], 1)
    y_complete_joint = pd.DataFrame(y_complete_joint, columns=["sigmoid", "label"])
    
    if visualize_original_predictions:
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=y_complete_joint, x="sigmoid", hue="label", bins=10, stat="density", kde=True, kde_kws={"cut":0})
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Frequency')
        plt.title('True Combined Output Hist Plot')
        plt.tight_layout()
        plt.show()
        
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=y_complete_joint, x="sigmoid", hue="label", common_grid=True, cut=0)
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Density')
        plt.title('True Combined Output Density Plot')
        plt.tight_layout()
        plt.show()
    
    
    
    
elif unique_outcomes >= 3:
    
    y_complete_hat = model.predict(X_complete)
    y_complete_hat_labels = np.argmax(y_complete_hat, axis=1)
    #y_complete_joint = np.stack([y_complete_hat, y_complete_hat_labels], 1)
    #y_complete_joint = pd.DataFrame(y_complete_joint, columns=["softmax", "label"])


    if visualize_original_predictions:
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=y_complete_hat_labels, bins=10, stat="count")
        plt.xlabel('Softmax Activations')
        plt.ylabel('Frequency')
        plt.title('True Combined Output Hist Plot')
        plt.tight_layout()
        plt.show()




if get_original_prediction_metrics:
    
    if unique_outcomes == 2:
        utils.create_metrics(y_complete, y_complete_hat_labels)
        plt.show()




##########################################################################################################################
# introduce missing data - aka. aleatoric uncertainty
##########################################################################################################################

if load_dataframe_miss:
  
    DATAFRAME_MISS = pd.read_pickle(os.path.join(dataset_path, "miss_frames", dataset, dataset + "_miss_rate_" + str(MISS_RATE) + ".dat"))    
    
    
    if visiualize_data:
        
        # Plotting combined distribution using histograms
        DATAFRAME_MISS.hist(column=column_names, bins=15, figsize=(12, 10), density=True)
        #plt.xlabel('Sigmoid Activations')
        #plt.ylabel('Density')
        plt.title('Input with missing data')
        plt.tight_layout()
        plt.show()
    
else:
    

    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=MISS_RATE) 
    DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)

    
    # save DATAFRAME_MISS to pickle.dat for better comparison
    DATAFRAME_MISS.to_pickle(os.path.join(dataset_path, "miss_frames", dataset, dataset + "_miss_rate_" + str(MISS_RATE) + ".dat"))
    
        
        

    if visiualize_data:
        
        # Plotting combined distribution using histograms
        DATAFRAME_MISS.hist(column=column_names, bins=15, figsize=(12, 10), density=True)
        #plt.xlabel('Sigmoid Activations')
        #plt.ylabel('Density')
        plt.title('Input with missing data')
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
    
    
if IMPUTE and IMPUTE_METHOD == "SimpleImputer//mean":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.copy()
    
    simp_imp = SimpleImputer(strategy="mean")
    DATAFRAME_IMPUTE = pd.DataFrame(simp_imp.fit_transform(DATAFRAME_IMPUTE), columns=column_names)
    
    
elif IMPUTE and IMPUTE_METHOD == "SimpleImputer//median":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.copy()
    
    simp_imp = SimpleImputer(strategy="median")
    DATAFRAME_IMPUTE = pd.DataFrame(simp_imp.fit_transform(DATAFRAME_IMPUTE), columns=column_names)


elif IMPUTE and IMPUTE_METHOD == "SimpleImputer//most_frequent":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.copy()
    
    simp_imp = SimpleImputer(strategy="most_frequent")
    DATAFRAME_IMPUTE = pd.DataFrame(simp_imp.fit_transform(DATAFRAME_IMPUTE), columns=column_names)
    
    
elif IMPUTE and IMPUTE_METHOD == "KNNImputer":
    DATAFRAME_IMPUTE = DATAFRAME_MISS.iloc[:,:-1].copy()
    
    knn_imp = KNNImputer(n_neighbors=5)
    DATAFRAME_IMPUTE = pd.DataFrame(knn_imp.fit_transform(DATAFRAME_IMPUTE), columns=column_names)
    
    
    
if SIMULATE:
    DATAFRAME_SIMULATE = DATAFRAME_MISS.copy()
    SIMULATE_METHOD = "KDE_Simulation"



if IMPUTE == False and SIMULATE == False:
    sys.exit()





##########################################################################################################################
# experiments modul 1 - with imputation --> full data --> get_predictions
##########################################################################################################################

if IMPUTE:
    
    print("\nPredictions for uncertain Dataset with uncertainties and imputed values:")
    
    X_impute = DATAFRAME_IMPUTE.iloc[:, 0:-1]
    
    if unique_outcomes == 2:
        
        y_impute_hat = model.predict(X_impute).flatten()
        y_impute_hat_labels = (y_impute_hat>0.5).astype("int32")
        y_impute_joint = np.stack([y_impute_hat, y_impute_hat_labels], 1)
        y_impute_joint = pd.DataFrame(y_impute_joint, columns=["sigmoid", "label"])
        
        
        if visualize_imputed_predictions:
            # visualize predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(data=y_impute_joint, x="sigmoid", hue="label", bins=10, stat="density", kde=False, kde_kws={"cut":0})
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Frequency')
            plt.title(f'Uncertain (deter.) Combined Output Hist Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {IMPUTE_METHOD}')
            plt.tight_layout()
            plt.show()
        
        
        
    elif unique_outcomes >= 3:
        
        y_impute_hat = model.predict(X_impute)
        y_impute_hat_labels = np.argmax(y_impute_hat, axis=1)
        #y_impute_joint = np.stack([y_impute_hat, y_impute_hat_labels], 1)
        #y_impute_joint = pd.DataFrame(y_impute_joint, columns=["sigmoid", "label"])
        
        
        if visualize_imputed_predictions:
            # visualize predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(data=y_impute_hat_labels, bins=10, stat="count", kde=False, kde_kws={"cut":0})
            plt.xlabel('Softmax Activations')
            plt.ylabel('Frequency')
            plt.title(f'Uncertain (deter.) Combined Output Hist Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {IMPUTE_METHOD}')
            plt.tight_layout()
            plt.show()
    
    
    """
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=y_impute_joint, x="sigmoid", hue="label", common_grid=True, cut=0)
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Density')
    plt.title(f'Uncertain Combined Output Density Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {IMPUTE_METHOD}')
    plt.tight_layout()
    plt.show()
    """
    
    
    """
    # compare imputation method against true distribution
    
    y_compare_joint = pd.concat([y_complete_joint, y_impute_joint], axis=1, ignore_index=True, sort=False)
    y_compare_joint.columns = ["True_Sigmoid", "True_Label", "Imputed_Sigmoid", "Imputed_Label"]
    y_compare_sigs = pd.DataFrame(data=[y_compare_joint["True_Sigmoid"], y_compare_joint["Imputed_Sigmoid"]]).transpose()
    
    plt.figure(figsize=(10, 6))
    #sns.kdeplot(data=y_compare_sigs, common_grid=True, cut=0)
    sns.histplot(data=y_compare_sigs, bins=15)
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Density')
    plt.title(f'True/Uncertain(deter.) Sigmoid Comparison Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {IMPUTE_METHOD}')
    plt.tight_layout()
    plt.show()
    """
    
    if get_imputed_prediction_metrics:
        
        if unique_outcomes == 2:
            utils.create_metrics(y_complete, y_impute_hat_labels)
            plt.show()





##########################################################################################################################
# experiments -- col wise simulations ----------> get kde values of dataframe
##########################################################################################################################


if SIMULATE:
    
    """
        KDE COLLECTION -- ORIGINAL 
        --> is equal to the true distribution of the underlying data of the specific dataset
    """
    
    kde_collection_original = []
    
    for column in DATAFRAME.columns:
        values = DATAFRAME[column].dropna().values
        
        kde = stats.gaussian_kde(values)   
        #kde_vis = [kde]
        
        kde_collection_original.append(kde)
        
    
    
    """
        KDE COLLECTION -- UNCERTAIN 
        --> is equal to the uncertain distribution of the underlying data of the specific dataset with missing values
    """
    
    kde_collection_uncertain = []
    
    for column in DATAFRAME_SIMULATE.columns:
        values = DATAFRAME_SIMULATE[column].dropna().values
        
        kde = stats.gaussian_kde(values)   
        #kde_vis = [kde]
        
        kde_collection_uncertain.append(kde)
        
        
        """
            Comperative Visualization of Certain (True) and Uncertain Column Distribution
            --> good for analyzing the differences between the two distribtions
        """
        
        if compare_col_kde_distributions:
            # Print the KernelDensity parameters for the current column
            #print(f"Column: {column}")            
    
            data_visualization_joint = pd.DataFrame(data={"Certain Distribution // DATAFRAME":DATAFRAME[column], 
                                                          "Uncertain Distribution // DATAFRAME_SIMULATE":DATAFRAME_SIMULATE[column]})
    
            # KDE Plot of column without missing data
            plt.figure(figsize=(8, 4))
            sns.kdeplot(data=data_visualization_joint, common_grid=True)
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.title(f'KDE Plot of Column: {column} - Miss-Rate: {MISS_RATE} - Method: {SIMULATE_METHOD}')
            plt.tight_layout()
            plt.show()
        


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



##########################################################################################################################
# experiments modul 2 - with simulation --> missing data (row wise) --> useage of kde of columns to simulate outcome
##########################################################################################################################

# impute == false is equal to stochastic simulation approach
if SIMULATE == True:
    
    # step 0 --> first for loop for getting row and simulating this specific row
    
    uncertain_simulation_history_mean = []
    uncertain_simulation_history_mean_labels = []    
    
    
    original_simulation_history_mean = []
    original_simulation_history_mean_labels = []  
    
    
    if SIMULATION_RANGE == None:
        SIMULATION_RANGE = range(len(DATAFRAME_SIMULATE))
    
    
    print("\nPredictions for uncertain Dataset with uncertainties and simulated values:")
    
    
    for i in tqdm(SIMULATION_RANGE):
    
        # step 1: get row to work with
        
        DATAFRAME_SIMULATE_ROW = pd.DataFrame(DATAFRAME_SIMULATE.loc[i])
        
        
        # step 2: find all the attributes with nan values
        
        inds_to_key = np.where(DATAFRAME_SIMULATE_ROW.isna().all(axis=1))[0]
        inds_to_key = [column_names[i] for i in inds_to_key]
    
    
    
        # step 3: sample a value from the specific kde of the missing value - aka. beginning of MonteCarlo Simulation
        # --> safe created value for this row in a history
        
        sample_history_uncertain = []
        sample_history_original = [] 
        
        for key in inds_to_key:
            
            sample_uncertain = kde_collection_uncertain[key].resample(SIMULATION_LENGTH)
            sample_original = kde_collection_original[key].resample(SIMULATION_LENGTH)
          
            sample_history_uncertain.append(sample_uncertain.flatten())
            sample_history_original.append(sample_original.flatten())
        
        #test = sample_history_uncertain
        sample_history_uncertain = pd.DataFrame(sample_history_uncertain).transpose()
        sample_history_uncertain.columns = inds_to_key
    
        sample_history_original = pd.DataFrame(sample_history_original).transpose()
        sample_history_original.columns = inds_to_key
    
    
        # step 4: create DATAFRAME for faster simulation (basis input) and replace missing values with sampled ones   
        
        # index of DATAFRAME_MISS_ROW is now equal to number of simulations
        DATAFRAME_MC_SIMULATION = DATAFRAME_SIMULATE_ROW.copy().transpose()
        DATAFRAME_MC_SIMULATION = pd.concat([DATAFRAME_MC_SIMULATION]*SIMULATION_LENGTH, ignore_index=True)
        
        UNCERTAIN_DATAFRAME_MC_SIMULATION = DATAFRAME_MC_SIMULATION.copy()
        ORIGINAL_DATAFRAME_MC_SIMULATION = DATAFRAME_MC_SIMULATION.copy()
        
        # replace the missing values of DATAFRAME_MISS_ROW/ (now MC_SIMULATION) with the created samples 
        for col in inds_to_key:
            UNCERTAIN_DATAFRAME_MC_SIMULATION[col] = sample_history_uncertain[col]
            ORIGINAL_DATAFRAME_MC_SIMULATION[col] = sample_history_original[col]
        
        
        
        #step 5.a: row prediction on uncertain samples
        
        """
            -----> Simulation procedure for uncertain kde induced simulation frames
        """
        
        X_uncertain_simulation = UNCERTAIN_DATAFRAME_MC_SIMULATION.iloc[:, 0:-1]
        y_uncertain_simulation = UNCERTAIN_DATAFRAME_MC_SIMULATION[column_names[-1]]
        
        y_uncertain_simulation_hat = model.predict(X_uncertain_simulation, verbose=0).flatten()
        y_uncertain_simulation_hat_mean = y_uncertain_simulation_hat.mean()
        y_uncertain_simulation_hat_std = y_uncertain_simulation_hat.std()
        y_uncertain_simulation_hat_labels = (y_uncertain_simulation_hat>0.5).astype("int32")
        y_uncertain_simulation_joint = np.stack([y_uncertain_simulation_hat, y_uncertain_simulation_hat_labels], 1)
        y_uncertain_simulation_joint = pd.DataFrame(y_uncertain_simulation_joint, columns=["sigmoid", "label"])
    
    
        uncertain_simulation_history_mean.append(y_uncertain_simulation_hat_mean)
        uncertain_simulation_history_mean_labels.append((y_uncertain_simulation_hat_mean>0.5).astype("int32"))
    
    
    
        #step 5.b: row prediction on original samples
        
        """
            -----> Simulation procedure for true original kde induced simulation frames
        """
    
        X_original_simulation = ORIGINAL_DATAFRAME_MC_SIMULATION.iloc[:, 0:-1]
        y_original_simulation = ORIGINAL_DATAFRAME_MC_SIMULATION[column_names[-1]]
        
        y_original_simulation_hat = model.predict(X_original_simulation, verbose=0).flatten()
        y_original_simulation_hat_mean = y_original_simulation_hat.mean()
        y_original_simulation_hat_std = y_original_simulation_hat.std()
        y_original_simulation_hat_labels = (y_original_simulation_hat>0.5).astype("int32")
        y_original_simulation_joint = np.stack([y_original_simulation_hat, y_original_simulation_hat_labels], 1)
        y_original_simulation_joint = pd.DataFrame(y_original_simulation_joint, columns=["sigmoid", "label"])
    
    
        original_simulation_history_mean.append(y_original_simulation_hat_mean)
        original_simulation_history_mean_labels.append((y_original_simulation_hat_mean>0.5).astype("int32"))
    
    
    
        if simulation_visualizations:
        
            """
                comparison of both simulations
            """
            y_compare_simulation_joint = np.stack([y_original_simulation_hat, y_uncertain_simulation_hat], 1)
            y_compare_simulation_joint = pd.DataFrame(y_compare_simulation_joint, columns=["Original_KDE_Simulation", "Uncertain_KDE_Simulation"])
            
        
        
            # step 6: visualize simulated predictions
            
            
            """
                Plot_1: Histogam which shows the simulated row sigmoid results with hue 
            """
            # visualize predictions with hist plots
            plt.figure(figsize=(10, 6))
            sns.histplot(data=y_uncertain_simulation_joint, x="sigmoid", hue="label", hue_order=[0, 1], bins=15, binrange=(0, 1), stat="count", kde=False, kde_kws={"cut":0})
            
            plt.axvline(x=y_complete[i], linewidth=4, linestyle = "-", color = "green", label="Original Label")
            plt.axvline(x=y_complete_hat_labels[i], linewidth=4, linestyle = "-", color = "red", label="Predicted Model Label")
            
            if IMPUTE:
                plt.axvline(x=y_impute_hat[i], linewidth=2, linestyle = "--", color = "purple", label="Imputated Prediction") # impute prediction
            
            plt.axvline(x=y_original_simulation_hat_mean, linewidth=2, linestyle = "-.", color = "black", label="Orig. Mean Sim. Value") # orig. simulation prediction mean
            plt.axvline(x=y_uncertain_simulation_hat_mean, linewidth=2, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # uncert. simulation prediction mean
            
            plt.title(f'Row: {i} Uncertain Sim. Output Hist Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {SIMULATE_METHOD}')
            plt.legend(["Original Label", "Predicted Model Label", "Imputated Prediction", "Orig. Mean Sim. Value", "Uncert. Mean Sim. Value", "Label 1", "Label 0"])
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
            
            
            """
            plt.figure(figsize=(10, 6))
            sns.histplot(data=y_original_simulation_joint, x="sigmoid", hue="label", hue_order=[0, 1], bins=15, binrange=(0, 1), stat="count", kde=False, kde_kws={"cut":0})
            plt.axvline(x=y_complete[i], linewidth=2, linestyle = "-", color = "red", label="True Value")
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Frequency')
            plt.title(f'Row: {i} Original Sim. Output Hist Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {SIMULATE_METHOD}')
            plt.tight_layout()
            plt.legend(["True Label", "Label 1", "Label 0"])
            if IMPUTE:
                plt.axvline(x=y_impute_hat[i], linewidth=2, linestyle = "--", color = "green") # impute prediction
            plt.show()
            
            
    
            # visualize predictions with kde plots without hue
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=y_compare_simulation_joint, common_grid=True)
            plt.axvline(x=y_complete[i], linewidth=1, linestyle = "-", color = "red", label="True Value")
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Density')
            plt.title(f'Row: {i} Uncertain Sim. Output Density Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {SIMULATE_METHOD}')
            plt.tight_layout()
            plt.legend(["Uncertain_KDE_Distrubution", "Original_KDE_Distribution", "True Value"])
            if IMPUTE:
                plt.axvline(x=y_impute_hat[i], linewidth=1, linestyle = "--", color = "green")
            plt.show()
            """
            
            """
            # visualize predictions with kde plots with hue
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=y_uncertain_simulation_joint, x="sigmoid", hue="label", hue_order=[0, 1], common_grid=True)
            plt.axvline(x=y_complete[i], linewidth=1, linestyle = "-", color = "red", label="True Value")
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Density')
            plt.title(f'Row: {i} Uncertain Sim. Output Density Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {SIMULATE_METHOD}')
            plt.tight_layout()
            plt.legend(["True Label", "Label 1", "Label 0"])
            plt.show()
            
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=y_original_simulation_joint, x="sigmoid", hue="label", hue_order=[0, 1], common_grid=True)
            plt.axvline(x=y_complete[i], linewidth=1, linestyle = "-", color = "red", label="True Value")
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Density')
            plt.title(f'Row: {i} Original Sim. Output Density Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {SIMULATE_METHOD}')
            plt.tight_layout()
            plt.legend(["True Label", "Label 1", "Label 0"])
            plt.show()
            """
            
            
            
            
            
            
            """
            #
            #
            #
            #
            #
                Plot_2: Visualizing the spred of the simulation == underlying uncertainty of the prediction induced by uncertain data
            #
            #
            #
            #
            #
            
            
            #x-axis ranges from 0 and 1 with .001 steps
            x = np.arange(0.0, 1.0, 0.001)
            
            #plot normal distribution with mean 0 and standard deviation 1
            plt.plot(x, stats.norm.pdf(x, y_original_simulation_hat_mean, y_original_simulation_hat_std), label="Orig. Sim. Distribution", color="black", linestyle = "-")
            plt.axvline(x=y_original_simulation_hat_mean, linewidth=1, linestyle = "-.", color = "black", label="Orig. Mean Sim. Value") # mean original kde prediction
            
            plt.plot(x, stats.norm.pdf(x, y_uncertain_simulation_hat_mean, y_uncertain_simulation_hat_std), label="Uncert. Sim. Distribution", color="grey", linestyle = "--")
            plt.axvline(x=y_uncertain_simulation_hat_mean, linewidth=1, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # mean uncertain kde prediction
            
            plt.axvline(x=y_complete[i], linewidth=2, linestyle = "-", color = "green", label="Original Label")
            plt.axvline(x=y_complete_hat_labels[i], linewidth=2, linestyle = "-", color = "red", label="Predicted Model Label")
            
            if IMPUTE:
                plt.axvline(x=y_impute_hat[i], linewidth=1, linestyle = "--", color = "purple", label="Imputated Prediction") # impute prediction
            
            plt.title(f'Row: {i} Underlying Uncertainty of the Simulation - Miss-Rate: {MISS_RATE} - Impute-Method: {SIMULATE_METHOD}')
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
            #plt.legend()
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Density of Sigmoid Activations')
            #plt.tight_layout()
            plt.show()
            """
            
            
            """
                KDE PLOT of Uncerlying uncertainty
            """
            
            #x-axis ranges from 0 and 1 with .001 steps
            x = np.arange(0.0, 1.0, 0.001)
            
            #plot normal distribution with mean 0 and standard deviation 1
            plt.plot(x, stats.norm.pdf(x, y_original_simulation_hat_mean, y_original_simulation_hat_std), label="Orig. Sim. Distribution", color="black", linestyle = "-")
            plt.axvline(x=y_original_simulation_hat_mean, linewidth=1, linestyle = "-.", color = "black", label="Orig. Mean Sim. Value") # mean original kde prediction
            
            kde_plot = stats.gaussian_kde(y_uncertain_simulation_hat).pdf(x) 
            plt.plot(x, kde_plot, label="Uncertain. Sim. Distribution // KDE", color="pink", linestyle = "--")
            
            plt.plot(x, stats.norm.pdf(x, y_uncertain_simulation_hat_mean, y_uncertain_simulation_hat_std), label="Uncert. Sim. Distribution", color="grey", linestyle = "--")
            plt.axvline(x=y_uncertain_simulation_hat_mean, linewidth=1, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # mean uncertain kde prediction
            
            plt.axvline(x=y_complete[i], linewidth=2, linestyle = "-", color = "green", label="Original Label")
            plt.axvline(x=y_complete_hat_labels[i], linewidth=2, linestyle = "-", color = "red", label="Predicted Model Label")
            
            if IMPUTE:
                plt.axvline(x=y_impute_hat[i], linewidth=1, linestyle = "--", color = "purple", label="Imputated Prediction") # impute prediction
            
            plt.title(f'Row: {i} Underlying Uncertainty of the Simulation - Miss-Rate: {MISS_RATE} - Impute-Method: {SIMULATE_METHOD}')
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
            #plt.legend()
            plt.xlabel('Sigmoid Activations')
            plt.ylabel('Density of Sigmoid Activations')
            #plt.tight_layout()
            plt.show()
    
    
    
    
    
    
    
    
    """
            ----------------> simulations process end ---> further analysis below
    """
    
    """
        Below: Comparisons between the prediction results of Uncertain and Certain KDE simulations
    """
    
    y_uncertain_simulation_history_joint = np.stack([uncertain_simulation_history_mean, uncertain_simulation_history_mean_labels], 1)
    y_uncertain_simulation_history_joint = pd.DataFrame(y_uncertain_simulation_history_joint, columns=["sigmoid", "label"])
    
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=y_uncertain_simulation_history_joint, x="sigmoid", hue="label", hue_order=[0, 1], bins=15, binrange=(0, 1), stat="count", kde=False, kde_kws={"cut":0})
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Frequency')
    plt.title(f'Uncertain (Unc. Stoch.) Combined Output Hist Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {SIMULATE_METHOD}')
    plt.tight_layout()
    plt.show()
    
    
    y_original_simulation_history_joint = np.stack([original_simulation_history_mean, original_simulation_history_mean_labels], 1)
    y_original_simulation_history_joint = pd.DataFrame(y_original_simulation_history_joint, columns=["sigmoid", "label"])
    
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=y_original_simulation_history_joint, x="sigmoid", hue="label", hue_order=[0, 1], bins=15, binrange=(0, 1), stat="count", kde=False, kde_kws={"cut":0})
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Frequency')
    plt.title(f'Uncertain (Ori. Stoch.) Combined Output Hist Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {SIMULATE_METHOD}')
    plt.tight_layout()
    plt.show()
    
    
    if get_simulated_prediction_metrics:
        
        utils.create_metrics(y_complete, uncertain_simulation_history_mean_labels)
        plt.show()
    
    
    #test = pd.DataFrame(uncertain_simulation_history)






"""
    ---> Comparison of everything
    
    Explanation of DATAFRAME_COMBINED_RESULTS:
        - Original Label is equal to the Label which is found originally in the dataset
        - 0: is the shortcut for a prediction with a trained model on full data without uncertainties
            -> only uncertainties found here are model uncertainties 
        - 1: is the shortcut for predictions with imputed values
        
        - 2: simulation results - metric mean 
        
"""
if IMPUTE == True and SIMULATE == True:
    
    if len(SIMULATION_RANGE) == len(DATAFRAME_SIMULATE):
        
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
    # ...

# Summary and discussion

# Limitations and model uncertainty predictions (bayes &&&)




