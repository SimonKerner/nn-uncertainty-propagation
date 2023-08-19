# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:24:32 2023

@author: Selii
"""

import os
import sys
#import pickle
#import random

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


import chaospy as cp


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
    -predict+students+dropout+and+academic+success
    
following all the different settings for this simulation run can be found
    -dataset = "choose dataset"
    -standardize_dataset = "used for standardizing the dataset -- values between 0 and 1 -- minmax"
"""
##########################################################################################################################

#choose working dataset: "australian" or "climate_simulation", "predict+students+dropout+and+academic+success", "wdbc" -> Breast Cancer Wisconsin (Diagnostic)
dataset = "wdbc" 


# set random state
RANDOM_STATE = None
np.random.seed(RANDOM_STATE)
np.random.RandomState(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# further settings
standardize_data = True
visiualize_data = False


use_normal_frame = True
use_probability_frame = False


# train or load model
train_model = False
load_model = True

get_true_prediction_metrics = True



bw_method='scott' 
bw_adjust=1



##########################################################################################################################
# load datasets
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



##########################################################################################################################
# standardization of values for better performance
##########################################################################################################################
    

if standardize_data:
    # use data scaler to norm the data
    scaler = MinMaxScaler()

    # change to dataframe
    DATAFRAME = pd.DataFrame(scaler.fit_transform(DATAFRAME))
    DATAFRAME.columns = column_names


DATAFRAME_describe = DATAFRAME.describe()




##########################################################################################################################
# visiualize true underlying data of Dataframe 
##########################################################################################################################


if visiualize_data:
    
    # Plotting combined distribution using histograms
    DATAFRAME.hist(column=column_names, bins=15, figsize=(12, 10), density=True, sharey=False, sharex=True)
    #plt.xlabel('Sigmoid Activations')
    #plt.ylabel('Density')
    plt.title('Input without missing data')
    plt.tight_layout()
    plt.show()
    
    
    

##########################################################################################################################
# choose frame mode and perform train - test - split
##########################################################################################################################

if use_normal_frame:
    
    X_complete = DATAFRAME.iloc[:, 0:-1]
    y_complete = DATAFRAME[column_names[-1]]
    
    
    X_complete_train, X_complete_test, y_complete_train,  y_complete_test = train_test_split(X_complete, y_complete, test_size=0.25, random_state=RANDOM_STATE)



# transform dataframe values into PDF values of corresponding DataFrame KDE values
if use_probability_frame:
 
    def frame_to_probability_frame(DATAFRAME, column_names):
        
        X_complete = DATAFRAME.iloc[:, 0:-1]
        
        # get KDE values of each column (X_complete) 
        DATAFRAME_KDE = []
        for i in range(len(column_names[:-1])):
            column_data = X_complete.iloc[:,i]
            column_kde = stats.gaussian_kde(column_data)
            
            DATAFRAME_KDE.append(column_kde)
        
            #print(column_kde.pdf(0))
        
        
        # transform values of X_complete into KDE probabilities
        DATAFRAME_PROBABILITY = []
        for row in range(len(X_complete)):
            get_row = DATAFRAME.loc[row].transpose()
            get_row_values = np.array(get_row)[:-1]
            get_row_label = np.array(get_row)[-1]
        
            # get probabilities PDF of original input value
            get_row_pdf = []
            for i, j in enumerate(DATAFRAME_KDE):
                get_row_pdf.append(np.float64(j.pdf(get_row_values[i])))
                
            get_row_pdf.append(get_row_label)
            
            DATAFRAME_PROBABILITY.append(get_row_pdf)
        
        #print(get_row_pdf)
    
        return DATAFRAME_PROBABILITY
    
    
    
    DATAFRAME_PROBABILITY = frame_to_probability_frame(DATAFRAME, column_names)
    
    DATAFRAME_PROBABILITY = np.array(DATAFRAME_PROBABILITY)
    DATAFRAME_PROBABILITY = pd.DataFrame(data=DATAFRAME_PROBABILITY, columns=column_names)    
    
    
    X_complete = DATAFRAME_PROBABILITY.iloc[:, 0:-1]
    y_complete = DATAFRAME_PROBABILITY[column_names[-1]]
    
    X_complete_train, X_complete_test, y_complete_train,  y_complete_test = train_test_split(X_complete, y_complete, test_size=0.25, random_state=RANDOM_STATE)
 




##########################################################################################################################
# create standard vanilla feed forward feural network
##########################################################################################################################

"""
#TODO
#TODO
#TODO   NN trained specifically for probabilistic input data --> probability frame
#TODO
#TODO
"""



if train_model:
    # layers of the network
    inputs = keras.Input(shape=(X_complete.shape[1]))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    # build model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # compile model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])
    
    
    # fit model
    model_history = model.fit(X_complete_train, y_complete_train, validation_data=[X_complete_test, y_complete_test], batch_size=15, epochs=50, verbose=0)
    
    
    # plot model
    utils.plot_history(model_history)


    if use_normal_frame:
        # save new model
        model.save(os.path.join(model_path, dataset + "_model"))
        
      
    if use_probability_frame:
        # save new model
        model.save(os.path.join(model_path, dataset + "_model_probability"))




##########################################################################################################################
# load model without training
##########################################################################################################################


if load_model:
    
    if use_normal_frame:
        model = keras.models.load_model(os.path.join(model_path, dataset + "_model"))
        #model.summary()
        
      
    if use_probability_frame:
        model = keras.models.load_model(os.path.join(model_path, dataset + "_model_probability"))
        #model.summary()




##########################################################################################################################
# singe prediction metrics with a perfectly trained model - no uncertainties -- deterministic as usual
##########################################################################################################################


y_complete_hat = model.predict(X_complete).flatten()
y_complete_hat_labels = (y_complete_hat>0.5).astype("int32")
y_complete_joint = np.stack([y_complete_hat, y_complete_hat_labels], 1)
y_complete_joint = pd.DataFrame(y_complete_joint, columns=["sigmoid", "label"])

"""
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


"""

if get_true_prediction_metrics:
    
    utils.create_metrics(y_complete, y_complete_hat_labels)
    plt.show()




##########################################################################################################################
# introduce missing data - aka. aleatoric uncertainty
##########################################################################################################################


# get KDE for each column
MISS_RATE=0.5


if use_normal_frame:
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=MISS_RATE, random_seed=RANDOM_STATE) 
    DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)
    
    
if use_probability_frame:
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME_PROBABILITY.iloc[:, :-1], miss_rate=MISS_RATE, random_seed=RANDOM_STATE) 
    DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME_PROBABILITY.iloc[:,-1], left_index=True, right_index=True)


if visiualize_data:
    
    # Plotting combined distribution using histograms
    DATAFRAME_MISS.hist(column=column_names, bins=15, figsize=(12, 10), density=True)
    #plt.xlabel('Sigmoid Activations')
    #plt.ylabel('Density')
    plt.title('Input with missing data')
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

IMPUTE = False

# choose between kde_imputer, SimpleImputer//mean, SimpleImputer//median, SimpleImputer//most_frequent, KNNImputer
IMPUTE_METHOD = "SimpleImputer//mean"

if IMPUTE and IMPUTE_METHOD == "kde_imputer":
    DATAFRAME_IMPUTE = utils.kde_Imputer(DATAFRAME_MISS, kernel="gaussian", bandwidth="scott")
    
elif IMPUTE and IMPUTE_METHOD == "SimpleImputer//mean":
    simp_imp = SimpleImputer(strategy="mean")
    DATAFRAME_IMPUTE = pd.DataFrame(simp_imp.fit_transform(DATAFRAME_MISS), columns=column_names)
    
elif IMPUTE and IMPUTE_METHOD == "SimpleImputer//median":
    simp_imp = SimpleImputer(strategy="median")
    DATAFRAME_IMPUTE = pd.DataFrame(simp_imp.fit_transform(DATAFRAME_MISS), columns=column_names)

elif IMPUTE and IMPUTE_METHOD == "SimpleImputer//most_frequent":
    simp_imp = SimpleImputer(strategy="most_frequent")
    DATAFRAME_IMPUTE = pd.DataFrame(simp_imp.fit_transform(DATAFRAME_MISS), columns=column_names)
    
elif IMPUTE and IMPUTE_METHOD == "KNNImputer":
    knn_imp = KNNImputer(n_neighbors=5)
    DATAFRAME_IMPUTE = pd.DataFrame(knn_imp.fit_transform(DATAFRAME_MISS), columns=column_names)
    
elif IMPUTE == False:
    DATAFRAME_IMPUTE = DATAFRAME_MISS
    IMPUTE_METHOD = "Column_KDE"
    
else:
    print("Error: None of the Imputation Methods was found")




##########################################################################################################################
# experiments -- col wise simulations 
##########################################################################################################################


######
"""
    col wise experiment - of DATAFRAME
"""

#KDE_VALUES OF EACH COLUMN
print_info = False


if use_normal_frame:
        
    kde_collection_original = []
    
    for column in DATAFRAME.columns:
        values = DATAFRAME[column].dropna().values
        
        kde = stats.gaussian_kde(values)   
        #kde_vis = [kde]
        
        kde_collection_original.append(kde)
        
    # to convert lists to dictionary
    kde_collection_original = {column_names[i]: kde_collection_original[i] for i in range(len(column_names))}
    
    
    
    
    kde_collection_uncertain = []

    
    for column in DATAFRAME_IMPUTE.columns:
        values = DATAFRAME_IMPUTE[column].dropna().values
        
        kde = stats.gaussian_kde(values)   
        #kde_vis = [kde]
        
        if print_info:
            # Print the KernelDensity parameters for the current column
            print(f"Column: {column}")
            
            
            """
            get the difference between true and underlying missing rate distrubution
            """
            
    
            data_visualization_joint = pd.DataFrame(data={"True Distribution // DATAFRAME":DATAFRAME[column], 
                                                          "False Distribution // DATAFRAME_IMPUTE":DATAFRAME_IMPUTE[column]})
    
            # KDE Plot of column without missing data
            plt.figure(figsize=(8, 4))
            sns.kdeplot(data=data_visualization_joint, common_grid=True)
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.title(f'KDE Plot of Column: {column} - Miss-Rate: {MISS_RATE} - Method: {IMPUTE_METHOD}')
            plt.tight_layout()
            plt.show()
        
        kde_collection_uncertain.append(kde)


    # to convert lists to dictionary
    kde_collection_uncertain = {column_names[i]: kde_collection_uncertain[i] for i in range(len(column_names))}





if use_probability_frame:
        
    kde_collection_original = []
    
    for column in DATAFRAME_PROBABILITY.columns:
        values = DATAFRAME_PROBABILITY[column].dropna().values
        
        kde = stats.gaussian_kde(values)   
        #kde_vis = [kde]
        
        kde_collection_original.append(kde)
    
    # to convert lists to dictionary
    kde_collection_original = {column_names[i]: kde_collection_original[i] for i in range(len(column_names))}
    
    
    
    
    kde_collection_uncertain = []
    
    for column in DATAFRAME_IMPUTE.columns:
        values = DATAFRAME_IMPUTE[column].dropna().values
        
        kde = stats.gaussian_kde(values)   
        #kde_vis = [kde]
        
        if print_info:
            # Print the KernelDensity parameters for the current column
            print(f"Column: {column}")
            
            
            """
            get the difference between true and underlying missing rate distrubution
            """
            
    
            data_visualization_joint = pd.DataFrame(data={"True Distribution // DATAFRAME":DATAFRAME_PROBABILITY[column], 
                                                          "False Distribution // DATAFRAME_IMPUTE":DATAFRAME_IMPUTE[column]})
    
            # KDE Plot of column without missing data
            plt.figure(figsize=(8, 4))
            sns.kdeplot(data=data_visualization_joint, common_grid=True)
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.title(f'KDE Plot of Column: {column} - Miss-Rate: {MISS_RATE} - Method: {IMPUTE_METHOD}')
            plt.tight_layout()
            plt.show()
        
        kde_collection_uncertain.append(kde)
        
        
    # to convert lists to dictionary
    kde_collection_uncertain = {column_names[i]: kde_collection_uncertain[i] for i in range(len(column_names))}
    
    
"""   ##-----------------------------------> just some tests
    
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






##########################################################################################################################
# experiments -- col wise imputation simulations 
##########################################################################################################################

count_missing = DATAFRAME_IMPUTE.isnull().sum().sum()
print("Count of missing values:", count_missing)


##########################################################################################################################
# experiments modul 1 - with imputation --> full data --> get_predictions
##########################################################################################################################

if IMPUTE:
    
    X_impute = DATAFRAME_IMPUTE.iloc[:, 0:-1]
    y_impute = DATAFRAME_IMPUTE[column_names[-1]]
    
    y_impute_hat = model.predict(X_impute).flatten()
    y_impute_hat_labels = (y_impute_hat>0.5).astype("int32")
    y_impute_joint = np.stack([y_impute_hat, y_impute_hat_labels], 1)
    y_impute_joint = pd.DataFrame(y_impute_joint, columns=["sigmoid", "label"])
    
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=y_impute_joint, x="sigmoid", hue="label", bins=10, stat="density", kde=True, kde_kws={"cut":0})
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Frequency')
    plt.title(f'Uncertain Combined Output Hist Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {IMPUTE_METHOD}')
    plt.tight_layout()
    plt.show()
    
    
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=y_impute_joint, x="sigmoid", hue="label", common_grid=True, cut=0)
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Density')
    plt.title(f'Uncertain Combined Output Density Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {IMPUTE_METHOD}')
    plt.tight_layout()
    plt.show()
    
    
    # compare against true distribution
    
    y_compare_joint = pd.concat([y_complete_joint, y_impute_joint], axis=1, ignore_index=True, sort=False)
    y_compare_joint.columns = ["True_Sigmoid", "True_Label", "Uncertain_Sigmoid", "Uncertain_Label"]
    y_compare_sigs = pd.DataFrame(data=[y_compare_joint["True_Sigmoid"], y_compare_joint["Uncertain_Sigmoid"]]).transpose()
    
    plt.figure(figsize=(10, 6))
    #sns.kdeplot(data=y_compare_sigs, common_grid=True, cut=0)
    sns.histplot(data=y_compare_sigs, bins=15)
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Density')
    plt.title(f'True/Uncertain(deter.) Sigmoid Comparison Plot - Miss-Rate: {MISS_RATE} - Impute-Method: {IMPUTE_METHOD}')
    plt.tight_layout()
    plt.show()
    
    
    if get_true_prediction_metrics:
        
        utils.create_metrics(y_complete, y_impute_hat_labels)
        plt.show()



##########################################################################################################################
# experiments modul 1 - with simulation --> missing data --> useage of kde of columns to simulate outcome
##########################################################################################################################

# step 0 --> first for loop for getting row and simulating this specific row


SIMULATION_LENGTH = 100


for i in range(len(DATAFRAME_MISS)):

    # step 1: get row to work with
    
    DATAFRAME_MISS_ROW = pd.DataFrame(DATAFRAME_MISS.loc[0])
    
    
    # step 2: find all the attributes with nan values
    
    inds_to_key = np.where(DATAFRAME_MISS_ROW.isna().all(axis=1))[0]
    inds_to_key = [column_names[i] for i in inds_to_key]


    sample_history_uncertain = []
    sample_history_original = []  
    
    
    # step 3: sample a value from the specific kde of the missing value - aka. beginning of MonteCarlo Simulation
    # --> safe created value for this row in a history
    
    for key in inds_to_key:
        
        sample_uncertain = kde_collection_uncertain[key].resample(SIMULATION_LENGTH)
        sample_original = kde_collection_original[key].resample(SIMULATION_LENGTH)
      
        sample_history_uncertain.append(sample_uncertain.flatten())
        sample_history_original.append(sample_original.flatten())
      
    
    sample_history_uncertain = sample_history_uncertain
    
    
    sample_history_uncertain = pd.DataFrame(sample_history_uncertain, columns=inds_to_key) 

    
    DATAFRAME_MISS_ROW = DATAFRAME_MISS_ROW.transpose()
    DATAFRAME_MISS_ROW = pd.concat([DATAFRAME_MISS_ROW]*SIMULATION_LENGTH)













