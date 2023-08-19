# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:24:32 2023

@author: Selii
"""

import os
import sys
import pickle
import random

import utils

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KernelDensity

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
#from sklearn.impute import IterativeImputer

from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

import chaospy as cp

##########################################################################################################################
# set important paths
##########################################################################################################################


# set path to different folders
dataset_path = os.path.join(os.getcwd(), 'datasets')
image_path = os.path.join(os.getcwd(), 'images')
model_path = os.path.join(os.getcwd(), 'models')




##########################################################################################################################
# set constant settings
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

# train or load model
train_model = False
load_model = True

get_true_prediction_metrics = False



bw_method='scott' 
bw_adjust=1



##########################################################################################################################
# load datasets
##########################################################################################################################
    
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
    




# load data for climate modal simulation crashes dataset
if dataset == "climate_simulation":
    
    with open(os.path.join(dataset_path, dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep="\s+", engine='python', header = 0)
    
    # drop the first two elements of the dataset -> not relevant
    DATAFRAME = DATAFRAME.iloc[:, 2:]

    column_names = DATAFRAME.columns.to_list()
    
    
    
    
# load data for climate modal simulation crashes dataset
if dataset == "wdbc":
    
    with open(os.path.join(dataset_path, dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep=",", engine='python', header = None)
    
    # drop the first 
    y_complete = DATAFRAME.iloc[:,1].copy()
    #y_complete = y_complete.rename("Outcome")
    DATAFRAME = DATAFRAME.iloc[:, 2:].copy()
    
    DATAFRAME = DATAFRAME.merge(y_complete, left_index=True, right_index=True)
    
    target_names = (['B', 'M'], [0, 1])
    DATAFRAME.iloc[:,-1].replace(target_names[0], target_names[1], inplace=True)
    
    column_names = DATAFRAME.columns.to_list()




# load data for predict+students+dropout+and+academic+success dataset
if dataset == "predict+students+dropout+and+academic+success":
    
    DATAFRAME = pd.read_csv(os.path.join(dataset_path, dataset + ".csv"), sep=";")
    
    target_names = (['Dropout', 'Enrolled' ,'Graduate'], [2, 0, 1])
    
    DATAFRAME['Target'].replace(target_names[0], target_names[1], inplace=True)
    
    column_names = DATAFRAME.columns.to_list()




#DATAFRAME = DATAFRAME.iloc[:,15:].copy()
#column_names = column_names[15:]


##########################################################################################################################
# standardization of values for better performance
##########################################################################################################################
    

if standardize_data:
    # use data scaler to norm the data
    scaler = MinMaxScaler()
    standardized = scaler.fit_transform(DATAFRAME)
    
    # change to dataframe
    DATAFRAME = pd.DataFrame(standardized)
    DATAFRAME.columns = column_names




##########################################################################################################################
# visiualize_data Dataframe 
##########################################################################################################################


if visiualize_data:
    
    # Plotting combined distribution using box plots
    DATAFRAME.boxplot(column=column_names, figsize=(12, 6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
    # Plotting combined distribution using histograms
    DATAFRAME.hist(column=column_names, bins=15, figsize=(12, 10))
    plt.tight_layout()
    plt.show()
    
    """
    # Visualizing correlation between variables using a heatmap
    corr_matrix = DATAFRAME.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
    plt.tight_layout()
    plt.show()
    
    
    # Create a KDE plot for each column
    for column in column_names:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(data=DATAFRAME[column], fill=True, color='skyblue', alpha=0.5, bw_method=bw_method, bw_adjust=bw_adjust)
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.title(f'KDE Plot of {column}')
        plt.tight_layout()
        plt.show()
    """   
        
    # Create a combined KDE plot
    plt.figure(figsize=(12, 6))
    for column in column_names[:-1]:
        sns.kdeplot(data=DATAFRAME[column], fill=True, label=column, bw_method=bw_method, bw_adjust=bw_adjust)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Combined Input KDE Plot')
    #plt.legend(False)
    plt.tight_layout()
    plt.show()




##########################################################################################################################
# Split data into X and y - optional scaling of data
##########################################################################################################################

'''
X_complete = DATAFRAME.iloc[:, 0:-1]
y_complete = DATAFRAME[column_names[-1]]


X_complete_train, X_complete_test, y_complete_train,  y_complete_test = train_test_split(X_complete, y_complete, test_size=0.25, random_state=RANDOM_STATE)


##########################################################################################################################
# create standard vanilla feed forward neural network
##########################################################################################################################


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


    # save new model
    model.save(os.path.join(model_path, dataset + "_model"))




##########################################################################################################################
# load model without training
##########################################################################################################################


if load_model:
    model = keras.models.load_model(os.path.join(model_path, dataset + "_model"))
    model.summary()




##########################################################################################################################
# some metric for the predictions
##########################################################################################################################


y_complete_hat = model.predict(X_complete)
y_complete_hat_labels = (y_complete_hat>0.5).astype("int32")


if get_true_prediction_metrics:
    
    utils.create_metrics(y_complete, y_complete_hat_labels)
    plt.show()



#DATAFRAME.iloc[:, :-1].plot.kde(column=column_names[:-1], figsize=(12, 10))
#plt.tight_layout()
#plt.show()


#sns.kdeplot(y_complete_hat)
#plt.show()



##########################################################################################################################
# remove data in original dataframe
##########################################################################################################################


# create new Dataset with random missing values
DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=0.5, random_seed=RANDOM_STATE) 
DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)

'''


##########################################################################################################################
# sample Kernel Density Estimate over missing dataset
##########################################################################################################################

"""
    # Monte Carlo Simulation with induced uncertainty
"""


