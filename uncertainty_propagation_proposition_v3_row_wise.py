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

#from sklearn.impute import SimpleImputer
#from sklearn.impute import KNNImputer
#from sklearn.impute import IterativeImputer

#from scipy.stats import gaussian_kde
#from sklearn.neighbors import KernelDensity


##########################################################################################################################
# set important paths
##########################################################################################################################


# set path to different folders
dataset_path = os.path.join(os.getcwd(), 'datasets')



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

get_true_prediction_metrics = False



#bw_method='scott' 
#bw_adjust=1



##########################################################################################################################
# load datasets
##########################################################################################################################

    
# load data for climate modal simulation crashes dataset
if dataset == "wdbc":
    
    with open(os.path.join(dataset_path, dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep=",", engine='python', header = None)
    
    # drop the first 
    outcomes = DATAFRAME.iloc[:,1].copy()
    #y_complete = y_complete.rename("Outcome")
    DATAFRAME = DATAFRAME.iloc[:, 2:].copy()
    
    DATAFRAME = DATAFRAME.merge(outcomes, left_index=True, right_index=True)
    
    DATAFRAME.iloc[:,-1].replace(['B', 'M'], [0, 1], inplace=True)
    
    column_names = DATAFRAME.columns.to_list()



#DATAFRAME = DATAFRAME.iloc[:,15:].copy()
#column_names = column_names[15:]



DATAFRAME_describe = DATAFRAME.describe()


##########################################################################################################################
# standardization of values for better performance
##########################################################################################################################
    

if standardize_data:
    # use data scaler to norm the data
    scaler = MinMaxScaler()

    # change to dataframe
    DATAFRAME = pd.DataFrame(scaler.fit_transform(DATAFRAME))
    DATAFRAME.columns = column_names


DATAFRAME_stand_describe = DATAFRAME.describe()
