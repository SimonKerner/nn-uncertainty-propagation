# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:24:32 2023

@author: Selii
"""

import os
import pickle

import utils

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from scipy.stats import gaussian_kde




##########################################################################################################################
# load datasets
##########################################################################################################################


# set path to dataframes
datasets = os.path.join(os.getcwd(), 'datasets')

    
with open(os.path.join(datasets, "data_australian.pkl"), 'rb') as file:
    data_australian = pickle.load(file)
    
    
with open(os.path.join(datasets, "saved_pdfs_australian.pkl"), 'rb') as file:
    data_australian_pdf = pickle.load(file)
    
    
with open(os.path.join(datasets, "australian.dat"), 'rb') as file:
    data = pd.read_table(file, sep=" ", engine="python", header=None)
    
    
# rename columns   
column_names=["Sex", "Age", "Mean time at adresses", "Home status", "Current occupation",
          "Current job status", "Mean time with employers", "Other investments",
          "Bank account", "Time with bank", "Liability reference", "Account reference",
          "Monthly housing expense", "Savings account balance", "0 - Reject / 1 - Accept"]


original_col_names = ["0", "1", "2", "3", "4", "5" , "6", "7", "8", "9", "10", "11", "12", "13", "14"]


# rename columns
data.columns = column_names
data_australian.columns = column_names
    



##########################################################################################################################
# set Variables to work with + standardization of values for better performance
##########################################################################################################################
    

# choose dataframe to work with
DATAFRAME = data 

# set random state
RANDOM_STATE = 24
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# use data scaler to norm the data
scaler = MinMaxScaler()
standardized = scaler.fit_transform(DATAFRAME)

# change to dataframe
DATAFRAME = pd.DataFrame(standardized)
DATAFRAME.columns = column_names




##########################################################################################################################
# visiualize Dataframe 
##########################################################################################################################


images = os.path.join(os.getcwd(), 'images')


visiualize = False

if visiualize:
    # plot Scatter matrix of dataframe
    sns.set_style("whitegrid")
    sns_plot = sns.PairGrid(DATAFRAME, diag_sharey=False, corner=False)
    sns_plot.map_lower(sns.kdeplot, levels=4, color = "red", fill=True, alpha=0.4)
    sns_plot.map_lower(sns.scatterplot, s=15)
    sns_plot.map_diag(sns.kdeplot, fill=True)
    plt.savefig(os.path.join(images, 'dataset_plot.png'))


    # Density and Histplot for exploration
    distplot_data = DATAFRAME.melt(var_name='column_names')
    distplot_plot = sns.displot(data=distplot_data, x='value', col='column_names', col_wrap=5, rug=False, kde=True, stat='density')
    plt.show()
    distplot_plot.savefig(os.path.join(images, 'distplot_plot.png'))
    plt.close()
    



##########################################################################################################################
# Split data into X and y - optional scaling of data
##########################################################################################################################


# not standardized dataframe
#X = data.iloc[:, 0:14]
#y = data["0 - Reject / 1 - Accept"]

# standardized dataframe
X = DATAFRAME.iloc[:, 0:14]
y = DATAFRAME["0 - Reject / 1 - Accept"]


X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)




##########################################################################################################################
# create standard vanilla feed forward neural network
##########################################################################################################################


# set path to models
models = os.path.join(os.getcwd(), 'models')


training = False

if training:
    # layers of the network
    inputs = keras.Input(shape=(14,))
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
    model_history = model.fit(X_train, y_train, validation_data=[X_test, y_test], batch_size=15, epochs=50)
    
    # plot model
    utils.plot_history(model_history)

    # save new model
    model.save(os.path.join(models, "standard_model"))




##########################################################################################################################
# load model without training
##########################################################################################################################


model = keras.models.load_model(os.path.join(models, "standard_model"))

#model.summary()




##########################################################################################################################
# some metric for the predictions
##########################################################################################################################

 
standard_predictions_sig = model.predict(X)
standard_predictions_scalar = (standard_predictions_sig>0.5).astype("int32")

metrics = True

if metrics:
    
    # output
    sns.kdeplot(standard_predictions_sig)
    
    # scores
    #accuracy = accuracy_score(y, standard_predictions_scalar)
    #precision = precision_score(y, standard_predictions_scalar)
    report = classification_report(y, standard_predictions_scalar, target_names=['Rejected', 'Accepted'])
    
    # Confusion Matrix
    confusion_matrix = confusion_matrix(y, standard_predictions_scalar)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['Rejected', 'Accepted'])
    display.plot()
    plt.savefig(os.path.join(images, 'ConfusionMatrix_Standard.png'))




##########################################################################################################################
# 
##########################################################################################################################










