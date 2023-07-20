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


use_normal_frame = False

use_probability_frame = True


# train or load model
train_model = False
load_model = True

get_true_prediction_metrics = False



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




DATAFRAME = DATAFRAME.iloc[:,15:].copy()
column_names = column_names[15:]




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
"""

values = np.arange(0, 1, step=0.01)

gauss_kde = stats.gaussian_kde(DATAFRAME["Attribute: 15"], bw_method=bw_method)

kde_probs = gauss_kde.pdf(values)
#kde_probs = np.exp(kde_probs)


plt.hist(DATAFRAME["Attribute: 15"], bins=15, density=True)
plt.plot(values, kde_probs)
plt.show()



sys.exit()


"""

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
            column_kde = cp.GaussianKDE(column_data)
            
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
# singe prediction metrics
##########################################################################################################################


y_complete_hat = model.predict(X_complete).flatten()
y_complete_hat_labels = (y_complete_hat>0.5).astype("int32")
y_complete_joint = np.stack([y_complete_hat, y_complete_hat_labels], 1)
y_complete_joint = pd.DataFrame(y_complete_joint, columns=["sigmoid", "label"])

# visualize predictions
plt.figure(figsize=(10, 6))
sns.histplot(data=y_complete_joint, x="sigmoid", hue="label", bins=10, stat="density", kde=True, kde_kws={"cut":0})
plt.xlabel('Sigmoid Activations')
plt.ylabel('Frequency')
plt.title('True Combined Input Hist Plot')
plt.tight_layout()
plt.show()


# visualize predictions
plt.figure(figsize=(10, 6))
sns.kdeplot(data=y_complete_joint, x="sigmoid", hue="label", common_grid=True, cut=0)
plt.xlabel('Sigmoid Activations')
plt.ylabel('Density')
plt.title('True Combined Input Density Plot')
plt.tight_layout()
plt.show()



if get_true_prediction_metrics:
    
    utils.create_metrics(y_complete, y_complete_hat_labels)
    plt.show()













##########################################################################################################################
# singe prediction metrics
##########################################################################################################################


# get KDE for each column

DATAFRAME_MISS = utils.add_missing_values(DATAFRAME_PROBABILITY.iloc[:, :-1], miss_rate=0.2, random_seed=RANDOM_STATE) 
DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME_PROBABILITY.iloc[:,-1], left_index=True, right_index=True)


if visiualize_data:
    
    # Plotting combined distribution using histograms
    DATAFRAME_MISS.hist(column=column_names, bins=15, figsize=(12, 10), density=True, sharey=False, sharex=True)
    #plt.xlabel('Sigmoid Activations')
    #plt.ylabel('Density')
    plt.title('Input with missing data')
    plt.tight_layout()
    plt.show()




first_row = DATAFRAME_PROBABILITY.loc[0][:-1]
first_row_outcome = DATAFRAME_PROBABILITY.loc[0][-1]

first_row_miss = DATAFRAME_MISS.loc[0][:-1]
first_row_miss_outcome = DATAFRAME_MISS.loc[0][-1]

first_row_joint = np.stack([first_row, first_row_miss], 1)
first_row_joint = pd.DataFrame(first_row_joint, columns=["first_row", "first_row_miss"])


sns.kdeplot(first_row_joint, fill=True, bw_adjust=1)
plt.show()


#first_row_y_hat = model.predict(np.reshape(first_row.values, (-1,15)))
#first_row_miss_y_hat = model.predict(np.reshape(first_row_miss.values, (-1,15)))

















sys.exit()





##########################################################################################################################
##########################################################################################################################
# Approach 1 Sampling - everything is random - MC
##########################################################################################################################
##########################################################################################################################

"""
    MC - simulation with original data
"""
import random
sim_length = 100
sim_history = []

for i in range(sim_length):
    
    input_sample = []
    for col in (X_complete):
        input_sample.append(random.choice(np.array(X_complete[col])))

    input_sample=np.array([input_sample])
    
    y_hat = model.predict(input_sample)
    sim_history.append(y_hat[0])
    

sim_history = [i[0] for i in sim_history]



############################
######################################
###############################################
######################################
############################



"""
    MC - simulation with uncertain data - 0.1
"""

# create new Dataset with random missing values

DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=0.1, random_seed=RANDOM_STATE) 
DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)

# fill missing values with kde values
DATAFRAME_UNCERTAIN = utils.kde_Imputer(DATAFRAME_MISS, kernel="gaussian", bandwidth=bw_method)

X_uncertain = DATAFRAME_UNCERTAIN.iloc[:, 0:-1]

sim_history_uncertain = []
for i in range(sim_length):
    
    input_sample = []
    for col in (X_uncertain):
        input_sample.append(random.choice(np.array(X_uncertain[col])))

    input_sample=np.array([input_sample])
    
    y_hat = model.predict(input_sample)
    sim_history_uncertain.append(y_hat[0])
    

sim_history_uncertain = [i[0] for i in sim_history_uncertain]



compare=[sim_history, sim_history_uncertain]
sns.kdeplot(compare, common_grid=True)
plt.title("Small uncertainty - 0.1")
plt.show()

############################################################################################




"""
    MC - sumulation with unceratin data 0.5
"""
# create new Dataset with random missing values

DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=0.5, random_seed=RANDOM_STATE) 
DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)

# fill missing values with kde values
DATAFRAME_UNCERTAIN = utils.kde_Imputer(DATAFRAME_MISS, kernel="gaussian", bandwidth=bw_method)

X_uncertain = DATAFRAME_UNCERTAIN.iloc[:, 0:-1]

sim_history_uncertain = []
for i in range(sim_length):
    
    input_sample = []
    for col in (X_uncertain):
        input_sample.append(random.choice(np.array(X_uncertain[col])))

    input_sample=np.array([input_sample])
    
    y_hat = model.predict(input_sample)
    sim_history_uncertain.append(y_hat[0])
    

sim_history_uncertain = [i[0] for i in sim_history_uncertain]



compare=[sim_history, sim_history_uncertain]
sns.kdeplot(compare, common_grid=True)
plt.title("Big uncertainty - 0.5")
plt.show()

"""
"""
sys.exit()
sys.exit()

sys.exit()
"""
"""


##########################################################################################################################
# remove data in original dataframe
##########################################################################################################################


# create new Dataset with random missing values
if use_normal_frame:
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=0.2, random_seed=RANDOM_STATE) 
    DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)
    
"""    
if use_probability_frame:
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME_PROBABILITY.iloc[:, :-1], miss_rate=0.2, random_seed=RANDOM_STATE) 
    DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME_PROBABILITY.iloc[:,-1], left_index=True, right_index=True)
"""


# fill missing values with kde values
DATAFRAME_UNCERTAIN = utils.kde_Imputer(DATAFRAME_MISS, kernel="gaussian", bandwidth=bw_method)

X_uncertain = DATAFRAME_UNCERTAIN.iloc[:, 0:-1]
y_uncertain = DATAFRAME_UNCERTAIN[column_names[-1]]



##########################################################################################################################
# sample Kernel Density Estimate over missing dataset
##########################################################################################################################






