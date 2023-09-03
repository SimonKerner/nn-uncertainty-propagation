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


use_normal_frame = True
use_probability_frame = False


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

"""
y_complete_hat = model.predict(X_complete).flatten()
y_complete_hat_labels = (y_complete_hat>0.5).astype("int32")
y_complete_joint = np.stack([y_complete_hat, y_complete_hat_labels], 1)
y_complete_joint = pd.DataFrame(y_complete_joint, columns=["sigmoid", "label"])

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



if get_true_prediction_metrics:
    
    utils.create_metrics(y_complete, y_complete_hat_labels)
    plt.show()

"""





##########################################################################################################################
# introduce missing data - aka. aleatoric uncertainty
##########################################################################################################################


# get KDE for each column
MISS_RATE=0.1


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
    
    
########



##################################################################################################################
# use of imputation methods for miss data
##########################################################################################################################

IMPUTE = False

if IMPUTE:pass
    #DATAFRAME_MISS = utils.kde_Imputer(DATAFRAME_MISS, kernel="gaussian", bandwidth="scott")

    # SimpleImputer imputer uses strategies --mean, median, most_frequent
    #simp_imp = SimpleImputer(strategy="median")
    #DATAFRAME_MISS = pd.DataFrame(simp_imp.fit_transform(DATAFRAME_MISS), columns=column_names)
 
    #knn_imp = KNNImputer(n_neighbors=5)
    #DATAFRAME_MISS = pd.DataFrame(knn_imp.fit_transform(DATAFRAME_MISS), columns=column_names)

##########################################################################################################################
# experiments -- row wise simulations (row wise induction - row wise random induction)
##########################################################################################################################


######
"""
    row wise experiment - of DATAFRAME
"""


get_spec_row = 1


simulation_visulizations = True
SIM_LENGTH = 2000





# statistics of experiments
exp_statistics = pd.DataFrame(data={"":""}, index=["Experiment 0"])


for row in range(len(DATAFRAME)):
    
    SIM_LENGTH = SIM_LENGTH
    
    # choose row to work with:
    if type(get_spec_row) == int:
        ROW = get_spec_row
    else:
        ROW = row
    
    
    if use_normal_frame:
        get_row = DATAFRAME.loc[ROW][:-1]
        get_row_outcome = DATAFRAME.loc[ROW][-1]
        
        get_row_miss = DATAFRAME_MISS.loc[ROW][:-1]
        get_row_miss_outcome = DATAFRAME_MISS.loc[ROW][-1]
        
        data_visualization_joint = pd.DataFrame(data={"DATAFRAME (TRUE)":get_row, 
                                                      "DATAFRAME_MISSING (FALSE)":get_row_miss})
        
        sns.histplot(data_visualization_joint, fill=True, bins=15)
        plt.show()
        sns.kdeplot(data=data_visualization_joint, fill=True, color='skyblue', alpha=0.3, common_grid=True, common_norm=False)  
        plt.show()
    
    
    
    
    if use_probability_frame:
        get_row = DATAFRAME_PROBABILITY.loc[ROW][:-1]
        get_row_outcome = DATAFRAME_PROBABILITY.loc[ROW][-1]
        
        get_row_miss = DATAFRAME_MISS.loc[ROW][:-1]
        get_row_miss_outcome = DATAFRAME_MISS.loc[ROW][-1]
        
        
        
        
        data_visualization_joint = pd.DataFrame(data={"DATAFRAME_PROBABILITY (TRUE)":get_row, 
                                                      "DATAFRAME_MISSING (FALSE)":get_row_miss})
        
        sns.histplot(data_visualization_joint, fill=True, bins=15)
        plt.show()
        sns.kdeplot(data=data_visualization_joint, fill=True, color='skyblue', alpha=0.3, common_grid=True, common_norm=False)  
        plt.show()
        
        
        sys.exit()
    
    ## take get KDE of row
    get_row_miss_kde = [stats.gaussian_kde(get_row_miss.dropna())]
    
    
    
    """
        following is completely random without imputation of nan values
    """
    exp_1_sim_length = SIM_LENGTH
    
    
    exp_1_sample_history = []
    for i in range(exp_1_sim_length):
        exp_1_get_row_miss_sample = get_row_miss_kde[0].resample(len(get_row))
    
        exp_1_sample_history.append(exp_1_get_row_miss_sample[0])
    
    exp_1_sample_history = pd.DataFrame(exp_1_sample_history)
    
    
    for i in range(exp_1_sim_length):
        sns.kdeplot(exp_1_sample_history.loc[i], common_norm=False, alpha=0.5, linewidth=0.1, common_grid=True, legend=False, color="grey")
    plt.title("(Input) exp_1_: Sample History")
    sns.kdeplot(data=data_visualization_joint, fill=False, color='skyblue', common_grid=True, common_norm=False)  
    plt.show()
    
    exp_1_sample_history_stats = exp_1_sample_history.describe()
    
    
    exp_1_predictions = model.predict(exp_1_sample_history).flatten()
    exp_1_predictions_labels = (exp_1_predictions>0.5).astype("int32")
    
    exp_1_predictions_joint = np.stack([exp_1_predictions, exp_1_predictions_labels], 1)
    exp_1_predictions_joint = pd.DataFrame(exp_1_predictions_joint, columns=["sigmoid", "label"])
    
    if simulation_visulizations:
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=exp_1_predictions, common_grid=True, cut=0)
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Density')
        plt.title('exp_1: Combined Output Density Plot')
        plt.tight_layout()
        plt.show()
        
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=exp_1_predictions_joint, x="sigmoid", hue="label", bins=10, stat="density", kde=True, kde_kws={"cut":0})
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Frequency')
        plt.title('exp_1: Combined Output Hist Plot')
        plt.tight_layout()
        plt.show()
        
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=exp_1_predictions_joint, x="sigmoid", hue="label", common_grid=True, cut=0)
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Density')
        plt.title('exp_1: Combined Output Density Plot')
        plt.tight_layout()
        plt.show()
    
    
    
    
    
    
    """
        following is completely random with imputation of nan values
    """
    
    exp_2_sim_length = SIM_LENGTH
    
    exp_2_sample_history = []
    for i in range(exp_2_sim_length):
        
        exp_2_get_row_miss = get_row_miss.copy()
        
        exp_2_get_row_miss_sample = get_row_miss_kde[0].resample(exp_2_get_row_miss.isna().sum())
    
        exp_2_sample = exp_2_get_row_miss_sample[0]
        
        
        exp_2_indices = exp_2_get_row_miss.isna()
        exp_2_indices = exp_2_indices[exp_2_indices].index
        #exp_2_indices = exp_2_indices.tolist()
    
    
        exp_2_use_to_induce = pd.Series(exp_2_sample, index=exp_2_indices)
        
        # induce nan with samples 
        exp_2_get_row_comp = exp_2_get_row_miss.fillna(exp_2_use_to_induce)
        exp_2_get_row_comp = np.array(exp_2_get_row_comp)
        
        exp_2_sample_history.append(exp_2_get_row_comp)
    
    exp_2_sample_history = pd.DataFrame(exp_2_sample_history)
    
    
    for i in range(exp_2_sim_length):
        sns.kdeplot(exp_2_sample_history.loc[i], common_norm=False, alpha=0.5, linewidth=0.1, common_grid=True, legend=False, color="grey")
    plt.title("(Input) exp_2_: Sample History")
    sns.kdeplot(data=data_visualization_joint, fill=False, color='skyblue', common_grid=True, common_norm=False)  
    plt.show()
    
    exp_2_sample_history_stats = exp_2_sample_history.describe()
    
    
    exp_2_predictions = model.predict(exp_2_sample_history).flatten()
    exp_2_predictions_labels = (exp_2_predictions>0.5).astype("int32")
    
    exp_2_predictions_joint = np.stack([exp_2_predictions, exp_2_predictions_labels], 1)
    exp_2_predictions_joint = pd.DataFrame(exp_2_predictions_joint, columns=["sigmoid", "label"])
    
    
    if simulation_visulizations:
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=exp_2_predictions, common_grid=True, cut=0)
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Density')
        plt.title('exp_2: Combined Output Density Plot')
        plt.tight_layout()
        plt.show()
        
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=exp_2_predictions_joint, x="sigmoid", hue="label", bins=10, stat="density", kde=True, kde_kws={"cut":0})
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Frequency')
        plt.title('exp_2: Combined Output Hist Plot')
        plt.tight_layout()
        plt.show()
        
        
        # visualize predictions
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=exp_2_predictions_joint, x="sigmoid", hue="label", common_grid=True, cut=0)
        plt.xlabel('Sigmoid Activations')
        plt.ylabel('Density')
        plt.title('exp_2: Combined Output Density Plot')
        plt.tight_layout()
        plt.show()
    
    
    
    
    """
        following are the results of collected simulation experiments
    """
    
    # comparisons between exp_1 and exp_2
    exp_0_experiment_collection = pd.DataFrame({"exp_1_random":exp_1_predictions, 
                                                "exp_2_induced":exp_2_predictions})
    exp_0_experiment_stats = exp_0_experiment_collection.describe()
    
    sns.kdeplot(exp_0_experiment_collection, common_norm=False, common_grid=True, legend=True)
    plt.title("Row: " + str(ROW) + " (Output) MC-Comparison of exp_1 and exp_2 predictions")
    plt.show()
    
    sns.violinplot(exp_0_experiment_collection)
    plt.title("Row: " + str(ROW) + " (Output) MC-Comparison of exp_1 and exp_2 predictions")
    plt.show()
    
    exp_0_experiment_collection.hist()
    plt.show()
    
    
    # exit simulation if specific row is chosen
    if type(get_spec_row) == int:
        break

    
    
    
    

















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






