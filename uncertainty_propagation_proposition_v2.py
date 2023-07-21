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


"""




##########################################################################################################################
# introduce missing data - aka. aleatoric uncertainty
##########################################################################################################################


# get KDE for each column

if use_normal_frame:
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=0.5, random_seed=RANDOM_STATE) 
    DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)
    
    
if use_probability_frame:
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME_PROBABILITY.iloc[:, :-1], miss_rate=0.5, random_seed=RANDOM_STATE) 
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
# experiments -- row wise simulations (row wise induction - row wise random induction)
##########################################################################################################################


######
"""
    row wise experiment - just for first row of DATAFRAME
"""

if use_normal_frame:
    first_row = DATAFRAME.loc[0][:-1]
    first_row_outcome = DATAFRAME.loc[0][-1]
    
    first_row_miss = DATAFRAME_MISS.loc[0][:-1]
    first_row_miss_outcome = DATAFRAME_MISS.loc[0][-1]
    
    data_visualization_joint = pd.DataFrame(data={"DATAFRAME (TRUE)":first_row, 
                                                  "DATAFRAME_MISSING (FALSE)":first_row_miss})
    
    sns.histplot(data_visualization_joint, fill=True, bins=15)
    plt.show()
    sns.kdeplot(data=data_visualization_joint, fill=True, color='skyblue', alpha=0.3, common_grid=True, common_norm=False)  
    plt.show()




if use_probability_frame:
    first_row = DATAFRAME_PROBABILITY.loc[0][:-1]
    first_row_outcome = DATAFRAME_PROBABILITY.loc[0][-1]
    
    first_row_miss = DATAFRAME_MISS.loc[0][:-1]
    first_row_miss_outcome = DATAFRAME_MISS.loc[0][-1]
    
    
    
    
    data_visualization_joint = pd.DataFrame(data={"DATAFRAME_PROBABILITY (TRUE)":first_row, 
                                                  "DATAFRAME_MISSING (FALSE)":first_row_miss})
    
    sns.histplot(data_visualization_joint, fill=True, bins=15)
    plt.show()
    sns.kdeplot(data=data_visualization_joint, fill=True, color='skyblue', alpha=0.3, common_grid=True, common_norm=False)  
    plt.show()



## take get KDE of row
first_row_miss_kde = [stats.gaussian_kde(first_row_miss.dropna())]



"""
    following is completely random without imputation fo nan values
"""
exp_1_sim_length = 500


exp_1_sample_history = []
for i in range(exp_1_sim_length):
    exp_1_first_row_miss_sample = first_row_miss_kde[0].resample(len(first_row))

    exp_1_sample_history.append(exp_1_first_row_miss_sample[0])

exp_1_sample_history = pd.DataFrame(exp_1_sample_history)


for i in range(exp_1_sim_length):
    sns.kdeplot(exp_1_sample_history.loc[i], common_norm=False, alpha=0.5, linewidth=0.1, common_grid=True, legend=False, color="grey")
plt.title("(Input) exp_1_: Sample History")
sns.kdeplot(data=data_visualization_joint, fill=False, color='skyblue', common_grid=True, common_norm=False)  
plt.show()

exp_1_sample_history_stats = exp_1_sample_history.describe()


exp_1_predictions = model.predict(exp_1_sample_history).flatten()




"""
    following is completely random with imputation of nan values
"""

exp_2_sim_length = 500

exp_2_sample_history = []
for i in range(exp_2_sim_length):
    
    exp_2_first_row_miss = first_row_miss.copy()
    
    exp_2_first_row_miss_sample = first_row_miss_kde[0].resample(exp_2_first_row_miss.isna().sum())

    exp_2_sample = exp_2_first_row_miss_sample[0]
    
    
    exp_2_indices = exp_2_first_row_miss.isna()
    exp_2_indices = exp_2_indices[exp_2_indices].index
    #exp_2_indices = exp_2_indices.tolist()


    exp_2_use_to_induce = pd.Series(exp_2_sample, index=exp_2_indices)
    
    # induce nan with samples 
    exp_2_first_row_comp = exp_2_first_row_miss.fillna(exp_2_use_to_induce)
    exp_2_first_row_comp = np.array(exp_2_first_row_comp)
    
    exp_2_sample_history.append(exp_2_first_row_comp)

exp_2_sample_history = pd.DataFrame(exp_2_sample_history)


for i in range(exp_2_sim_length):
    sns.kdeplot(exp_2_sample_history.loc[i], common_norm=False, alpha=0.5, linewidth=0.1, common_grid=True, legend=False, color="grey")
plt.title("(Input) exp_2_: Sample History")
sns.kdeplot(data=data_visualization_joint, fill=False, color='skyblue', common_grid=True, common_norm=False)  
plt.show()

exp_2_sample_history_stats = exp_2_sample_history.describe()


exp_2_predictions = model.predict(exp_2_sample_history).flatten()





"""
    following are the results of collected simulation experiments
"""

# comparisons between exp_1 and exp_2
exp_0_experiment_collection = pd.DataFrame({"exp_1_random":exp_1_predictions, 
                                            "exp_2_induced":exp_2_predictions})
exp_0_experiment_stats = exp_0_experiment_collection.describe()

sns.kdeplot(exp_0_experiment_collection, common_norm=False, common_grid=True, legend=True)
plt.title("(Output) MC-Comparison of exp_1 and exp_2 predictions")
plt.show()




"""
    column wise imputation of missing values
"""






'''
def kde_Imputer(dataframe, kernel="gaussian", bandwidth="scott", random_state=None, print_info=False):

    imputed_df = dataframe.copy()
    
    for column in dataframe.columns:
        values = dataframe[column].dropna().values
        
        kde = stats.gaussian_kde(values)   
        kde_vis = [kde]
        
        if print_info:
            # Print the KernelDensity parameters for the current column
            #print(f"Column: {column}")
            #print(f"Kernel: {kde.kernel}")
            #print(f"Bandwidth: {kde.bandwidth}\n")

            # KDE Plot of column without missing data
            plt.figure(figsize=(8, 4))
            
            test = DATAFRAME_PROBABILITY[column]
            
            data_visualization_joint = pd.DataFrame(data={"DATAFRAME_PROBABILITY (TRUE)":test, "DATAFRAME_UNDERLYING (ESTIMATED)":dataframe[column]})
            
            sns.kdeplot(data=data_visualization_joint, fill=True, color='skyblue', alpha=0.5, common_grid=True, common_norm=False)      
     
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.title(f'KDE Plot of Column: {column}')
            plt.tight_layout()
            plt.show()


        missing_values = imputed_df[column].isnull()
        num_missing = missing_values.sum()

        if num_missing > 0:
            kde_samples = kde.sample(num_missing, random_state=random_state)    
            
            # Limit samples to the range of 0 and 1 for binary columns
            if np.array_equal(np.unique(values), [0., 1.]):
                
                kde_samples = np.random.choice(np.clip(kde_samples, a_min=0., a_max=1.).flatten(), num_missing, replace=True)
            
            # if original columns do not have negative values, clip at lower limit
            elif (values<0).sum() == 0:
    
                kde_samples = np.random.choice(np.clip(kde_samples, a_min=0., a_max=None).flatten(), num_missing, replace=True)
                      
            kde_samples = np.random.choice(kde_samples.reshape(-1), num_missing, replace=True)  # Reshape to match missing values
            imputed_df.loc[missing_values, column] = kde_samples

    return imputed_df




kde = kde_Imputer(DATAFRAME_MISS, print_info=True)
'''





















sys.exit()
#first_row_y_hat = model.predict(np.reshape(first_row.values, (-1,15)))
#first_row_miss_y_hat = model.predict(np.reshape(first_row_miss.values, (-1,15)))



DATAFRAME_IMPUTE = utils.kde_Imputer(DATAFRAME_MISS, kernel="gaussian", bandwidth=0.5, random_state=None, print_info=False)


first_row_impute = DATAFRAME_IMPUTE.loc[0][:-1]
first_row_impute_outcome = DATAFRAME_IMPUTE.loc[0][-1]

first_row_comp_joint = np.stack([first_row, first_row_miss, first_row_impute], 1)
first_row_comp_joint = pd.DataFrame(first_row_comp_joint, columns=["first_row", "first_row_miss", "first_row_impute"])


sns.kdeplot(first_row_comp_joint, fill=False, bw_adjust=1, common_grid=True)
plt.show()





"""
DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=0.5, random_seed=RANDOM_STATE) 
DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)


first_row = DATAFRAME.loc[0][:-1]
first_row_outcome = DATAFRAME.loc[0][-1]

first_row_miss = DATAFRAME_MISS.loc[0][:-1]
first_row_miss_outcome = DATAFRAME_MISS.loc[0][-1]

first_row_joint = np.stack([first_row, first_row_miss], 1)
first_row_joint = pd.DataFrame(first_row_joint, columns=["first_row", "first_row_miss"])


sns.kdeplot(first_row_joint, fill=True, bw_adjust=1)
plt.show()
"""










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






