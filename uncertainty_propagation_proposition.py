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




DATAFRAME = DATAFRAME.iloc[:,15:].copy()
column_names = column_names[15:]


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


X_complete = DATAFRAME.iloc[:, 0:-1]
y_complete = DATAFRAME[column_names[-1]]


#X_complete_train, X_complete_test, y_complete_train,  y_complete_test = train_test_split(X_complete, y_complete, test_size=0.25, random_state=RANDOM_STATE)


##########################################################################################################################
# transform dataframe values into PDF values of corresponding DataFrame KDE values
##########################################################################################################################

# get KDE values of each column (X_complete) 
DATAFRAME_KDE = []
for i in range(len(column_names[:-1])):
    column_data = X_complete.iloc[:,i]
    column_kde = gaussian_kde(column_data)
    
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
        get_row_pdf.append(j.pdf(get_row_values[i])[0])
    get_row_pdf.append(get_row_label)
    
    DATAFRAME_PROBABILITY.append(get_row_pdf)
    
    #print(get_row_pdf)

DATAFRAME_PROBABILITY = np.array(DATAFRAME_PROBABILITY)
DATAFRAME_PROBABILITY = pd.DataFrame(data=DATAFRAME_PROBABILITY, columns=column_names)    


X_complete = DATAFRAME_PROBABILITY.iloc[:, 0:-1]
y_complete = DATAFRAME_PROBABILITY[column_names[-1]]


X_complete_train, X_complete_test, y_complete_train,  y_complete_test = train_test_split(X_complete, y_complete, test_size=0.25, random_state=RANDOM_STATE)




DATAFRAME_PROBABILITY.iloc[:, :-1].plot.kde(column=column_names[:-1], figsize=(12, 10))
plt.tight_layout()
plt.show()

sns.kdeplot(get_row)
plt.show()
sns.kdeplot(get_row_pdf)
plt.show()

sys.exit()

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
    model.save(os.path.join(model_path, dataset + "_model_probability"))




##########################################################################################################################
# load model without training
##########################################################################################################################


if load_model:
    model = keras.models.load_model(os.path.join(model_path, dataset + "_model_probability"))
    model.summary()




##########################################################################################################################
# some metric for the predictions
##########################################################################################################################


y_complete_hat = model.predict(X_complete)
y_complete_hat_labels = (y_complete_hat>0.5).astype("int32")


if get_true_prediction_metrics:
    
    utils.create_metrics(y_complete, y_complete_hat_labels)
    plt.show()



#DATAFRAME_PROBABILITY.iloc[:, :-1].plot.kde(column=column_names[:-1], figsize=(12, 10))
#plt.tight_layout()
#plt.show()


#sns.kdeplot(y_complete_hat)
#plt.show()



##########################################################################################################################
# remove data in original dataframe
##########################################################################################################################


# create new Dataset with random missing values
DATAFRAME_MISS = utils.add_missing_values(DATAFRAME_PROBABILITY.iloc[:, :-1], miss_rate=0.5, random_seed=RANDOM_STATE) 
DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME_PROBABILITY.iloc[:,-1], left_index=True, right_index=True)




##########################################################################################################################
# sample Kernel Density Estimate over missing dataset
##########################################################################################################################

"""
    # Monte Carlo Simulation with induced uncertainty
"""

# Monte Carlo Simulation Length
sim_length = 50

"""
DATAFRAME_MISS_LIST = []
for i in range(sim_length):
    DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=0.5, random_seed=RANDOM_STATE) 
    DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)
    
    DATAFRAME_MISS_LIST.append(DATAFRAME_MISS)
"""

# following => INPUT DATAFRAMES
sim_history_dataframes = []     # with imputed dataframes (including uncertainty)

# following => KDE VALUES FOR EACH DATAFRAME
sim_history_dataframe_kde = []

# following => OUTPUT PREDICTIONS as 
sim_history_predictions = []    # with prdictions on the x-axis

# following => OUTPUT PREDICTION as LABELS
sim_history_prediction_labels = []


# following => MC-Simulation
for i in range(sim_length):
    """
    DATAFRAME_MISS = DATAFRAME_MISS_LIST[i]
    """
    # kde imputer uses --from sklearn.neighbors import KernelDensity
    kernel = "gaussian"
    bandwidth = "scott"
    DATAFRAME_UNCERTAIN = utils.kde_Imputer(DATAFRAME_MISS, kernel=kernel, bandwidth=bandwidth)

    # SimpleImputer imputer uses strategies --mean, median, most_frequent
    #simp_imp = SimpleImputer(strategy="median")
    #DATAFRAME_UNCERTAIN = pd.DataFrame(simp_imp.fit_transform(DATAFRAME_MISS), columns=column_names)
 
    #knn_imp = KNNImputer(n_neighbors=5)
    #DATAFRAME_UNCERTAIN = pd.DataFrame(knn_imp.fit_transform(DATAFRAME_MISS), columns=column_names)
    
    X_uncertain = DATAFRAME_UNCERTAIN.iloc[:, 0:-1]
    y_uncertain = DATAFRAME_UNCERTAIN[column_names[-1]]

    # Get the predicted probabilities for the imputed dataset
    y_uncertain_hat = model.predict(X_uncertain).flatten()

    #kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    #kde.fit(X_uncertain)

    sim_history_dataframes.append(DATAFRAME_UNCERTAIN)
    #sim_history_dataframe_kde.append(kde)
    sim_history_predictions.append(y_uncertain_hat)
    #sim_history_prediction_labels.append((y_uncertain_hat>0.5).astype("int32"))


    """
    if i % 50 == 0:    
        
        # change sim_history_predictions to dataframe for better inspection
        sim_history_predictions_df = pd.DataFrame(data=sim_history_predictions)
        # summary statistics of simulation history
        sim_history_predictions_df_describtion = sim_history_predictions_df.describe(include="all")
        
        sim_history_predictions_mean = sim_history_predictions_df_describtion.loc["mean"]
        
        test = [y_complete_hat.flatten(), sim_history_predictions_mean]
        
        sns.kdeplot([test[0], test[1]], fill=True)
        plt.tight_layout()
        plt.show()
    """
    
    

# change sim_history_predictions to dataframe for better inspection
sim_history_predictions_df = pd.DataFrame(data=sim_history_predictions)
# summary statistics of simulation history
sim_history_predictions_df_describtion = sim_history_predictions_df.describe()

sim_history_predictions_mean = sim_history_predictions_df_describtion.loc["mean"]


"""
sim_history_predictions_minus_mean = sim_history_predictions_mean - sim_history_predictions_df_describtion.loc["std"]
sim_history_predictions_minus_mean = sim_history_predictions_minus_mean.rename("-std")
sim_history_predictions_plus_mean = sim_history_predictions_mean + sim_history_predictions_df_describtion.loc["std"]
sim_history_predictions_plus_mean = sim_history_predictions_plus_mean.rename("+std")
"""

#for i in range(sim_length):   
#    sns.kdeplot(sim_history_predictions_df.loc[i], alpha=.2, linewidth=0.4, color = "grey", common_grid=True, cumulative=False, thresh=0, cut=0)


print_results = [y_complete_hat.flatten(), sim_history_predictions_mean]

sns.kdeplot(print_results, common_grid=False, cumulative=False, thresh=0)
plt.tight_layout()
plt.show()


# plot histograms
sns.histplot(print_results[:-1], thresh=None, bins=10)
plt.tight_layout()
plt.show()

"""
plt.fill_between(np.arange())
ax_l1 = ax.get_lines()[1].get_data()
ax_l2 = ax.get_lines()[0].get_data()
ax.fill_between(ax_l1[0], ax_l1[1], ax_l2[1], alpha=0.2)
"""




"""
# used to calculate the differences between the True distrubution labels and mc labels
simulation_summary = []
for i in range(sim_length):
    differences = sim_history_prediction_labels[i] == y_complete_hat_labels.flatten()
    simulation_summary.append(differences)

simulation_summary = pd.DataFrame(simulation_summary)#.transpose()
simulation_summary_description = simulation_summary.describe(include="all") 
"""



