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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KernelDensity

from scipy.stats import gaussian_kde




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

#choose working dataset: "australian" or "climate_simulation"
dataset = "climate_simulation" 


# set random state
RANDOM_STATE = None
np.random.seed(RANDOM_STATE)
np.random.RandomState(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# further settings
standardize_data = False
visiualize_data = False

# train or load model
train_model = False
load_model = True


get_prediction_metrics = False




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
    
    # set feature_specs
    continuous_features = ["Age", "Mean time at adresses", "Mean time with employers",
                           "Time with bank", "Monthly housing expense", "Savings account balance"]



# load data for climate modal simulation crashes dataset
if dataset == "climate_simulation":
    
    with open(os.path.join(dataset_path, dataset + ".dat"), 'rb') as DATAFRAME:
        DATAFRAME = pd.read_table(DATAFRAME, sep="\s+", engine='python', header = 0)
    
    # drop the first two elements of the dataset -> not relevant
    DATAFRAME = DATAFRAME.iloc[:, 2:]

    column_names = DATAFRAME.columns.to_list()
    
    continuous_features = column_names[:-1]




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
    """
    # plot Scatter matrix of dataframe
    sns.set_style("whitegrid")
    sns_plot = sns.PairGrid(DATAFRAME, diag_sharey=False, corner=False)
    sns_plot.map_lower(sns.kdeplot, levels=4, color = "red", fill=True, alpha=0.4)
    sns_plot.map_lower(sns.scatterplot, s=15)
    sns_plot.map_diag(sns.kdeplot, fill=True)
    plt.savefig(os.path.join(image_path, 'dataset_plot.png'))
    

    # Density and Histplot for exploration
    distplot_data = DATAFRAME.melt(var_name='column_names')
    distplot_plot = sns.displot(data=distplot_data, x='value', col='column_names', col_wrap=5, rug=False, kde=True, stat='density')
    plt.show()
    distplot_plot.savefig(os.path.join(image_path, 'distplot_plot.png'))
    plt.close()
    """
    
    # Plotting combined distribution using box plots
    DATAFRAME.boxplot(column=column_names, figsize=(12, 6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    
    # Plotting combined distribution using histograms
    DATAFRAME.hist(column=column_names, bins=15, figsize=(12, 10))
    plt.tight_layout()
    plt.show()
    
    
    # Visualizing correlation between variables using a heatmap
    corr_matrix = DATAFRAME.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
    plt.tight_layout()
    plt.show()
    
    
    # Create a KDE plot for each column
    for column in column_names:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(data=DATAFRAME[column], fill=True, color='skyblue', alpha=0.5)
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.title(f'KDE Plot of {column}')
        plt.tight_layout()
        plt.show()
        
        
    # Create a combined KDE plot
    plt.figure(figsize=(12, 6))
    for column in column_names:
        sns.kdeplot(data=DATAFRAME[column], fill=True, label=column)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Combined KDE Plot')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    combined_kde = None
    for column in column_names:
        # Calculate the KDE for the current column
        data = DATAFRAME[column]
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 1000)
        y = kde(x)
        
        if combined_kde is None:
            combined_kde = np.zeros_like(y)
        
        # Add the KDE estimates to the combined KDE
        combined_kde += y
    
    
    # Plot the combined KDE
    plt.plot(x, combined_kde, label="Combined KDE")
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Combined KDE Plot')
    plt.legend()
    plt.tight_layout()
    plt.show()
    



##########################################################################################################################
# Split data into X and y - optional scaling of data
##########################################################################################################################


X_complete = DATAFRAME.iloc[:, 0:-1]
y_complete = DATAFRAME[column_names[-1]]


X_complete_train, X_complete_test, y_complete_train,  y_complete_test = train_test_split(X_complete, y_complete, test_size=0.25, random_state=RANDOM_STATE)




##########################################################################################################################
# create standard vanilla feed forward neural network
##########################################################################################################################


if train_model:
    # layers of the network
    inputs = keras.Input(shape=(X_complete.shape[1],))
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
    model_history = model.fit(X_complete_train, y_complete_train, validation_data=[X_complete_test, y_complete_test], batch_size=15, epochs=50)
    
    
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


if get_prediction_metrics:
    
    utils.create_metrics(y_complete, y_complete_hat_labels)




##########################################################################################################################
# remove data in original dataframe
##########################################################################################################################


# create new Dataset with random missing values
DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :-1], miss_rate=0.9, random_seed=RANDOM_STATE) 
DATAFRAME_MISS = DATAFRAME_MISS.merge(DATAFRAME.iloc[:,-1], left_index=True, right_index=True)




##########################################################################################################################
# sample Kernel Density Estimate over missing dataset
##########################################################################################################################


"""
kernel_types = {
    "Sex": "exponential",
    "Age": "exponential",
    "Mean time at addresses": "exponential",
    "Home status": "gaussian",
    "Current occupation": "gaussian",
    "Current job status": "gaussian",
    "Mean time with employers": "exponential",
    "Other investments": "gaussian",
    "Bank account": "gaussian",
    "Time with bank": "gaussian",
    "Liability reference": "exponential",
    "Account reference": "exponential",
    "Monthly housing expense": "gaussian",
    "Savings account balance": "gaussian"
}
"""


sim_length = 20
sim_history = []    # with prdictions on the x-axis

for i in range(sim_length):
    
    DATAFRAME_UNCERTAIN = utils.kde_Imputer(DATAFRAME_MISS, bandwidth=0.1)
    
    X_uncertain = DATAFRAME_UNCERTAIN.drop("outcome", axis=1)
    y_uncertain = DATAFRAME_UNCERTAIN["outcome"]
    
    X_uncertain = X_uncertain.values
    y_uncertain = y_uncertain.values

    # Get the predicted probabilities for the imputed dataset
    y_uncertain_hat = model.predict(X_uncertain).flatten()

    sim_history.append(y_uncertain_hat)


# change sim_history to dataframe for better inspection
sim_history_df = pd.DataFrame(data=sim_history)
sim_history_df_describtion = sim_history_df.describe()
    



"""
# get first 10 predictions


# visualize some prediction with histograms
sim_history_df = sim_history_df.iloc[:, :5]
sim_history_df.boxplot(column=sim_history_df.columns.to_list(), figsize=(12, 6))
plt.tight_layout()
plt.show()
"""
sys.exit()





'''


from scipy.stats import norm, beta  # For fitting parametric distributions
from sklearn.neighbors import KernelDensity  # For non-parametric kernel density estimation



def generate_input_samples(data, column_names, continuous_features, plot_distributions=True):
    """
    Generate input samples according to the desired probability distributions.

    Args:
        data (pandas.DataFrame): The dataset containing the values for each attribute.
        column_names (list): List of column names in the dataset.
        continuous_features (list): List of column names for continuous features.

    Returns:
        numpy.ndarray: Generated input samples.
    """
    features = data.iloc[:, :-1].values  # Select all columns except the last as features

    # Calculate distribution parameters for continuous features
    distribution_params = {}
    for feature in continuous_features:
        mean = data[feature].mean()
        std = data[feature].std()
        distribution_params[feature] = {"mean": mean, "std": std}

    # Generate input samples for each feature
    input_samples = []
    for feature in column_names:
        if feature in continuous_features:
            # Continuous feature
            mean = distribution_params[feature]['mean']
            std = distribution_params[feature]['std']
            samples = np.random.normal(mean, std, size=len(features))
        else:
            # Nominal feature
            categories = data[feature].unique()
            samples = np.random.choice(categories, size=len(features), replace=True)
        input_samples.append(samples)
        
        # Plot the distribution if required
        if plot_distributions:
            plt.figure()
            plt.hist(samples, bins=30)
            plt.title(f"Distribution of {feature}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()

    # Combine all the generated samples
    input_samples = np.column_stack(input_samples)
    return input_samples






# Generate input samples
input_samples = generate_input_samples(DATAFRAME_UNCERTAIN, column_names, continuous_features)


# Calculate the output probabilities and output labels
output_probs = model.predict(input_samples)
output_probs = output_probs.flatten()  # Flatten to 1D array
output_labels = np.round(output_probs).astype(int)





# Visualize the output distributions for each label
for label in np.unique(output_labels):
    label_probs = output_probs[output_labels == label]

    # Fit Gaussian distribution to label probabilities
    mean, std = norm.fit(label_probs)

    # Visualize the distribution
    plt.figure()
    plt.hist(label_probs, bins=30, density=True, alpha=0.6, label=f'Label {label}')
    x = np.linspace(0, 1, 100)
    plt.plot(x, norm.pdf(x, mean, std), 'r-', label='Gaussian Fit')
    plt.xlabel('Output Probabilities')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Perform kernel density estimation
    kde = KernelDensity(bandwidth=0.05).fit(label_probs.reshape(-1, 1))
    x = np.linspace(0, 1, 100)
    log_density = kde.score_samples(x.reshape(-1, 1))
    density = np.exp(log_density)

    # Visualize the distribution
    plt.figure()
    plt.plot(x, density, label=f'Label {label}')
    plt.xlabel('Output Probabilities')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    
# Fit Gaussian distribution to output probabilities
mean, std = norm.fit(output_probs)

# Visualize the distribution
plt.figure()
plt.hist(output_probs, bins=30, density=True, alpha=0.6, color='g')
x = np.linspace(0, 1, 100)
plt.plot(x, norm.pdf(x, mean, std), 'r-', label='Gaussian Fit')
plt.xlabel('Output Probabilities')
plt.ylabel('Density')
plt.legend()
plt.show()

# Perform kernel density estimation
kde = KernelDensity(bandwidth=0.05).fit(output_probs.reshape(-1, 1))
x = np.linspace(0, 1, 100)
log_density = kde.score_samples(x.reshape(-1, 1))
density = np.exp(log_density)

# Visualize the distribution
plt.figure()
plt.plot(x, density, 'b-', label='Kernel Density Estimation')
plt.xlabel('Output Probabilities')
plt.ylabel('Density')
plt.legend()
plt.show()


'''