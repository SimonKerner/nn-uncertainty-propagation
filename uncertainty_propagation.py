# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:24:32 2023

@author: Selii
"""

import os
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
# load datasets
##########################################################################################################################


# set path to dataframes
datasets = os.path.join(os.getcwd(), 'datasets')

    
#with open(os.path.join(datasets, "data_australian.pkl"), 'rb') as file:
#    data_australian = pickle.load(file)
    
    
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
#data_australian.columns = column_names
    



##########################################################################################################################
# set Variables to work with + standardization of values for better performance
##########################################################################################################################
    

# choose dataframe to work with
DATAFRAME = data 

# set random state
RANDOM_STATE = 24
np.random.seed(RANDOM_STATE)
np.random.RandomState(RANDOM_STATE)
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
    
    
    # Plotting combined distribution using box plots
    DATAFRAME.boxplot(column=column_names, figsize=(12, 6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    
    # Plotting combined distribution using histograms
    DATAFRAME.hist(column=column_names, bins=10, figsize=(12, 10))
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
    
    
    """
    # Calculate the entropy for each feature
    entropies = []
    target_column = "0 - Reject / 1 - Accept"
    target_counts = DATAFRAME[target_column].value_counts()
    total_samples = target_counts.sum()

    for column in column_names:
        feature_entropy = 0
        feature_counts = DATAFRAME.groupby([column, target_column]).size()
        unique_values = DATAFRAME[column].unique()

        for value in unique_values:
            value_counts = feature_counts[value]
            value_total = value_counts.sum()
            value_probabilities = value_counts / value_total
            value_entropy = -np.sum(value_probabilities * np.log2(value_probabilities))
            feature_entropy += (value_total / total_samples) * value_entropy

        entropies.append(feature_entropy)

    # Find the index of the feature with the highest entropy
    most_uncertain_feature_index = np.argmax(entropies)
    most_uncertain_feature = column_names[most_uncertain_feature_index]

    print("Most Uncertain Feature:", most_uncertain_feature)

    # Plot the entropies
    plt.figure(figsize=(6, 10))
    plt.barh(column_names, entropies)
    plt.xlabel("Entropy")
    plt.ylabel("Features")
    plt.title("Entropy of Features")
    plt.show()
    """




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


metrics = False

if metrics:
    
    utils.create_metrics(y, standard_predictions_scalar)




##########################################################################################################################
# add missing data
##########################################################################################################################


# create new Dataset with random missing values
DATAFRAME_MISS = utils.add_missing_values(DATAFRAME.iloc[:, :14], 0.1, RANDOM_STATE) 
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


Imputed_df = utils.kde_Imputer(DATAFRAME_MISS, bandwidth=0.1)

#neg = (Imputed_df<0).sum()





"""
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the complete dataset and the KDE imputed dataset
complete_data = DATAFRAME
imputed_data = Imputed_df

# Split the data into features (X) and target variable (y)
X_complete = complete_data.drop("0 - Reject / 1 - Accept", axis=1)
y_complete = complete_data["0 - Reject / 1 - Accept"]

X_imputed = imputed_data.drop("0 - Reject / 1 - Accept", axis=1)
y_imputed = imputed_data["0 - Reject / 1 - Accept"]

# Convert the data to numpy arrays
X_complete = X_complete.values
y_complete = y_complete.values

X_imputed = X_imputed.values
y_imputed = y_imputed.values

# Define the architecture of the sequential feed-forward network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_complete.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with an appropriate loss function and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the complete dataset
model.fit(X_complete, y_complete, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the imputed dataset
loss, accuracy = model.evaluate(X_imputed, y_imputed, verbose=1)
print("Accuracy on the imputed dataset:", accuracy)

# Get the predicted probabilities for the complete dataset
predicted_probabilities_complete = model.predict(X_complete)

# Get the predicted probabilities for the imputed dataset
predicted_probabilities = model.predict(X_imputed)

# Calculate the uncertainty measures (e.g., variance and entropy)
uncertainty_variance = np.var(predicted_probabilities)
uncertainty_entropy = -np.sum(predicted_probabilities * np.log2(predicted_probabilities))

print("Uncertainty variance:", uncertainty_variance)
print("Uncertainty entropy:", uncertainty_entropy)


# Create a histogram of predicted probabilities for the complete dataset
plt.figure(figsize=(8, 6))
plt.hist(predicted_probabilities_complete, bins=20, alpha=0.5, label='Complete Dataset')

# Create a histogram of predicted probabilities for the imputed dataset
plt.hist(predicted_probabilities, bins=20, alpha=0.5, label='KDE Imputed Dataset')

plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Output Distribution Comparison')
plt.legend()
plt.show()




# Create a KDE plot for the predicted probabilities of the complete dataset
sns.kdeplot(predicted_probabilities_complete.flatten(), label='Complete Dataset')

# Create a KDE plot for the predicted probabilities of the imputed dataset
sns.kdeplot(predicted_probabilities.flatten(), label='KDE Imputed Dataset')

plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Output Distribution Comparison')
plt.legend()
plt.show()
"""


from scipy.stats import norm, beta  # For fitting parametric distributions
from sklearn.neighbors import KernelDensity  # For non-parametric kernel density estimation
import matplotlib.pyplot as plt  # For visualization


def generate_input_samples(data, column_names, continuous_features, nominal_features, plot_distributions=True):
    """
    Generate input samples according to the desired probability distributions.

    Args:
        data (pandas.DataFrame): The dataset containing the values for each attribute.
        column_names (list): List of column names in the dataset.
        continuous_features (list): List of column names for continuous features.
        nominal_features (list): List of column names for nominal features.

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



# Load your dataset into a pandas DataFrame
data = Imputed_df

# Define the necessary parameters
column_names = ["Sex", "Age", "Mean time at adresses", "Home status", "Current occupation",
                "Current job status", "Mean time with employers", "Other investments",
                "Bank account", "Time with bank", "Liability reference", "Account reference",
                "Monthly housing expense", "Savings account balance"]

continuous_features = ["Age", "Mean time at adresses", "Mean time with employers",
                       "Time with bank", "Monthly housing expense", "Savings account balance"]

nominal_features = ["Sex", "Home status", "Current occupation", "Current job status",
                    "Other investments", "Bank account", "Liability reference", "Account reference"]

# Generate input samples
input_samples = generate_input_samples(data, column_names, continuous_features, nominal_features)


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


