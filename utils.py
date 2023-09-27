# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:25:12 2023

@author: Selii
"""

from __future__ import division, print_function
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd

import numpy as np
import random

from sklearn.neighbors import KernelDensity

import seaborn as sns



##########################################################################################################################
# Define helper functions
##########################################################################################################################
    
    
def plot_history(history, smooth=False, log_scale=False, model_type=None):
    
    """
    Plots the training and validation metrics from the training history.
    
    Args:
        history (History): Training history object.
        smooth (bool): Whether to apply smoothing to the metrics curves. Defaults to False.
        log_scale (bool): Whether to use a logarithmic scale on the y-axis. Defaults to False.
    
    """
    

    if model_type == "binary":
        metric_labels = {
            'loss': 'Loss',
            'accuracy': 'Accuracy',
            'mse': 'Mean Squared Error',
            'mae': 'Mean Absolute Error',
            'mape': 'Mean Absolute Percentage Error',
            'msle': 'Mean Squared Logarithmic Error',
            'cosine_similarity': 'Cosine Similarity'
        }

    elif model_type == "multi":
        
        metric_labels = {
            'tf.nn.softmax_loss': 'tf.nn.softmax_loss',
            'tf.nn.softmax_accuracy': 'tf.nn.softmax_accuracy',
            'tf.math.sigmoid_loss': 'tf.math.sigmoid_loss',
            'tf.math.sigmoid_accuracy': 'tf.math.sigmoid_accuracy',
        }
        
    else:
        return None
    

    num_metrics = len(history.history)
    subplot_rows = (num_metrics + 1) // 2  # Adjust the subplot layout based on the number of metrics

    fig, axes = plt.subplots(subplot_rows, 2, figsize=(12, 4 * subplot_rows))

    idx = 0  # Keep track of the current subplot index

    for metric, values in history.history.items():
        if metric in metric_labels:
            row = idx // 2
            col = idx % 2

            ax = axes[row, col]

            if smooth:
                metric_values = smooth_curve(values)
            else:
                metric_values = values

            ax.plot(range(1, len(metric_values) + 1), metric_values, label='Training ' + metric_labels[metric])

            val_metric = 'val_' + metric
            if val_metric in history.history:
                val_metric_values = history.history[val_metric]
                ax.plot(range(1, len(val_metric_values) + 1), val_metric_values, label='Validation ' + metric_labels[metric])

            ax.set_title(metric_labels[metric])

            if log_scale:
                ax.set_yscale('log')

            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric_labels[metric])
            ax.legend()

            idx += 1  # Increment the subplot index

    # Remove empty subplots
    while idx < subplot_rows * 2:
        row = idx // 2
        col = idx % 2
        fig.delaxes(axes[row, col])
        idx += 1

    plt.tight_layout()
    plt.show()


def smooth_curve(points, factor=0.8):
    
    """
    Applies smoothing to a list of points using exponential moving average.
    
    Args:
        points (list): List of numeric values.
        factor (float): Smoothing factor. Defaults to 0.8.
    
    Returns:
        list: Smoothed list of points.
    
    """
    
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points




def create_metrics(y_true, predictions):
    
    """
    Calculates various evaluation metrics based on the true labels and predicted values.
    
    Args:
        y_true (array-like): True labels.
        predictions (array-like): Predicted values.
    
    Returns:
        dict: Dictionary containing the calculated metrics.

    """
    print()
    # Scores

        
    report = classification_report(y_true, predictions, digits=4, output_dict=True)
    
    print(classification_report(y_true, predictions, digits=4, output_dict=False))
        

    # Confusion Matrix
    cm = confusion_matrix(y_true, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()

    return report


def add_missing_values(df, miss_rate, delete_mode="static", random_seed=None):
    
    """
    Adds missing values to a DataFrame based on the specified missing rate.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        miss_rate (float): Proportion of missing values to add, between 0 and 1.
        random_seed (int): Random seed for reproducibility. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame with missing values added.
    
    """
    
    random.seed(random_seed)
    
    df_original = df.copy()
    df = df.iloc[:, :-1].copy()
    
    # -----> original technique for deleting values // deletes over the whole sample space by percentage
    # miss rate should be a float value between 0 and 1
    if delete_mode == "percentage":
    
        ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
        
        for row, col in random.sample(ix, int(round(miss_rate * len(ix)))):
            df.iat[row, col] = np.nan
        
        df_miss = df.merge(df_original.iloc[:,-1], left_index=True, right_index=True)
        
        return df_miss
 
    
     
    # -----> deletes specific amount of values in a row 
    # miss rate should be a static int value smaller then max(column length) and bigger then 0
    elif delete_mode == "static":
        
        for row in range(df.shape[0]):
    
            ix = [(row, col) for col in range(df.shape[1])]        
    
            for row, col in random.sample(ix, int(miss_rate)):
                df.iat[row, col] = np.nan
    
    
        df_miss = df.merge(df_original.iloc[:,-1], left_index=True, right_index=True)

        return df_miss
    
    else:
        print("No valid delete mode found!")

    
    
"""
"""         # test for the equality of kde samples from attributes with different missing rates of the different kde's
"""    
    
   ##-----------------------------------> just some tests
    
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
"""
"""
