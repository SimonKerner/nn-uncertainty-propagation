# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:25:12 2023

@author: Selii
"""


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
    
    
def plot_history(history, smooth=False, log_scale=False):
    
    """
    Plots the training and validation metrics from the training history.
    
    Args:
        history (History): Training history object.
        smooth (bool): Whether to apply smoothing to the metrics curves. Defaults to False.
        log_scale (bool): Whether to use a logarithmic scale on the y-axis. Defaults to False.
    
    """
    
    metric_labels = {
        'loss': 'Loss',
        'accuracy': 'Accuracy',
        'mse': 'Mean Squared Error',
        'mae': 'Mean Absolute Error',
        'mape': 'Mean Absolute Percentage Error',
        'msle': 'Mean Squared Logarithmic Error',
        'cosine_similarity': 'Cosine Similarity'
    }

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
    
    # Scores
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    report = classification_report(y_true, predictions, target_names=['Rejected', 'Accepted'])

    # Confusion Matrix
    cm = confusion_matrix(y_true, predictions)
    display_labels = ['Rejected/Failure', 'Accepted/Success']
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    display.plot()

    # Additional Metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
    }

    # Print Additional Metrics
    for metric, value in metrics.items():
        print(f'{metric.capitalize()}:')
        print(value)
        print()

    return metrics




def add_missing_values(df, miss_rate, random_seed=None):
    
    """
    Adds missing values to a DataFrame based on the specified missing rate.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        miss_rate (float): Proportion of missing values to add, between 0 and 1.
        random_seed (int): Random seed for reproducibility. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame with missing values added.
    
    """
    
    df = df.copy()
    
    ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    
    random.seed(random_seed)
    
    for row, col in random.sample(ix, int(round(miss_rate * len(ix)))):
        df.iat[row, col] = np.nan
        
    return df





def kde_Imputer(dataframe, kernel="gaussian", bandwidth="scott", random_state=None, print_info=False):
    
    """
    Imputes missing values in a DataFrame using Kernel Density Estimation (KDE).
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        kernel (str or dict): Kernel type or a mapping of column names to kernel types.
                              Defaults to "gaussian".
        bandwidth (str, float): Bandwidth estimation method or a fixed bandwidth value. Defaults to "scott".
        random_state (int): Random seed for reproducibility. Defaults to None.
        print_info (bool): Whether to print information about the KDE parameters. Defaults to True.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed using KDE.
    
    Raises:
        ValueError: If invalid arguments are provided.
    
    """
    
    # Input validation
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The 'dataframe' argument must be a valid pandas DataFrame.")
    
    valid_kernels = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    if isinstance(kernel, str) and kernel not in valid_kernels:
        raise ValueError(f"Invalid kernel type '{kernel}'. Valid options are: {', '.join(valid_kernels)}.")
    
    if isinstance(kernel, dict):
        for col, k in kernel.items():
            if k not in valid_kernels:
                raise ValueError(f"Invalid kernel type '{k}' for column '{col}'. Valid options are: {', '.join(valid_kernels)}.")
    
    if not isinstance(bandwidth, (str, float, int)):
        raise ValueError("The 'bandwidth' argument must be a string or a float.")
    
    if isinstance(kernel, str):
        kernel_mapping = {column: kernel for column in dataframe.columns}
    elif isinstance(kernel, dict):
        kernel_mapping = kernel


    imputed_df = dataframe.copy()
    
    for column in dataframe.columns:
        values = dataframe[column].dropna().values.reshape(-1, 1)
        
        kde = KernelDensity(kernel=kernel_mapping.get(column), bandwidth=bandwidth,)
        kde.fit(values)
        
        if print_info:
            # Print the KernelDensity parameters for the current column
            print(f"Column: {column}")
            print(f"Kernel: {kde.kernel}")
            print(f"Bandwidth: {kde.bandwidth}\n")

            # KDE Plot of column without missing data
            plt.figure(figsize=(8, 4))
            sns.kdeplot(data=values, fill=True, color='skyblue', alpha=0.5)
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.title(f'KDE Plot of {column}')
            plt.tight_layout()
            plt.show()

        missing_values = imputed_df[column].isnull()
        num_missing = missing_values.sum()

        if num_missing > 0:
            kde_samples = kde.sample(num_missing, random_state=random_state)    
            """
            # Limit samples to the range of 0 and 1 for binary columns
            if np.array_equal(np.unique(values), [0., 1.]):
                
                kde_samples = np.random.choice(np.clip(kde_samples, a_min=0., a_max=1.).flatten(), num_missing, replace=True)
            
            # if original columns do not have negative values, clip at lower limit
            elif (values<0).sum() == 0:
    
                kde_samples = np.random.choice(np.clip(kde_samples, a_min=0., a_max=None).flatten(), num_missing, replace=True)
            """         
            kde_samples = np.random.choice(kde_samples.reshape(-1), num_missing, replace=True)  # Reshape to match missing values
            imputed_df.loc[missing_values, column] = kde_samples

    return imputed_df