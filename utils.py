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





'''Functions for drawing contours of Dirichlet distributions.'''

# Author: Thomas Boggs



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_AREA = 0.5 * 1 * 0.75**0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.

    Arguments:

        `xy`: A length-2 sequence containing the x and y value.
    '''
    coords = np.array([tri_area(xy, p) for p in _pairs]) / _AREA
    return np.clip(coords, tol, 1.0 - tol)

class Dirichlet(object):
    def __init__(self, alpha):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     np.multiply.reduce([gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * np.multiply.reduce([xx ** (aa - 1)
                                                for (xx, aa)in zip(x, self._alpha)])
    def sample(self, N):
        '''Generates a random sample of size `N`.'''
        return np.random.dirichlet(self._alpha, N)

def draw_pdf_contours(dist, border=False, nlevels=200, subdiv=8, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).

    Arguments:

        `dist`: A distribution instance with a `pdf` method.

        `border` (bool): If True, the simplex border is drawn.

        `nlevels` (int): Number of contours to draw.

        `subdiv` (int): Number of recursive mesh subdivisions to create.

        kwargs: Keyword args passed on to `plt.triplot`.
    '''
    from matplotlib import ticker, cm
    import math

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.triplot(_triangle, linewidth=1)

def plot_points(X, barycentric=True, border=True, **kwargs):
    '''Plots a set of points in the simplex.

    Arguments:

        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.

        `barycentric` (bool): Indicates if `X` is in barycentric coords.

        `border` (bool): If True, the simplex border is drawn.

        kwargs: Keyword args passed on to `plt.plot`.
    '''
    if barycentric is True:
        X = X.dot(_corners)
    plt.plot(X[:, 0], X[:, 1], 'k.', ms=1, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.triplot(_triangle, linewidth=1)

"""
def plot_dirichlet(alphas):
    
    alphas = [alphas]
    
    f = plt.figure(figsize=(8, 6))
    
    for (i, alpha) in enumerate(alphas):
        plt.subplot(2, len(alphas), i + 1)
        dist = Dirichlet(alpha)
        draw_pdf_contours(dist)
        title = r'$\alpha$ = (%.3f, %.3f, %.3f)' % tuple(alpha)
        plt.title(title, fontdict={'fontsize': 8})
        plt.subplot(2, len(alphas), i + 1 + len(alphas))
        plot_points(dist.sample(1000))
    #plt.savefig('dirichlet_plots.png')
    print('Wrote plots to "dirichlet_plots.png".')
    plt.show()
    
"""
    
    
    
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
from math import gamma
from operator import mul
from functools import reduce
sns.set(style='white', font_scale=1.2, font='consolas')

def plot_mesh(corners):
    """Subdivide the triangle into a triangular mesh and plot the original and subdivided triangles."""
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)
    
    plt.figure(figsize=(6, 4))
    for i, mesh in enumerate((triangle, trimesh)):
        plt.subplot(1, 2, i+1)
        plt.triplot(mesh)
        plt.axis('off')
        plt.axis('equal')    
        
        
class Dirichlet:
    """Define the Dirichlet distribution with vector parameter alpha."""
    def __init__(self, alpha):
        
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / reduce(mul, [gamma(a) for a in self._alpha])
    
    def pdf(self, x):
        """Returns pdf value for `x`. """
        return self._coef * reduce(mul, [xx ** (aa-1) for (xx, aa) in zip(x, self._alpha)])
        
        
class PlotDirichlet:
    """
    Plot the Dirichlet distribution as a contour plot on a 2-Simplex.
    """
    def __init__(self, corners):
        self._corners = corners
        self._triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
        # Midpoints of triangle sides opposite of each corner
        self._midpoints = [(corners[(i+1) % 3] + corners[(i+2) % 3]) / 2.0 for i in range(3)]
        
    def xy2bc(self, xy, tol=1.e-3):
        """Map the x-y coordinates of the mesh vertices to the simplex coordinate space (aka barycentric coordinates).
        Here we use a simple method that uses vector algebra. For some values of alpha, calculation of the Dirichlet pdf 
        can become numerically unstable at the boundaries of the simplex so our conversion function will take an optional 
        tolerance that will avoid barycentric coordinate values directly on the simplex boundary.        
        """
        s = [(self._corners[i] - self._midpoints[i]).dot(xy - self._midpoints[i]) / 0.75 for i in range(3)]
        return np.clip(s, tol, 1.0-tol)
        
    def draw_pdf_contours(self, ax, dist, label=None, nlevels=200, subdiv=8, **kwargs):
        """Draw pdf contours for a Dirichlet distribution"""
        # Subdivide the triangle into a triangular mesh
        refiner = tri.UniformTriRefiner(self._triangle)
        trimesh = refiner.refine_triangulation(subdiv=subdiv)
        
        # convert to barycentric coordinates and compute probabilities of the given distribution 
        pvals = [dist.pdf(self.xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    
        ax.tricontourf(trimesh, pvals, nlevels, **kwargs)
        #plt.axis('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.75**0.5)
        ax.set_title(str(label))
        ax.axis('off') 
        return ax

"""       
if __name__ == '__main__':
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    plot_dirichlet = PlotDirichlet(corners)
   
    f, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes[0, 0]
    alpha = (0.85, 5.85, 0.85)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[0, 1]
    alpha = (1, 1, 1)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[0, 2]
    alpha = (5, 5, 5)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[1, 0]
    alpha = (1, 2, 3)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[1, 1]
    alpha = (2, 5, 10)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)

    ax = axes[1, 2]
    alpha = (50, 50, 50)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)
"""

def plot_dirichlet_2(alpha):

    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    plot_dirichlet = PlotDirichlet(corners)
    f, axes = plt.subplots(1, 1, figsize=(14, 8))
    ax = axes

    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, alpha)
    
    
    
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
