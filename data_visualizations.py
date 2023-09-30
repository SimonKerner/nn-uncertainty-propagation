# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:50:40 2023

@author: Selii
"""


import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


image_path = os.path.join(os.getcwd(), 'images')


def remove_images(image_path=image_path):
    
    remove = [os.remove(os.path.join(image_path, 'simulation_rows_hist', fname)) for fname in os.listdir(os.path.join(image_path, 'simulation_rows_hist')) if fname.endswith('.png')]
    remove = [os.remove(os.path.join(image_path, 'simulation_rows_kde', fname)) for fname in os.listdir(os.path.join(image_path, 'simulation_rows_kde')) if fname.endswith('.png')]
    remove = [os.remove(os.path.join(image_path, fname)) for fname in os.listdir(image_path) if fname.endswith('.png')]


def plot_dataframe(dataframe, column_names, title):
    
    # Plotting combined distribution using histograms
    # hist is so that the plot shape is accasible later on
    
    hist = dataframe.hist(column=column_names, bins=10, figsize=(20, 12), 
                          density=False, sharey=False, sharex=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, title))
    plt.show()
    
    return hist


def roc_curves(y_original, prediction_metric, save_prefix):
    
    # @ https://ethen8181.github.io/machine-learning/model_selection/auc/auc.html

    title = 'Receiver Operator Characteristic'
    
    for key in prediction_metric:
        
        if type(key) == dict:
            get = key["y_hat"]
            key = "Original"
            
        elif type(key) == str:
            get = prediction_metric[key]["y_hat"]

        fpr, tpr, thresholds = roc_curve(y_original, get)

        # AUC score that summarizes the ROC curve
        roc_auc = roc_auc_score(y_original, get)
        
        plt.plot(fpr, tpr, lw = 2, label = key + ' ROC AUC: {:.2f}'.format(roc_auc))
        
    plt.plot([0, 1], [0, 1],
             linestyle = '--',
             color = (0.6, 0.6, 0.6),
             label = 'random guessing')
    plt.plot([0, 0, 1], [0, 1, 1],
             linestyle = ':',
             color = 'black', 
             label = 'perfect performance')
        
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title(title)
    
        
    plt.legend(loc = "lower right", fontsize=9)
    plt.tight_layout()  
    plt.savefig(os.path.join(image_path, save_prefix + "_" + title))
    plt.show()



def pre_recall_curve(y_original, prediction_metric, save_prefix):
    
    # @ https://ethen8181.github.io/machine-learning/model_selection/auc/auc.html
    
    title = 'Precision Recall Curve'
    
    for key in prediction_metric:
        
        if type(key) == dict:
            get = key["y_hat"]
            key = "Original"
            
        elif type(key) == str:
            get = prediction_metric[key]["y_hat"]
        
        precision, recall, thresholds = precision_recall_curve(
        y_original, get)
        
        # AUC score that summarizes the precision recall curve
        avg_precision = average_precision_score(y_original, get)
        
        label = key + ' PRC AUC: {:.2f}'.format(avg_precision)
        plt.plot(recall, precision, lw = 2, label = label)
        
    plt.xlabel('Recall')  
    plt.ylabel('Precision')  
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, save_prefix + "_" + title))
    plt.show()


    
def plot_binary_predictions(y_pred, y_labels, title):
    
    # visualize predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(data={"sigmoid" : y_pred, "label" : y_labels}, 
                 x="sigmoid", 
                 hue="label", 
                 bins=25, 
                 binrange=(0, 1), 
                 stat="count", 
                 kde=False, 
                 kde_kws={"cut":0})
    plt.xlabel('Sigmoid Activations')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, title))
    plt.show()
    
    
def plot_frame_comparison(data, title):
    
    # comparison of two dataframes
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, title))
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
    plt.savefig(os.path.join(image_path, "Model Evaluation"))
    plt.show()
    
    
    
    
def column_wise_kde_plot(data1, data2, name1, name2, column_names, miss_rate, sim_method, bw_method):
    
    for column in column_names:
        
        title = f'KDE Plot of Column: {column} - Miss-Rate: {miss_rate} - Method: {sim_method}'
        
        # KDE Plot of column without missing data
        plt.figure(figsize=(8, 4))
        sns.kdeplot(data={name1 : data1[column], 
                          name2 : data2[column]}, 
                    common_grid=True, 
                    bw_method=bw_method)
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, "single_kde_attributes", title))
        plt.show()
        
        
def combined_col_kde_plot(data1, data2, name1, name2, plt_shape, column_names, bw_method):
    
    plt.figure(0, figsize=(18, 10))
    column_count = 0
    
    for i in range(plt_shape.shape[0]):
        for j in range(plt_shape.shape[1]):
            
            if column_count >= len(column_names) or column_names[column_count] == column_names[-1]:
                continue
            
            ax = plt.subplot2grid((plt_shape.shape[0], plt_shape.shape[1]), (i,j))
            sns.kdeplot(data={name1 : data1[column_names[column_count]], 
                              name2 : data2[column_names[column_count]]}, 
                        common_grid=True, 
                        legend = False, 
                        bw_method=bw_method)
            plt.title(column_names[column_count])
            ax.plot()
            
            column_count += 1
        
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(image_path, "Combined KDE Plot"))
    plt.show()
    
    
    
def simulation_hist_plot(simulation_row_results, y_original, original_metrics, plotmode="autosearch", row=None):
    
    # find correct results in list
    if plotmode == "autosearch":
        
        for data in simulation_row_results:
            
            if data["0_Overall_Row_Data"]["0.1_row_id"] == row:
                
                #print("Printing hist-plot for row: " + str(data["0_Overall_Row_Data"]["0.1_row_id"]))
                simulation_row_results = data
                break
                
    elif plotmode == "specific":
        simulation_row_results = simulation_row_results
        
    else: 
        print("No plot has been found!")
        sys.exit()
    
    
    
    info = simulation_row_results["0_Overall_Row_Data"]
    uncertain_row =  simulation_row_results["Uncertain_Simulation"]
    original_row = simulation_row_results["Original_Simulation"]
    
    
    
    _fig, _axs = plt.subplots(2, 1, figsize=(17, 11))
    
    """
    Part 1: Plot_5.1.a: Histogam which shows the uncertain kde simulated row sigmoid results with hue 
    """
    # visualize predictions with hist plots
    sns.histplot(data=uncertain_row["y_hat"], 
             bins=max(25, int(info["0.5_Simulation_length"]/1000)), 
             binrange=(0, 1), 
             fill=True,
             alpha=0.3,
             stat="count", 
             kde=False, 
             kde_kws={"cut":0},
             ax=_axs[0]).set_title(label=f'Row: {info["0.1_row_id"]} Uncertain KDE Sim. Output Hist. Plot - Miss-Rate: {info["0.3_miss_rate"]} - Sim.-Length: {info["0.5_Simulation_length"]}')
    
    _axs[0].axvline(x=y_original[info["0.1_row_id"]], linewidth=8, linestyle = "-", color = "green", label="Original Label")
    _axs[0].axvline(x=original_metrics["y_hat"][info["0.1_row_id"]], linewidth=4, alpha=1, linestyle = "--", color = "red", label="Predicted Model Label")
    
    
    _axs[0].axvline(x=uncertain_row["mean"], linewidth=4, linestyle = "-.", color = "grey", label="Uncert. Mean Sim. Value") # uncert. simulation prediction mean
    
    # Max Density Vertical Lines
    #_axs[0].axvline(x=_uncertain_max_density_sigmoid, color="black", linestyle = "-.", linewidth=4, label="Uncert. KDE Max Density") 
    
    
    _legend1 = _axs[0].legend(["Original Label", "Predicted Model Label", "Uncert. Sim. Mean", "Uncertain Max Density"], loc="upper right", fontsize=12)
    
    _legend2 = _axs[0].legend(["Mean: " + str(np.round(uncertain_row["mean"], 3)), 
                               "Std: " + str(np.round(uncertain_row["std"], 3)),
                               #"MaxDensity: " + str(np.round(_uncertain_max_density_sigmoid, 3)),
                               "Density>0.5: " + str(np.round(uncertain_row["upper_probability"]*100, 3)) + "%",
                               "Density<0.5: " + str(np.round(uncertain_row["lower_probability"]*100, 3)) + "%"],
                              loc="upper left", edgecolor="white", fontsize=14, handlelength=0, handletextpad=0)
    
    _axs[0].add_artist(_legend1)
    _axs[0].add_artist(_legend2)
    


    """
    Part 2: Plot_5.1.b: Histogam which shows the original kde simulated row sigmoid results with hue 
    """
    # visualize predictions with hist plots
    sns.histplot(data=original_row["y_hat"],
             bins=max(25, int(info["0.5_Simulation_length"]/1000)), 
             binrange=(0, 1), 
             fill=True,
             alpha=0.3,
             stat="count", 
             kde=False, 
             kde_kws={"cut":0},
             ax=_axs[1]).set_title(label=f'Row: {info["0.1_row_id"]} Original KDE Sim. Output Hist. Plot - Miss-Rate: {info["0.3_miss_rate"]} - Sim.-Length: {info["0.5_Simulation_length"]}')
    
    _axs[1].axvline(x=y_original[info["0.1_row_id"]], linewidth=8, linestyle = "-", color = "green", label="Original Label")
    _axs[1].axvline(x=original_metrics["y_hat"][info["0.1_row_id"]], linewidth=4, alpha=1, linestyle = "--", color = "red", label="Predicted Model Label")
    
    # Simulation Mean 
    _axs[1].axvline(x=original_row["mean"], linewidth=4, linestyle = "-.", color = "grey", label="Orig. Mean Sim. Value") # orig. simulation prediction mean
    
    # Max Density Vertical Lines
    #_axs[1].axvline(x=_original_max_density_sigmoid, color="black", linestyle = "-.", linewidth=4, label="Orig. KDE Max Density")
    
    
    _legend1 = _axs[1].legend(["Original Label", "Predicted Model Label", "Orig. Sim. Mean", "Original Max Density"], loc="upper right", fontsize=12)
    
    
    _legend2 = _axs[1].legend(["Mean: " + str(np.round(original_row["mean"], 3)), 
                          "Std: " + str(np.round(original_row["std"], 3)),
                         #"MaxDensity: " + str(np.round(_original_max_density_sigmoid, 3)),
                         "Density>0.5: " + str(np.round(original_row["upper_probability"]*100, 3)) + "%",
                         "Density<0.5: " + str(np.round(original_row["lower_probability"]*100, 3))+ "%"],
                         loc="upper left", edgecolor="white", fontsize=14, handlelength=0, handletextpad=0)
    
    _axs[1].add_artist(_legend1)
    _axs[1].add_artist(_legend2)
    
    plt.savefig(os.path.join(image_path, "simulation_rows_hist", 'Simulaton Hist Plot - Row_' + str(row)))
    
    plt.show()
    
    
    
    
def simulation_kde_plot(x_axis, simulation_row_results, y_original, original_metrics, impute_metrics, plotmode="autosearch", row=None):    
    
    # find correct results in list
    if plotmode == "autosearch":
        
        for data in simulation_row_results:
            
            if data["0_Overall_Row_Data"]["0.1_row_id"] == row:
                
                #print("Printing kde-plot for row: " + str(data["0_Overall_Row_Data"]["0.1_row_id"]))
                simulation_row_results = data
                break
                
    elif plotmode == "specific":
        simulation_row_results = simulation_row_results
        
    else: 
        print("No plot has been found!")
        sys.exit()
    
    
    
    info = simulation_row_results["0_Overall_Row_Data"]
    uncertain_row =  simulation_row_results["Uncertain_Simulation"]
    original_row = simulation_row_results["Original_Simulation"]
    

    """
        Plot_combined_output: KDE PLOT of Uncerlying uncertainty
    """

    # KDE Distributions
    plt.plot(uncertain_row["pdf_x_axis"], uncertain_row["kde_pdf_y_axis"], 
             label="Uncertain KDE Distribution", linewidth=5,
             color = "black", alpha=0.3, linestyle = "--")
    
    plt.plot(original_row["pdf_x_axis"], original_row["kde_pdf_y_axis"], 
             label="Original KDE Distribution", linewidth=2,
             color = "black", alpha=0.55, linestyle = "--")


    plt.axvline(x=uncertain_row["mean"], linewidth=5, linestyle = "-.", color = "black", 
                alpha=0.3, ymin = 0, ymax = uncertain_row["kde_pdf_y_axis"][int(uncertain_row["mean"]*len(uncertain_row["pdf_x_axis"]))]-0.1, label="Uncert. Sim. Mean") 
    
    plt.axvline(x=original_row["mean"], linewidth=2, linestyle = "-", color = "black", 
                alpha=0.7, ymin = 0, ymax = original_row["kde_pdf_y_axis"][int(original_row["mean"]*len(original_row["pdf_x_axis"]))]-0.1, label="Orig. Sim. Mean") 
  

    ## Max Density Vertical Lines
    #plt.vlines(x=_uncertain_max_density_sigmoid, 
    #           ymin = 0, ymax = max(_uncertain_kde_density_peak_pdf), 
    #           color="grey", linestyle = "-.", linewidth=0.9, 
    #           label="Uncert. KDE Max Density") 
    
    #plt.vlines(x=_original_max_density_sigmoid, 
    #           ymin = 0, ymax = max(_original_kde_density_peak_pdf), 
    #           color="black", linestyle = "-.", linewidth=0.9, 
    #           label="Orig. KDE Max Density")
    
    
    # Predicted and Original Label
    plt.axvline(x=y_original[info["0.1_row_id"]], 
                linewidth=5, linestyle = "-", 
                color = "green", label="Original Label")
    
    
    plt.axvline(x=original_metrics["y_hat"][info["0.1_row_id"]], 
                linewidth=3, alpha=1, linestyle = "--", 
                color = "red", label="Predicted Model Label")
    
    # Impute Vertical Line
    if True: # _IMPUTE:
        plt.axvline(x=impute_metrics["ITER_IMPUTE"]["y_hat"][info["0.1_row_id"]], 
                    linewidth=5, alpha=0.4, linestyle = "-", color = "blue", 
                    label="Impute Prediction" + " (" + "IterativeImputer" + ")")
    
    
    plt.title(f'Row: {info["0.1_row_id"]} Underlying Uncertainty of the Simulation - Miss-Rate: {info["0.3_miss_rate"]} - Sim.-Length: {info["0.5_Simulation_length"]}')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.xlabel('Sigmoid Activation')
    plt.ylabel('Density')
    plt.ylim([0, max(max(uncertain_row["kde_pdf_y_axis"]), max(original_row["kde_pdf_y_axis"])) + 0.1])
    
    plt.savefig(os.path.join(image_path, "simulation_rows_kde", 'Simulaton Hist Plot - Row_' + str(row)))
    
    plt.show()
    
    