# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:25:12 2023

@author: Selii
"""

from __future__ import division, print_function
import matplotlib.pyplot as plt
import time
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

from sklearn.neighbors import KernelDensity

import seaborn as sns

import statsmodels.api as sm

import scipy
import scipy.stats as stats
from scipy import interpolate
from scipy.special import ndtr

from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse


##########################################################################################################################
# Define helper functions
##########################################################################################################################
"""
import math

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])
       
x = sys.getsizeof(_uncertain_sim_row_metrics)*5000
convert_size(x)
"""

   
def create_metrics(y_true, predictions, print_report):
    
    """
    Calculates various evaluation metrics based on the true labels and predicted values.
    
    Args:
        y_true (array-like): True labels.
        predictions (array-like): Predicted values.
    
    Returns:
        dict: Dictionary containing the calculated metrics.

    """
    
    # Scores
    report = classification_report(y_true, predictions, digits=4, output_dict=True)
    
    if print_report:
        
        print("\n" + classification_report(y_true, predictions, digits=4, output_dict=False))
        
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

    
    

def kde_collection_creator(dataframe, column_names, bw_method):
    
    # this creator collects the column wise kernel density estimations of a given dataframe
    
    kde_collection = []
    
    for column in column_names:
        
        # get the kde of all values inside a column of the dataset
        column_values = dataframe[column].dropna().values
        
        kde = stats.gaussian_kde(column_values, bw_method=bw_method)   
        kde_collection.append(kde)
    
    # to convert lists to dictionary
    kde_collection = {column_names[i]: kde_collection[i] for i in range(len(column_names))}  
    
    return kde_collection



def kde_latin_hypercube_sampler(kde_collection, sim_length, random_state, mode="fast", visualize_lhs_samples=False, attributekey=""):
    
    ###
        ### FUNCTION PART 1 --> KDE METRICS
    ###
    
    if mode =="accurate":
        
        # get statsmodels kde of underlying scipy gaussian kde dataset
        kde_fit = sm.nonparametric.KDEUnivariate(kde_collection.dataset.flatten())
        kde_fit.fit()
        
        support = kde_fit.support
        cdf = kde_fit.cdf
    
    
    if mode=="fast":
        
        # @https://stackoverflow.com/a/47419857
        stdev = np.sqrt(kde_collection.covariance)[0, 0]
        support = np.linspace(0, 1, kde_collection.n)
        cdf = ndtr(np.subtract.outer(support, kde_collection.dataset.flatten())/stdev).mean(axis=1)
        

    
    # preprocessing of cdf values --> drop duplicates in cdf and support,
    # if not, there could be problems with interpolation
    preproc = pd.DataFrame(data={"cdf" : cdf,  
                                 "support" : support})  
    
    preproc = preproc.copy().drop_duplicates(subset='cdf')
    
    # calculate inverse of cdf to obtain inverse cdf
    # inversefunction can be used to sample values with the latin hypercube
    inversefunction = interpolate.interp1d(preproc["cdf"], preproc["support"], kind='cubic', bounds_error=False)
    
    ###
        ### FUNCTION PART 2 --> LATIN HYPERCUBE METRICS
    ###
    
    # sample in 1-dimension with specific simulation length
    lhs_sampler = stats.qmc.LatinHypercube(1, seed=random_state)
    lhs_sample = lhs_sampler.random(n=sim_length) 

    # scale the created lhs samples to min and max cdf values
    lhs_sample_scaled = stats.qmc.scale(lhs_sample, min(preproc["cdf"]), max(preproc["cdf"])).flatten()

    ###
        ### FUNCTION PART 3 --> CALCULATE LHS SAMPLES
    ###

    generate_samples = inversefunction(lhs_sample_scaled)


    if visualize_lhs_samples:
        sns.histplot(generate_samples)
        plt.title(f"LH-Sample: {attributekey} - Sample Size: {sim_length}")
        plt.show()
        
    return generate_samples




def categorical_latin_hypercube_sampler(dataframe, key, sim_length, random_state):
    
    """
    import dataset_loader
    dataframe = dataset_loader.load_dataframe("australian", standardize_data=True)[0]
    key = "Attribute: 4"
    sim_length=100#_SIMULATION_LENGTH
    random_state = 42
    """
    
    # get frame column wit categorical data 
    column_values = dataframe.loc[:,key]
    
    # get unique values and normalize to probabilities // nan values get deleted
    unique_values = column_values.value_counts(normalize=True).sort_index()
    
    # get sorted categories & probabilities
    categories = np.array(list(unique_values.index))
    probabilities = unique_values.values
    
    # create cummulative probabilities
    cum_probs = np.cumsum(probabilities)
    
    
    # latin hyper cube sampling down below

    # sample in 1-dimension with specific simulation length
    lhs_sampler = stats.qmc.LatinHypercube(1, seed=random_state)
    lhs_sample = lhs_sampler.random(n=sim_length) 

    # create bins out of lhs samples with corresponding cummulative probabilities
    value_bins = np.searchsorted(cum_probs, lhs_sample)
    
    # get corresponding samples from categories list
    cat_lhs_samples = categories[value_bins].flatten()
    
    """
    # visual comparison
    plt.hist(column_values, density=True)
    plt.title("Original data distribution")
    plt.show()
   
    
    plt.hist(samples, density=True)
    plt.title("Latin hypercube sample (sample length" + str(sim_length) + ")")
    plt.show()
    """
  
    return cat_lhs_samples
    
    
    
    

def categorical_distribution_sample(dataframe, key, sim_length, random_state):
    
    """
    import dataset_loader
    dataframe = dataset_loader.load_dataframe("australian", standardize_data=True)[0]
    key = "Attribute: 4"
    sim_length=100#_SIMULATION_LENGTH
    random_state = 42
    """
    
    # get frame column wit categorical data 
    column = dataframe.loc[:,key]
    
    # get unique values and normalize to probabilities // nan values get deleted
    unique = column.value_counts(normalize=True).sort_index()
    
    # get sorted categories
    categories = np.array(list(unique.index))
    
    # get probabilities of catagories
    probabilities = unique.values
    
    # randomly sample from categories with corresponding probability
    if random_state is None: pass
    else: np.random.seed(random_state)
    
    draw_sample = np.random.choice(categories, sim_length, p=probabilities)
    
    """
    # visual comparison
    plt.hist(column, density=True)
    plt.title("Original data distribution")
    plt.show()
       
    
    plt.hist(draw_sample)
    plt.title("Monte carlo sample (sample length" + str(sim_length) + ")")
    plt.show()
    """
    
    return draw_sample  





# Weighting and clipping
# Amount of density below 0 & above 1
def adjust_edgeweight(y_hat, bw_method):
    
    # @https://andrewpwheeler.com/2021/06/07/kde-plots-for-predicted-probabilities-in-python/
    
    # if chosen kde bandwidth is not a number, reuturn weights 0 and compute default values
    if type(bw_method) not in [int, float]:
        edgeweight = None
        return edgeweight
    
    below_0 = stats.norm.cdf(x=0, loc=y_hat, scale=bw_method)
    above_1 = 1 - stats.norm.cdf(x=1, loc=y_hat, scale=bw_method)
    
    edgeweight = 1 / (1 - below_0 - above_1)
    
    return edgeweight






def generate_simulation_sample_collection(uncertain_attributes, dataframe_categorical, kde_collection, monte_carlo, latin_hypercube, standardize_data, datatype_map, column_names,
                                          simulation_length, random_state, lhs_mode, visualize_lhs_samples, lhs_prefix="None"):
    
        """
            # step 3: sample a value from the specific kde of the missing value - aka. beginning of MonteCarlo Simulation
            # --> and safe sampled values for this row in a history
        """
        
        sample_collection = []
        
        
        adjusted_random_state = random_state
        
        # sample from uncertain and original kde for input imputation
        for key in uncertain_attributes:

            # for each increament in the simulated column, a different random state will be used
            if adjusted_random_state == None: pass
            else:  adjusted_random_state+=2
            

            # sample from categorical distribution
            if datatype_map[key] == "Categorical":

                if monte_carlo:
    
                    # random samples from with respective probabilities
                    categorical_sample = categorical_distribution_sample(dataframe_categorical, key, simulation_length, random_state=adjusted_random_state)
                    
                    
                if latin_hypercube:
                
                    categorical_sample = categorical_latin_hypercube_sampler(dataframe_categorical, key, simulation_length, random_state=adjusted_random_state)
                
                # append draws to collection
                sample_collection.append(categorical_sample)



            # sample from categorical distribution // KDE Distributions
            elif datatype_map[key] == "Continuous":
                
                if monte_carlo:
                    
                    # resample randomly a new dataset of the underlying kde
                    distribution_sample = kde_collection[key].resample(simulation_length, seed=adjusted_random_state).flatten()

                if latin_hypercube:
                    
                    # 1 dimensional latin hypercube sampling
                    distribution_sample = kde_latin_hypercube_sampler(kde_collection[key], simulation_length, adjusted_random_state, 
                                                                      mode=lhs_mode, visualize_lhs_samples=visualize_lhs_samples,
                                                                      attributekey=key + lhs_prefix).flatten()
                
                # if standardize is true and values x are x < 0 or x > 1, then set x respectively to 0 or 1
                if standardize_data:
                    
                    distribution_sample[(distribution_sample < 0)] = 0
                    distribution_sample[(distribution_sample > 1)] = 1
                    

                sample_collection.append(distribution_sample)

            
            else: 
                print("Error in datatype_map")
                sys.exit()


        
        sample_collection = pd.DataFrame(sample_collection).transpose()
        sample_collection.columns = uncertain_attributes

        
        return sample_collection
    
    
    
def generate_simulation_inputs(simulation_row, simulation_length, uncertain_attributes, sample_collection):
    
    """
        # step 4: create DATAFRAME for faster simulation (basis input) and replace missing values with sampled ones   
        # index length of DATAFRAME_MISS_ROW is now equal to number of simulations
    """
    
    # expand simulation row to be as long as the simulation length
    sim_foundation = simulation_row.copy().transpose()
    sim_foundation = pd.concat([sim_foundation] * simulation_length, ignore_index=True)

    # basis dataframe used for simulation
    dataframe_distribution = sim_foundation.copy()
    
    # replace the missing values in dataframe_distribution with the created distribution samples 
    for col in uncertain_attributes:
        
        dataframe_distribution[col] = sample_collection[col]

    return dataframe_distribution



"""
def create_pred_simulation_metrics(y_simulation_hat, original_input_row_outcome, bw_method, x_axis, normalize):
    
        # simulation non-parametric statistics
        simulation_result_kde = stats.gaussian_kde(y_simulation_hat, bw_method=bw_method, 
                                                   weights=adjust_edgeweight(y_simulation_hat, bw_method))
        
        kde_pdfs = simulation_result_kde.pdf(x_axis) 
        
        if normalize:
            scaler = MinMaxScaler()
            kde_pdfs = scaler.fit_transform(kde_pdfs.reshape(-1, 1)).reshape(-1)   
        
        
        row_metrics = {"y_hat" : y_simulation_hat,
                       "y_hat_labels" : (y_simulation_hat>0.5).astype("int32"),
                       "mean" : y_simulation_hat.mean(),
                       "median" : np.median(y_simulation_hat),

                       #"min" : min(y_simulation_hat),
                       #"max" : max(y_simulation_hat),
                       "std" : y_simulation_hat.std(),
                       #"kurtosis" : stats.kurtosis(y_simulation_hat),
                       #"skewness" : stats.skew(y_simulation_hat),
                       "pdfs" : kde_pdfs,
                       "pdf_x_axis" : x_axis,
                       "lower_probability" : round(simulation_result_kde.integrate_box_1d(float("-inf"), 0.5), 8),
                       "upper_probability" : round(simulation_result_kde.integrate_box_1d(0.5, float("inf")), 8),
                       }
        
        #row_metrics["mean_label"] = (row_metrics["mean"]>0.5).astype("int32")
        #row_metrics["median_label"] = (row_metrics["median"]>0.5).astype("int32")
        #row_metrics["mode_label"] = (row_metrics["mode"]>0.5).astype("int32")
        
        row_metrics["probability_label"] = ((row_metrics["upper_probability"])>0.5).astype("int32")
        row_metrics["probability_sum"] = round(row_metrics["lower_probability"] + row_metrics["upper_probability"], 2)
        
        row_metrics["accuracy"] = metrics.accuracy_score(original_input_row_outcome, row_metrics["y_hat_labels"])
        #row_metrics["f1_macro"] = metrics.f1_score(original_input_row_outcome, row_metrics["y_hat_labels"], average="macro")
        #row_metrics["precision_macro"] = metrics.precision_score(original_input_row_outcome, row_metrics["y_hat_labels"], average="macro")
        
        #row_metrics["rmse_pred_error"] = mse(original_input_row_outcome, y_simulation_hat, squared=False)
        
        #row_metrics["metrics"] = create_metrics(original_input_row_outcome, row_metrics["y_hat_labels"], print_report=False)
        
        return row_metrics
"""    


                              # original_input_row, original_input_row_outcome,
def binary_sample_and_predict(model, simulation_row, dataframe_categorical, 
                              uncertain_attributes, standardize_data, datatype_map, column_names, simulation_length, random_state, 
                              monte_carlo, kde_collection, normalize_kde, bw_method, x_axis,  latin_hypercube, lhs_mode, 
                              visualize_lhs_samples, lhs_prefix):
    
        # PART 1: SAMPLE FROM DISTRIBUTION
    
        start_sample_time = time.time()
        
        INPUT_SAMPLE_COLLECTION = generate_simulation_sample_collection(uncertain_attributes = uncertain_attributes, 
                                                                        dataframe_categorical = dataframe_categorical, 
                                                                        kde_collection = kde_collection, 
                                                                        monte_carlo = monte_carlo, 
                                                                        latin_hypercube = latin_hypercube, 
                                                                        standardize_data = standardize_data, 
                                                                        datatype_map = datatype_map, 
                                                                        column_names = column_names,                                            
                                                                        simulation_length = simulation_length, 
                                                                        random_state = random_state, 
                                                                        lhs_mode = lhs_mode, 
                                                                        visualize_lhs_samples = visualize_lhs_samples, 
                                                                        lhs_prefix = lhs_prefix)
        
        
        # PART 2: COMBINE(IMPUTE) DATAFRAM WITH SAMPLES TO CREATE INPUT
        
        _X_SIMULATION_INPUT = generate_simulation_inputs(simulation_row = simulation_row, 
                                                         simulation_length = simulation_length, 
                                                         uncertain_attributes = uncertain_attributes, 
                                                         sample_collection = INPUT_SAMPLE_COLLECTION).iloc[:,:-1]
        
        end_sample_time = time.time() - start_sample_time



        # PART 3: GET PREDICTION AND METRICS        

        start_pred_time = time.time()
        
        Y_SIMULATION_HAT = model.predict(_X_SIMULATION_INPUT, verbose=0).flatten()
        
        end_pred_time = time.time() - start_pred_time
        
        """
        sim_row_metrics = create_pred_simulation_metrics(y_simulation_hat=Y_SIMULATION_HAT, 
                                                         original_input_row_outcome=original_input_row_outcome,
                                                         bw_method=bw_method, 
                                                         x_axis=x_axis, 
                                                         normalize=normalize_kde)
        """
        
        sim_row_metrics = {"y_hat" : Y_SIMULATION_HAT,
                           "y_hat_labels" : (Y_SIMULATION_HAT>0.5).astype("int32"),
                           
                           "mean" : Y_SIMULATION_HAT.mean(),
                           "median" : np.median(Y_SIMULATION_HAT),
                           "min" : min(Y_SIMULATION_HAT),
                           "max" : max(Y_SIMULATION_HAT),
                           
                           "std" : Y_SIMULATION_HAT.std(),
                           "var" : Y_SIMULATION_HAT.var(),
                           
                           "kurtosis" : stats.kurtosis(Y_SIMULATION_HAT),
                           "skewness" : stats.skew(Y_SIMULATION_HAT),
                            
                           "simulation_input" : _X_SIMULATION_INPUT,
                           "pdf_x_axis" : x_axis}
        
        
        sim_row_metrics["mean_label"] = (sim_row_metrics["mean"]>0.5).astype("int32")
        sim_row_metrics["median_label"] = (sim_row_metrics["median"]>0.5).astype("int32")

        sim_row_metrics["sample_time"] = end_sample_time
        sim_row_metrics["prediction_time"] = end_pred_time
        
        
        return sim_row_metrics
    
    
    


def calculate_simulation_results(DATAFRAME_ORIGINAL, SIMULATION_ROW_RESULTS, _SIMULATION_LENGTH, _SIMULATION_RANGE, bw_method, normalize):

    mean_values = {}

    print("    --Postprocess Stage 1\n")

    # change shape of original rows to equal the length of SIMULATION_LENGTH
    original_rows = [DATAFRAME_ORIGINAL.loc[row] for row in DATAFRAME_ORIGINAL.iloc[_SIMULATION_RANGE].index]
    
    
    # calculate mean accuracy for each row and then take the mean
    original_mc_style_input = [np.concatenate([[np.array(row.iloc[:-1])]] * _SIMULATION_LENGTH) for row in original_rows]
    original_mc_style_outcome = [[row["Outcome"]] * _SIMULATION_LENGTH for row in original_rows]
    
    
    for prefix in tqdm(["Original", "Uncertain"]): 

        row_wise_accuracy = [accuracy_score(original_mc_style_outcome[index], sim[prefix+"_Simulation"]["y_hat_labels"]) for index, sim in enumerate(SIMULATION_ROW_RESULTS)]
        row_wise_accuracy = pd.Series(row_wise_accuracy, index=_SIMULATION_RANGE)
        mean_values[prefix+"_Mean_Accuracy"] = np.mean(row_wise_accuracy)
        
        # input rmse for uncertain and original simulation sample results
        input_rmse = [mse(original, np.array(SIMULATION_ROW_RESULTS[row][prefix+"_Simulation"]["simulation_input"]), squared=False) for row, original in enumerate(original_mc_style_input)]
        input_rmse = pd.Series(input_rmse, index=_SIMULATION_RANGE)
        mean_values[prefix+"_Simulation_Input_RMSE"] = np.mean(input_rmse)
    
        #append results to specific rows
        for index, sim_index in enumerate(_SIMULATION_RANGE):
            
            SIMULATION_ROW_RESULTS[index][prefix+"_Simulation"]["accuracy"] = row_wise_accuracy.loc[sim_index]
            SIMULATION_ROW_RESULTS[index][prefix+"_Simulation"]["input_rmse"] = input_rmse.loc[sim_index]
        
    
    
    del original_mc_style_outcome, original_mc_style_input
    del input_rmse, row_wise_accuracy, original_rows

    print("\n    --Postprocess Stage 2\n")

    for row in tqdm(SIMULATION_ROW_RESULTS):
        
        for prefix in ["Original", "Uncertain"]: 
            
            # simulation non-parametric statistics
            simulation_result_kde = stats.gaussian_kde(row[prefix+"_Simulation"]["y_hat"], bw_method=bw_method, 
                                                       weights=adjust_edgeweight(row[prefix+"_Simulation"]["y_hat"], bw_method))
            
            kde_pdfs = simulation_result_kde.pdf(row[prefix+"_Simulation"]["pdf_x_axis"]) 
            
            
            row[prefix+"_Simulation"]["mode"] = row[prefix+"_Simulation"]["pdf_x_axis"][np.argmax(kde_pdfs)]
            
            if normalize:
                scaler = MinMaxScaler()
                kde_pdfs = scaler.fit_transform(kde_pdfs.reshape(-1, 1)).reshape(-1)   
            
            
            row[prefix+"_Simulation"]["mode_label"] = (row[prefix+"_Simulation"]["mode"]>0.5).astype("int32")
              
            row[prefix+"_Simulation"]["kde_pdf_y_axis"] = kde_pdfs
            
            row[prefix+"_Simulation"]["lower_probability"] = round(simulation_result_kde.integrate_box_1d(float("-inf"), 0.5), 8)
            row[prefix+"_Simulation"]["upper_probability"] = round(simulation_result_kde.integrate_box_1d(0.5, float("inf")), 8)
            
            row[prefix+"_Simulation"]["probability_label"] = ((row[prefix+"_Simulation"]["upper_probability"])>0.5).astype("int32")
            row[prefix+"_Simulation"]["probability_sum"] = round(row[prefix+"_Simulation"]["lower_probability"] + row[prefix+"_Simulation"]["upper_probability"], 2)
            

    for deletion in SIMULATION_ROW_RESULTS:
        del deletion["Original_Simulation"]["simulation_input"]
        del deletion["Uncertain_Simulation"]["simulation_input"]
    
    
    print("\n    --Postprocess Stage 3\n")
    
    # collect all labels
    for prefix in tqdm(["Original", "Uncertain"]): 
        
        mean_values[prefix+"_Mean"] = [results[prefix+"_Simulation"]["mean"] for results in SIMULATION_ROW_RESULTS]
        mean_values[prefix+"_Mode"] = [results[prefix+"_Simulation"]["mode"] for results in SIMULATION_ROW_RESULTS]
        mean_values[prefix+"_Median"] = [results[prefix+"_Simulation"]["median"] for results in SIMULATION_ROW_RESULTS]
        mean_values[prefix+"_Mode"] = [results[prefix+"_Simulation"]["mode"] for results in SIMULATION_ROW_RESULTS]
        
        mean_values[prefix+"_Mean_Labels"] = [results[prefix+"_Simulation"]["mean_label"] for results in SIMULATION_ROW_RESULTS]
        mean_values[prefix+"_Mode_Labels"] = [results[prefix+"_Simulation"]["mode_label"] for results in SIMULATION_ROW_RESULTS]
        mean_values[prefix+"_Median_Labels"] = [results[prefix+"_Simulation"]["median_label"] for results in SIMULATION_ROW_RESULTS]
        mean_values[prefix+"_Mode_Labels"] = [results[prefix+"_Simulation"]["mode_label"] for results in SIMULATION_ROW_RESULTS]
        mean_values[prefix+"_Probability_Labels"] = [results[prefix+"_Simulation"]["probability_label"] for results in SIMULATION_ROW_RESULTS]
        
    return SIMULATION_ROW_RESULTS, mean_values

  
    
  