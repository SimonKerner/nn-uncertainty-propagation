# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 02:48:38 2023

@author: Selii
"""


from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
    
import matplotlib.pyplot as plt
from numpy import percentile

"""
    # load explorer first
"""

SIMULATION_ROW_RESULTS = SIMULATION_ROW_RESULTS
DATAFRAME_ORIGINAL = DATAFRAME_ORIGINAL

_SIMULATION_RANGE = range(len(DATAFRAME_ORIGINAL))
_SIMULATION_LENGTH = SIMULATION_ROW_RESULTS[-1]["0_Overall_Row_Data"]["0.5_Simulation_length"]


def calculate_ci_95(y_hat, median):
    
    # central_tendency
    median = median
     
    # sigmoid predictions           
    y_hat = pd.Series(y_hat)
    
    # calculate 95% confidence intervals (100 - alpha)
    alpha = 5.0
    
    # calculate lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    lower = max(0.0, percentile(y_hat, lower_p))
    
    # calculate upper percentile (e.g. 97.5)
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at upper percentile
    upper = min(1.0, percentile(y_hat, upper_p))
    
    
    ci_95 = y_hat[(y_hat.between(lower, upper) == True)]
    ci_95_labels = (ci_95>0.5).astype("int32")
    
    return ci_95_labels


additional_data = {}

# mean std propagated through the neural network // amount of uncertainty
for prefix in tqdm(["Original", "Uncertain"]): 

    row_wise_std = [np.std(sim[prefix+"_Simulation"]["y_hat"]) for index, sim in enumerate(SIMULATION_ROW_RESULTS)]
    row_wise_std = pd.Series(row_wise_std, index=_SIMULATION_RANGE)
    additional_data[prefix+"_Mean_std"] = np.mean(row_wise_std)
    

    # mean var propagated through the neural network // amount of uncertaintyfor prefix in tqdm(["Original", "Uncertain"]): 

    row_wise_var = [np.var(sim[prefix+"_Simulation"]["y_hat"]) for index, sim in enumerate(SIMULATION_ROW_RESULTS)]
    row_wise_var = pd.Series(row_wise_var, index=_SIMULATION_RANGE)
    additional_data[prefix+"_Mean_var"] = np.mean(row_wise_var)
    
    
    
    # change shape of original rows to equal the length of SIMULATION_LENGTH
    original_rows = [DATAFRAME_ORIGINAL.loc[row] for row in DATAFRAME_ORIGINAL.iloc[_SIMULATION_RANGE].index]
    
    
    # calculate mean accuracy for each row and then take the mean
    original_mc_style_outcome = [[row["Outcome"]] * _SIMULATION_LENGTH for row in original_rows]    
        
    
    row_wise_95_accuracy = []
    for index, sim in enumerate(SIMULATION_ROW_RESULTS):
        
        ci_95_labels = calculate_ci_95(sim[prefix+"_Simulation"]["y_hat"], sim[prefix+"_Simulation"]["median"])
        original_mc_style_outcome = [original_rows[index]["Outcome"]] * len(ci_95_labels)
        ci_95_accuracy = accuracy_score(original_mc_style_outcome, ci_95_labels)
        
        row_wise_95_accuracy.append(ci_95_accuracy)
        

    row_wise_95_accuracy = pd.Series(row_wise_95_accuracy, index=_SIMULATION_RANGE)
    additional_data[prefix+"_Mean_Accuracy_95_CI"] = np.mean(row_wise_95_accuracy)
    
   

