# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:38:28 2023

@author: Selii
"""


import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from utils import add_missing_values


def get_dataset_path():
    
    # set overall path to datasets
    dataset_path = os.path.join(os.getcwd(), 'datasets')
    
    return dataset_path




def standardize_col_names(dataframe):
    
    column_names = ["Attribute: " + str(i) for i in range(len(dataframe.columns))]
    column_names[-1] = "Outcome"
    dataframe.columns = column_names    
    
    return dataframe




def create_datatype_mapping(dataframe_columns, datatypes):
    
    datatype_map = {dataframe_columns[i] : datatypes[i] for i in range(len(dataframe_columns))}
    
    return datatype_map



def standardize_dataframe(dataframe):
    
    # use data scaler to norm the data (scaler used = MinM_axsclaer, values between 0 and 1)
    scaler = MinMaxScaler()
    
    # save column names
    column_names = dataframe.columns
        
    # scale data to min-max, so that every value is between 0 and 1
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe))
    
    # change column names back to original
    dataframe.columns = column_names
        
    return dataframe




def load_wdbc():
    
    path = get_dataset_path()
    
    with open(os.path.join(path, "wdbc" + ".dat"), 'rb') as df:
        df = pd.read_table(df, sep=",", engine='python', header = None)
    
    # drop the first column (contains ids) and move the orig. second colum (contains outcomes) to the end
    outcomes, df = [df.iloc[:,1].copy(), df.iloc[:, 2:].copy()]
    df = df.merge(outcomes, left_index=True, right_index=True)
    
    # change string outcome values to type int
    df.iloc[:,-1].replace(['B', 'M'], [0, 1], inplace=True)
    
    # set datatypes for each attribute
    datatypes=["Continuous", "Continuous", "Continuous", "Continuous", "Continuous", "Continuous", "Continuous", 
               "Continuous", "Continuous", "Continuous", "Continuous", "Continuous", "Continuous", "Continuous",
               "Continuous", "Continuous", "Continuous", "Continuous", "Continuous", "Continuous", "Continuous",
               "Continuous", "Continuous", "Continuous", "Continuous", "Continuous", "Continuous", "Continuous",
               "Continuous", "Continuous", "Categorical"]
    
    return df, datatypes




def load_climate_simulation():
    
    path = get_dataset_path()
    
    with open(os.path.join(datasetpath, "climate_simulation" + ".dat"), 'rb') as df:
        df = pd.read_table(df, sep="\s+", engine='python', header = 0)
    
    # drop the first two elements of the dataset -> not relevant
    df = df.iloc[:, 2:]
    
    
    # set datatypes for each attribute
    datatypes=["Categorical", "Continuous", "Continuous", "Categorical", "Categorical", "Categorical", "Continuous", 
                "Categorical", "Categorical", "Continuous", "Categorical", "Categorical", "Continuous", "Continuous",
                "Categorical"]
    
    return df, datatypes




def load_australian():
    
    path = get_dataset_path()
    
    with open(os.path.join(path, "australian" + ".dat"), 'rb') as df:
        df = pd.read_table(df, sep=" ", engine="python", header=None)    
    
    # set datatypes for each attribute
    datatypes=["Categorical", "Continuous", "Continuous", "Categorical", "Categorical", "Categorical", "Continuous", 
                "Categorical", "Categorical", "Continuous", "Categorical", "Categorical", "Continuous", "Continuous",
                "Categorical"]
    
    return df, datatypes



def load_dataframe(dataframe_name, standardize_data):
    
    # choose datased to load
    if dataframe_name == "wdbc": dataframe, datatypes = load_wdbc()
     
    elif dataframe_name == "climate_simulation": dataframe, datatypes = load_climate_simulation()

    elif dataframe_name == "australian": dataframe, datatypes = load_australian()

    else: 
        print("No valid dataset found!")
        sys.exit()
    
    # standardize column names to attributes with a specific number (increasing)
    dataframe = standardize_col_names(dataframe)

    # create a datatype mapping in form of "attribute : type"
    datatype_map = create_datatype_mapping(dataframe.columns, datatypes)
    
    # standardize dataframe with min_max_scaler
    if standardize_data:
        dataframe = standardize_dataframe(dataframe)
    
    return dataframe, datatype_map




def create_miss_dataframe(df_name, dataframe, path, miss_rate, delete_mode, random_seed):
    
    df_miss = add_missing_values(dataframe, delete_mode =delete_mode, miss_rate=miss_rate, random_seed=random_seed) 

    # save DATAFRAME_MISS to pickle.dat 
    df_miss.to_pickle(os.path.join(path, "miss_frames", df_name, df_name + "_miss_rate_" + str(miss_rate) + ".dat"))

    return df_miss




def load_miss_dataframe(df_name, dataframe, miss_rate, delete_mode, random_seed, load_dataframe_miss, create_dataframe_miss):
    
    
    path = get_dataset_path()
    
    
    # second part is a statement to check if a dataframe really exists and if not, a new one will be created even if load is true
    if load_dataframe_miss and Path(os.path.join(path, "miss_frames", df_name, df_name + "_miss_rate_" + str(miss_rate) + ".dat")).exists() and create_dataframe_miss==False:
      
        """
            already created DATAFRAME_MISS will be loaded
        """

        df_miss = pd.read_pickle(os.path.join(path, "miss_frames", df_name, df_name + "_miss_rate_" + str(miss_rate) + ".dat"))    
        
        
        
    elif create_dataframe_miss or Path(os.path.join(path, "miss_frames", df_name, df_name + "_miss_rate_" + str(miss_rate) + ".dat")).exists() == False:
        
        """
            a new DATAFRAME_MISS will be created and saved
                # if dataset folder does not exist, create a new one
        """

        if Path(os.path.join(path, "miss_frames", df_name)).exists() == False:
            os.mkdir(os.path.join(path, "miss_frames", df_name)) 
        
        df_miss = create_miss_dataframe(df_name, dataframe, path, miss_rate, delete_mode, random_seed)

        

    print("\nDataframe MISS Statistics:")
    print("Size of Original Dataframe:", dataframe.size)
    print(f"Deletion Settings: Mode={delete_mode} and Rate={miss_rate}")
    print("Deleted:", round(df_miss.isnull().sum().sum()), "Values from Original")
    # missing values per column
    #DATAFRAME_MISS.isnull().mean() * 100
    print("Missing data: ~" + str(round(df_miss.isnull().sum().sum() * 100 / dataframe.size, 2)), "%\n")

    
    return df_miss