# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:23:57 2023

@author: Selii
"""


import os

from data_visualizations import plot_history

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_model_path():
    
    # set overall path to tensorflow models
    model_path = os.path.join(os.getcwd(), 'models')
    
    return model_path




def load_wdbc_settings(inputs):
    
    # train settings
    batch_size = 15
    epochs = 50
    
    # layers of the network
    x = layers.Dense(16, activation='relu')(inputs)  
    outputs = layers.Dense(1, activation="sigmoid")(x) 
    
    return batch_size, epochs, outputs




def load_climate_simulation_settings(inputs):
    
    # train settings
    batch_size = 10
    epochs = 100
    
    # layers of the network
    x = layers.Dense(9, activation='relu')(inputs)
    x = layers.Dense(5, activation='relu')(x)  
    #x = layers.Dense(8, activation='relu')(x)
    outputs = layers.Dense(1, activation="sigmoid")(x) 
    
    return batch_size, epochs, outputs




def load_australian_settings(inputs):
    
    # train settings
    batch_size = 15
    epochs = 50
    
    # layers of the network
    x = layers.Dense(30, activation='relu')(inputs)
    x = layers.Dense(10, activation='relu')(x)  
    #_x = layers.Dense(8, activation='relu')(x)
    outputs = layers.Dense(1, activation="sigmoid")(x) 
    
    return batch_size, epochs, outputs



def create_binary_model(dataframe_name, X_train, y_train, X_test, y_test, save_model):
    
    inputs = keras.Input(shape=(X_original.shape[1]))
    
    # load rest of the model layers of a dataset
    if dataframe_name == "wdbc": batch_size, epochs, outputs = load_wdbc_settings()
     
    elif dataframe_name == "climate_simulation": batch_size, epochs, outputs = load_climate_simulation_settings()

    elif dataframe_name == "australian": batch_size, epochs, outputs = load_australian_settings()

    else: 
        print("No valid model found!")
        sys.exit()
                 
        
    # build model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # compile model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])
    
    # fit model
    model_history = model.fit(X_train, 
                              y_train, 
                              validation_data=[X_test, y_test], 
                              batch_size=batch_size, 
                              epochs=epochs, 
                              verbose=0)
    
    if save_model:
        #get global model path & save new model
        model_path = get_model_path()
        model.save(os.path.join(model_path, _dataset + "_binary_model.keras"))
        

    # plot model history
    plot_history(model_history, model_type="binary")

    
    return trained_model



def load_binary_model(dataframe_name):
    
    model_path = get_model_path()
    model = keras.models.load_model(os.path.join(model_path, dataframe_name + "_binary_model.keras"))
    
    print("\nShowing trained model summary:\n")
    model.summary()
    
    return model