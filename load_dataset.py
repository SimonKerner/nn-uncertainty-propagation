# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:50:15 2023

@author: LocalAdmin
"""

import pickle
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


with open("data_crashes.pkl", 'rb') as file:
    data_crashes = pickle.load(file)
    
with open("saved_pdfs_crashes.pkl", 'rb') as file:
    data_crashes_pdf = pickle.load(file)
    
with open("data_australian.pkl", 'rb') as file:
    data_australian = pickle.load(file)
    
with open("saved_pdfs_australian.pkl", 'rb') as file:
    data_australian_pdf = pickle.load(file)