import os
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt


def load_csv(DATASET_DIR):
    # Loading Data Files
    df = pd.read_csv(DATASET_DIR)
    return df



def load_table(DATASET_DIR, name):
    return pd.read_table(DATASET_DIR,header=None, names=name, sep='::')



                
def data_save(df, DATA_file):
    print('\nStarting save to .csv file')

    df.to_csv(
            DATA_file,
            sep=',',
            header=True,
            index=False,
            columns=df.columns 
            )
    print('Saved to ', DATA_file)

    

