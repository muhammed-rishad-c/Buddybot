from src.loggings.custom_loggings import logging
from src.exception.custom_exception import CustomException
import pandas as pd
import numpy as np
import os,sys


def save_df(file:pd.DataFrame,filepath:str):
    try:
        dirname=os.path.dirname(filepath)
        os.makedirs(dirname,exist_ok=True)
        file.to_csv(filepath)
    except Exception as e:
        raise CustomException(e,sys)
    
def saving_raw_data(file,filepath):
    try:
        dirname=os.path.dirname(filepath)
        os.makedirs(dirname,exist_ok=True)
        file.save_to_disk(filepath)
    except Exception as e:
        raise CustomException(e,sys)
    
def read_df(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_np_array(object,filepath):
    try:
        dirname=os.path.dirname(filepath)
        os.makedirs(dirname,exist_ok=True)
        np.save(filepath,object)
        
    except Exception as e:
        raise CustomException(e,sys)