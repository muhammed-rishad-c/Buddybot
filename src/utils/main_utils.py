from src.loggings.custom_loggings import logging
from src.exception.custom_exception import CustomException
import pandas as pd
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