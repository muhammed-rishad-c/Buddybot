from datasets import load_dataset
from src.config.config_entity import DataIngestionConfig
from src.config.artifact_entity import DataIngestionArtifact
from src.utils.main_utils import save_df,saving_raw_data
from src.loggings.custom_loggings import logging
from src.exception.custom_exception import CustomException
import os,sys

class DataIngestion:
    def __init__(self,data_ingestion_config=DataIngestionConfig):
        self.data_ingestion_config=data_ingestion_config
    
    def check_data_integrity(self,dataset):
        try:
            total_dialogues=len(dataset)
            short_dialogues=0
            empty_dialogues=0
        
            for item in dataset:
                if not item['dialog']:
                    empty_dialogues+=1
                elif len(item['dialog'])<3:
                    short_dialogues+=1
                
        
            print(f"total dialogue : {total_dialogues}")
            print(f"short dialogue : {short_dialogues}")
            print(f'empty dialogue : {empty_dialogues}')
        except Exception as e:
            raise CustomException(e,sys)
        
    def filter_by_dialogue_length(self, example):
        try:
            return len(example['dialog'])>=3
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def initiate_data_ingestion(self):
        try:
            logging.info("initiate data ingestion....")
            daily_dialog_dataset=load_dataset('daily_dialog')
            
            logging.info("dowloaded dataset from daily dialog")
        
            print('checking the train dataset integrity')
            self.check_data_integrity(daily_dialog_dataset['train'])
                
            print('checking the test dataset integrity')
            self.check_data_integrity(daily_dialog_dataset['test'])

            print('checking the validation dataset integrity')
            self.check_data_integrity(daily_dialog_dataset['validation'])
            
            filter_daily_dialog=daily_dialog_dataset.filter(self.filter_by_dialogue_length)
            
            logging.info("dataset are filter which short dialoge is removed")
            
            self.check_data_integrity(filter_daily_dialog['train'])
            self.check_data_integrity(filter_daily_dialog['test'])
            self.check_data_integrity(filter_daily_dialog['validation'])
            
            save_df(filter_daily_dialog['train'],self.data_ingestion_config.train_file_name)
            save_df(filter_daily_dialog['test'],self.data_ingestion_config.test_file_name)
            saving_raw_data(daily_dialog_dataset,self.data_ingestion_config.featured_store_file_path)
            
            data_ingestion_artifact=DataIngestionArtifact(
                train_filepath=self.data_ingestion_config.train_file_name,
                test_filepath=self.data_ingestion_config.test_file_name,
                raw_filepath=self.data_ingestion_config.featured_store_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
        