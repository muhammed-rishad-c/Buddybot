import os
import sys
from datetime import datetime
from src import training_pipeline

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        #self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifacts_name=training_pipeline.ARTIFACT_DIR
        self.artifacts_dir=os.path.join(self.artifacts_name,timestamp)
        self.timestamp:str=timestamp
        
class DataIngestionConfig:
    def __init__(self,training_pipeline_config=TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(training_pipeline_config.artifacts_dir,
                        training_pipeline.DATA_INGESTION_DIR_NAME)
        self.featured_store_file_path:str=os.path.join(self.data_ingestion_dir,
                        training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
                        training_pipeline.FILE_NAME)
        self.train_file_name:str=os.path.join(self.data_ingestion_dir,
                        training_pipeline.DATA_INGESTION_INGESTED_DIR,
                        training_pipeline.TRAIN_FILE_NAME)
        self.test_file_name:str=os.path.join(self.data_ingestion_dir,
                        training_pipeline.DATA_INGESTION_INGESTED_DIR,
                        training_pipeline.TEST_FILE_NAME)
        
class DataTransformationConfig:
    def __init__(self,training_pipeline_config=TrainingPipelineConfig):
        self.data_transformation_dir:str=os.path.join(training_pipeline_config.artifacts_dir,
                        training_pipeline.DATA_TRANSFORAMTION_TRANSFORM_DIR)
        self.data_transformation_train_input_ids:str=os.path.join(self.data_transformation_dir,
                        training_pipeline.DATA_TRANSFORAMTION_TRANSFORM_DIR,
                        training_pipeline.DATA_TRANSFORMATION_TRAIN_INPUT_ID)
        self.data_transformation_train_attention_mask:str=os.path.join(self.data_transformation_dir,
                        training_pipeline.DATA_TRANSFORAMTION_TRANSFORM_DIR,
                        training_pipeline.DATA_TRANSFORMATION_TRAIN_ATTENTION_MASK)
        self.data_transformation_test_input_ids:str=os.path.join(self.data_transformation_dir,
                        training_pipeline.DATA_TRANSFORAMTION_TRANSFORM_DIR,
                        training_pipeline.DATA_TRANSFORMATION_TEST_INPUT_ID)
        self.data_transformation_test_attention_mask:str=os.path.join(self.data_transformation_dir,
                        training_pipeline.DATA_TRANSFORAMTION_TRANSFORM_DIR,
                        training_pipeline.DATA_TRANSFORMATION_TEST_ATTENTION_MASK)
        
        