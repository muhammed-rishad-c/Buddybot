from src.components.data_ingestion import DataIngestion
from src.config.config_entity import TrainingPipelineConfig
from src.config.config_entity import DataIngestionConfig
from src.config.artifact_entity import DataIngestionArtifact


if __name__=="__main__":
    training_pipeline=TrainingPipelineConfig()
    data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline)
    data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact=data_ingestion.initiate_data_ingestion()