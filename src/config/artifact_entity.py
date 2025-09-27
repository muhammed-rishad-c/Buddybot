from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_filepath:str
    test_filepath:str
    raw_filepath:str

@dataclass
class DataTransformationArtifact:
    train_input_id:str
    train_attention_mask:str
    test_input_id:str
    test_attention_mask:str