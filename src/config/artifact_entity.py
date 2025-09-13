from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_filepath:str
    test_filepath:str
    valid_filepath:str
    raw_filepath:str