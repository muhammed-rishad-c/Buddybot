from datasets import load_dataset

class DataIngestion:
    def __init__(self):
        pass
    
    def check_data_integrity(self,dataset):
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
        
    def filter_by_dialogue_length(self, example):
        return len(example['dialog'])>=3
    
    
    def initiate_data_ingestion(self):
        daily_dialog_dataset=load_dataset('daily_dialog')
        
        print('checking the train dataset integrity')
        self.check_data_integrity(daily_dialog_dataset['train'])
            
        print('checking the test dataset integrity')
        self.check_data_integrity(daily_dialog_dataset['test'])

        print('checking the validation dataset integrity')
        self.check_data_integrity(daily_dialog_dataset['validation'])
        
        filter_daily_dialog=daily_dialog_dataset.filter(self.filter_by_dialogue_length)
        
        self.check_data_integrity(filter_daily_dialog['train'])
        self.check_data_integrity(filter_daily_dialog['test'])
        self.check_data_integrity(filter_daily_dialog['validation'])
        
        
        
if __name__=="__main__":
    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()
        
        
