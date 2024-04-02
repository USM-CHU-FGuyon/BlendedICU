import pandas as pd

from database_processing.dataprocessor import DataProcessor

class DiagnosesProcessor(DataProcessor):
    
    def __init__(self, dataset,):
        super().__init__(dataset)

        self.preprocessed_diagnoses_pth = self.diagnoses_pths[dataset]
    
        self.labels = self.load(self.labels_pths[dataset])
    