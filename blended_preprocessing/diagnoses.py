import pandas as pd

from database_processing.dataprocessor import DataProcessor

class blended_DiagProcessor(DataProcessor):
    def __init__(self, datasets):
        super().__init__('blended')
        
        self.datasets = datasets
        self.loadcols = ['uniquepid',
                         'patient',
                         'diagnosis_name',
                         'diagnosis_concept_id',
                         'diagnosis_start',
                         'diagnosis_source_value']
        self.diagnoses = self._load_diagnoses()


    def _load_diagnoses(self):
        """
        Loads the preprocessed_diagnoses.parquet file from each source database.
        """
        diag_dic = {d: self._load_diagnosis(d) for d in self.datasets}
        diagnoses = (pd.concat(diag_dic)
                     .rename_axis(('source_dataset', 'uniquepid'))
                     .reset_index())
        return diagnoses
    
    def _load_diagnosis(self, dataset):
        
        pth = self.diagnoses_pths[dataset]
        df = self.load(pth, columns=self.loadcols).set_index('uniquepid')
        return df

    def run(self):
        
        self.diagnoses['uniquepid'] = (self.diagnoses['source_dataset']
                                       +'-'
                                       +self.diagnoses['uniquepid'].astype(str))
        
        self.save(self.diagnoses, self.diag_savepath)
        
        