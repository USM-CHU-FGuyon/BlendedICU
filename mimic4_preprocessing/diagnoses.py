import pandas as pd

from database_processing.diagnosesprocessor import DiagnosesProcessor

class mimic4_DiagProcessor(DiagnosesProcessor):
    def __init__(self):
        super().__init__(dataset='mimic4')
        self.diagnoses = self.load(self.diag_savepath)
        
    def run(self):
        
        self.diagnoses['icd_code'] = self.diagnoses['icd_version'].astype(str)+'_'+self.diagnoses['icd_code']
        
        diagnoses = (self.diagnoses
                     .rename(columns={'subject_id': self.uniquepid_col,
                                      'stay_id': self.idx_col})
                     .assign(
                       patient=lambda x: self.dataset+'-'+x[self.idx_col].astype(str),
                       diagnosis_name=lambda x: x.icd_code.map(self.diag_mapping),
                       diagnosis_concept_id=lambda x: x.diagnosis_name.map(self.diag_concept_id),
                       diagnosis_source_value=lambda x: x.long_title)
                     .fillna({'diagnosis_name': 'needs omopization',
                              'diagnosis_concept_id': -1})
                     .astype({'diagnosis_concept_id': int}))
        
        self.labels = (self.labels.assign(
                            outtime=lambda x: ((pd.to_datetime(x.intime)
                              + pd.to_timedelta(x.lengthofstay, unit='D')
                              ).dt.round('s')))
                        .astype({'hadm_id': int}))
        
        diagnoses = (diagnoses
                     .merge(self.labels['outtime'],
                            left_on=self.idx_col,
                            right_index=True)
                     .rename(columns={'outtime': 'diagnosis_start'}))
        
        self.save(diagnoses, self.preprocessed_diagnoses_pth)
    
    
    
    


