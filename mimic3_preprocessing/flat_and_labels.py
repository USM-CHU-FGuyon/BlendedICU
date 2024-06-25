import pandas as pd

from database_processing.flatandlabelsprocessor import FlatAndLabelsProcessor


class mimic3_FLProcessor(FlatAndLabelsProcessor):
    def __init__(self):
        super().__init__(dataset='mimic3')
        self.flat = self.load(self.flat_savepath)
        self.labels = self.load(self.labels_savepath)
        self.seconds_in_a_year = 3.154e+7
        
    def preprocess_labels(self):
        
        flat = (self.flat.rename(columns={'ICUSTAY_ID': self.idx_col})
                    .set_index(self.idx_col)
                    .sort_index())
        
        flat['raw_height'] = flat['height']
        flat['raw_weight'] = flat['weight']
        
        flat = (flat.replace({'GENDER': {'M': 1, 'F': 0}})
                    .pipe(self.clip_and_norm,
                          cols=['height', 'weight'],
                          recompute_quantiles=True,
                          clip=True)
                    .pipe(self.medianfill,
                          cols=['height', 'weight']))
        
        labels = (self.labels.rename(columns={
                                        'ICUSTAY_ID': self.idx_col,
                                        'SUBJECT_ID': 'uniquepid',
                                        'HOSPITAL_EXPIRE_FLAG': self.mor_col,
                                        'FIRST_CAREUNIT': 'unit_type',
                                        'DISCHARGE_LOCATION': 'discharge_location',
                                        'INTIME': 'intime',
                                        'LOS': self.los_col,
                                        'HADM_ID': 'hadm_id'})
                  .set_index(self.idx_col)
                  .dropna(subset='intime')
                  .astype({'uniquepid': str, 'hadm_id': str})
                  .sort_index())

        age = pd.concat([pd.to_datetime(labels['intime']).dt.date,
                        pd.to_datetime(flat['DOB']).dt.date], 
                       axis=1).dropna()
        age['age'] = ((age['intime'] - age['DOB'])
                      .apply(lambda x: x.total_seconds()/self.seconds_in_a_year)
                      .astype(int))
        labels['sex'] = flat['GENDER']
        labels['age'] = age['age'].where(age['age']<=300, 90) # https://github.com/MIT-LCP/mimic-code/issues/637
        labels['raw_height'] = flat['raw_height']
        labels['height'] = flat['height']
        labels['raw_weight'] = flat['raw_weight']
        labels['weight'] = flat['weight']
        labels['origin'] = flat['ADMISSION_LOCATION']
        return labels
