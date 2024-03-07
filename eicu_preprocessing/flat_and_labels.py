import pandas as pd

from database_processing.flatandlabelsprocessor import FlatAndLabelsProcessor


class eicu_FLProcessor(FlatAndLabelsProcessor):
    def __init__(self):
        super().__init__(dataset='eicu')
        self.flat = self.load(self.flat_savepath)
        self.labels = self.load(self.labels_savepath)

    def preprocess_flat(self):
        flat = self.flat
        flat = (flat.replace({'teachingstatus': {'t': 1, 'f': 0},
                              'gender': {'Male': 1, 'Female': 0},
                              'age': {'> 89': 90}})
                .rename(columns={'patientunitstayid': self.idx_col})
                .set_index(self.idx_col)
                .drop(columns=['apacheadmissiondx']))

        flat['gender'] = (flat['gender'].pipe(pd.to_numeric, errors='coerce')
                                        .fillna(0.5))
        
        flat = flat.pipe(self.medianfill, cols=['age']).astype({'age': float})
        
        flat['origin'] = flat['unitadmitsource']
        flat['raw_age'] = flat['age']
        flat['raw_admissionheight'] = flat['admissionheight']
        flat['raw_admissionweight'] = flat['admissionweight']

        categorical_features = ['ethnicity', 'unittype', 'unitadmitsource',
                                'unitvisitnumber', 'unitstaytype', ]
        features_for_min_max = ['admissionweight',
                                'admissionheight', 'age', 'hour']
        medianfill_cols = ['admissionheight', 'admissionweight']

        flat = (flat.pipe(self.categorical_dummies, cols=categorical_features)
                    .pipe(self.clip_and_norm, cols=features_for_min_max)
                    .pipe(self.medianfill, cols=medianfill_cols)
                    .sort_index())

        return flat

    def preprocess_labels(self):
        flat = self.preprocess_flat()
        labels = self.labels

        labels = (labels.replace({'unitdischargestatus': {'Expired': 1,
                                                          'Alive': 0}})
                  .rename(columns={
                              'patientunitstayid': self.idx_col,
                              'unitdischargestatus': self.mor_col,
                              'unitdischargelocation': 'discharge_location',
                              'hospitalid': 'care_site',
                              'unittype': 'unit_type'})
                  .astype({'care_site': int})
                  .set_index(self.idx_col)
                  .sort_index())

        labels['age'] = flat['raw_age']
        labels['sex'] = flat['gender']
        labels['raw_height'] = flat['raw_admissionheight']
        labels['raw_weight'] = flat['raw_admissionweight']
        labels['weight'] = flat['raw_admissionweight']
        labels['height'] = flat['raw_admissionweight']
        labels['origin'] = flat['origin']

        labels = (labels.pipe(self.clip_and_norm,
                              clip=True,
                              recompute_quantiles=True,
                              cols=['height', 'weight'])
                  .pipe(self.harmonize_los,
                        label_los_col='unitdischargeoffset',
                        unit='minute'))

        return labels
