from database_processing.flatandlabelsprocessor import FlatAndLabelsProcessor


class mimic4_FLProcessor(FlatAndLabelsProcessor):
    def __init__(self):
        super().__init__(dataset='mimic4')
        self.flat = self.load(self.flat_savepath)
        self.labels = self.load(self.labels_savepath)

    def preprocess_labels(self):
        flat = (self.flat.rename(columns={'stay_id': self.idx_col})
                    .set_index(self.idx_col)
                    .sort_index())

        flat['raw_age'] = flat['age']
        flat['raw_height'] = flat['height']
        flat['raw_weight'] = flat['weight']
        flat['origin'] = flat['admission_location']

        flat = (flat.replace({'gender': {'M': 1, 'F': 0}})
                    .pipe(self.clip_and_norm,
                          cols=['height', 'weight'],
                          recompute_quantiles=True,
                          clip=True)
                    .pipe(self.medianfill,
                          cols=['height', 'weight'])
                )
        
        labels = (self.labels.rename(columns={
                                        'stay_id': self.idx_col,
                                        'subject_id': 'uniquepid',
                                        'hospital_expire_flag': self.mor_col,
                                        'first_careunit': 'unit_type'})
                  .set_index(self.idx_col)
                  .pipe(self.harmonize_los,
                        label_los_col='los')
                  .astype({'uniquepid': str, 'hadm_id': str})
                  .sort_index())

        labels['sex'] = flat['gender']
        labels['age'] = flat['raw_age']
        labels['raw_height'] = flat['raw_height']
        labels['height'] = flat['height']
        labels['raw_weight'] = flat['raw_weight']
        labels['weight'] = flat['weight']
        labels['origin'] = flat['origin']

        return labels
