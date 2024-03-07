import pandas as pd
import numpy as np

from database_processing.flatandlabelsprocessor import FlatAndLabelsProcessor


class Hir_FLProcessing(FlatAndLabelsProcessor):
    def __init__(self):
        super().__init__(dataset='hirid')
        self.labels = self.load(f'{self.savepath}/labels.parquet')

    def preprocess_labels(self):
        labels = self.labels

        labels['admissionid'] = pd.to_numeric(labels['admissionid'],
                                              errors='coerce',
                                              downcast='integer')

        labels['admissiontime'] = pd.to_datetime(labels['admissiontime'],
                                                 errors='coerce')

        labels['uniquepid'] = labels['admissionid']
        labels['raw_height'] = labels['height']
        labels['raw_weight'] = labels['weight']
        labels['origin'] = np.nan
        labels['unit_type'] = np.nan

        labels['discharge_location'] = (labels.discharge_status
                                              .replace({'dead': 'death',
                                                        'alive': 'unknown'}))

        labels = (labels.pipe(self.medianfill,
                              cols=['height', 'weight'])
                  .pipe(self.harmonize_los,
                        label_los_col='lengthofstay',
                        unit='second')
                  .rename(columns={'discharge_status': self.mor_col,
                                   'admissionid': self.idx_col})
                  .replace({'sex': {'M': 1, 'F': 0},
                            self.mor_col: {'dead': 1, 'alive': 0}})
                  .set_index(self.idx_col)
                  .dropna(subset=['mortality'])
                  .astype({'uniquepid': str,
                           'age': float}))
        return labels
