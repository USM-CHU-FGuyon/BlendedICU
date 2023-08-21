from pathlib import Path

import pandas as pd

from database_processing.dataprocessor import DataProcessor


class FlatAndLabelsProcessor(DataProcessor):
    def __init__(self,
                 dataset):
        super().__init__(dataset)
        self.flat = None
        self.labels = None
        self.ts_patients = self.get_ts_patients()

    def flat_preprocessing(self):
        """
        This should be defined in each subclass.
        """
        raise NotImplementedError

    def labels_preprocessing(self):
        """
        This should be defined in each subclass.
        """
        raise NotImplementedError

    def _checknans(self):
        if self.flat.isna().sum().sum() > 0:
            raise ValueError(f'Flat contains Nans : {self.flat.isna().sum()}')

    def harmonize_los(self, labels, label_los_col, unit='day'):
        """
        Harmonizes the unit of the length of stay. 
        Applies the lower bound of length of stay: 
            stays with length lower than lower_los are removed
        length of stay longer than upper_los are clipped.
        """
        if unit == 'second':
            labels[f'true_{self.los_col}'] = labels[label_los_col]/(3600*24)
        elif unit == 'minute':
            labels[f'true_{self.los_col}'] = labels[label_los_col]/1440
        elif unit == 'hour':
            labels[f'true_{self.los_col}'] = labels[label_los_col]/24
        elif unit == 'day':
            labels[f'true_{self.los_col}'] = labels[label_los_col]
        else:
            raise ValueError(f'unit {unit} not understood.')
        labels[self.los_col] = labels['true_lengthofstay'].clip(upper=self.upper_los)
        labels = labels.loc[labels[self.los_col] > self.lower_los]
        return labels

    def categorical_dummies(self, df, cols, min_count=1000):
        """
        function for creating dummies with categorical variables. 
        only categories that are present less than "min_count" times are 
        put into 'misc' category.
        """
        df[cols] = df[cols].astype(str)
        for f in cols:
            vcounts = df[f].value_counts()
            df.loc[df[f].isin(vcounts.loc[vcounts < 1000].index), f] = 'misc'
        return pd.get_dummies(df, columns=cols)

    def get_ts_patients(self):
        """
        Gets the sorted list of patients having timeseries data.
        """
        preprocessed_ts_dir = Path(self.savepath+'partially_processed_timeseries/')
        preprocessed_ts_dir.mkdir(exist_ok=True, parents=True)
        stems = [f.stem for f in Path(preprocessed_ts_dir).iterdir()]
        try:
            ts_patients = pd.to_numeric(stems, downcast='integer')
        except ValueError:
            ts_patients = stems
        return sorted(ts_patients)

    def _reindexing(self, df, astypes={}):
        df.index = df.index.map(lambda x: f'{self.dataset}-{x}')
        return (df.rename_axis(self.idx_col)
                  .reindex(self.ts_patients)
                  .astype(astypes)
                  .dropna(how='all'))

    def run(self):
        self.flat = self.preprocess_flat()
        self.labels, self.flat = self.preprocess_labels()

        print(f'Initial number of admissions : {len(self.labels)}')
        if self.dataset != 'blended':
            self.flat = self._reindexing(self.flat)
            self.labels = self._reindexing(self.labels)

        print(f'number of admissions after preprocessing : {len(self.labels)}')

        self.save(self.flat, f'{self.savepath}preprocessed_flat.parquet')
        self.save(self.labels, f'{self.savepath}preprocessed_labels.parquet')
        return self.flat, self.labels
