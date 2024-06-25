import pandas as pd
import polars as pl

from database_processing.dataprocessor import DataProcessor


class FlatAndLabelsProcessor(DataProcessor):
    def __init__(self, dataset,  **kwargs):
        super().__init__(dataset, **kwargs)
        self.flat = None
        self.labels = None
        #self.all_patients = self._get_all_patients()
        
    def preprocess_labels(self):
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

    def _reindexing(self, df, astypes={}):
        df.index = df.index.map(lambda x: f'{self.dataset}-{x}')
        return (df.rename_axis(self.idx_col)
                  .astype(astypes)
                  .dropna(how='all'))

    '''
    def _get_patients_from_labels(self):
        lst = []
        for pth in self.labels_pths.values():
            lf = (pl.scan_parquet(pth)
                  .select(pl.col('patient').unique())
                  )
            lst.append(lf)
        patient_list = pl.concat(lst).collect().to_pandas().patient.to_list()
        return patient_list
    '''

    '''check if necessary.
    def _get_patients_from_meds(self):
        lst = (pl.scan_parquet(self.dir_long_medication+'/*.parquet')
              .select(pl.col('patient').unique())
              .collect(streaming=True)
              .to_pandas().patient.to_list())
        return lst
    '''


    def _get_all_stays(self):
        """
        Gets the list of patients having flat, timeseries or medication data.
        """
        labels_patients = self._get_patients_from_labels()
        #med_patients = self._get_patients_from_meds()


        return

    def run_labels(self):
        self.labels = self.preprocess_labels()
        
        print(f'Initial number of admissions : {len(self.labels)}')
        if self.dataset != 'blended':
            self.labels = self._reindexing(self.labels)
        
        print(f'number of admissions after preprocessing : {len(self.labels)}')
        self.save(self.labels, f'{self.savepath}preprocessed_labels.parquet')
        return self.labels
        
    def run_flat_and_labels(self):        
        self.labels, self.flat = self.preprocess_flat_and_labels()

        print(f'Initial number of admissions : {len(self.labels)}')
        if self.dataset != 'blended':
            self.flat = self._reindexing(self.flat)
            self.labels = self._reindexing(self.labels)

        print(f'number of admissions after preprocessing : {len(self.labels)}')
        self.save(self.flat, f'{self.savepath}preprocessed_flat.parquet')
        self.save(self.labels, f'{self.savepath}preprocessed_labels.parquet')
        return self.flat, self.labels
    