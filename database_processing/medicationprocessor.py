import operator
from functools import reduce

import pandas as pd

from utils.parquet_utils import compute_offset
from database_processing.dataprocessor import DataProcessor


class MedicationProcessor(DataProcessor):
    """
    This class allows to process the drug data of each database in a unified 
    manner. It maps the brand names and labels in the source database to 
    omop ingredients using the drug mapping (auxillary_files/medications.json).
    The outputs a dataframe that is saved as medication.parquet file or as 
    several parquet chunks when filesize is inconvenient.
    """
    def __init__(self,
                 dataset,
                 labels,
                 col_pid,
                 col_med,
                 col_time,
                 col_los,
                 unit_los,
                 unit_offset=None,
                 offset_calc=False,
                 col_admittime=None):
        super().__init__(dataset)

        self.med_mapping = self._load_med_mapping()
        self.new_med_mapping = self._new_med_mapping()

        if not offset_calc:
            if unit_offset is None:
                raise ValueError('unit_offset should be specified.')
        lab_keepcols = labels.columns[labels.columns.isin([col_pid,
                                                           col_los,
                                                           col_admittime])]
        self.labels = labels.loc[:, lab_keepcols]
        self.col_med = col_med
        self.col_pid = col_pid
        self.col_time = col_time
        self.col_los = col_los
        self.col_admittime = col_admittime
        self.offset_calc = offset_calc
        self.unit_los = unit_los
        self.unit_offset = unit_offset

    def _clip_time(self, df):
        """
        Stop tracking drugs after discharge and ignore drug admission prior
        to self.preadm_anteriority. This includes past icu stays of the same 
        patients.
        These parameters are necessary when resampling to hourly data, as we 
        don't want to resample the time in between admission when it is very 
        long.
        """
        keep_idx = ((df[self.col_time] < df[self.col_los])
                    & (df[self.col_time] > -self.preadm_anteriority*24*3600))
        return df.loc[keep_idx].drop(columns=self.col_los)

    def _new_med_mapping(self):
        return {key: val[self.dataset] for key, val in self.ohdsi_med.items()}

    def _load_med_mapping(self):
        mapping = ({v: key for v in val[self.dataset]}
                   for key, val in self.ohdsi_med.items())
        return reduce(operator.ior, mapping)

    def _add_dummy_value(self, df):
        """
        Ensures that the medication tables have the same format as timeseries
        tables. 
        """
        df['value'] = 1
        return df

    def _start_and_end(self, df):
        """
        For each drug administration, creates a time frame of drug exposure
        using the user-defined variable self.drug_exposure_time.
        In a further development, this variable could be adapted to each drug.
        """
        df = df.rename(columns={self.col_time: 'start'})
        df['end'] = df.start+self.drug_exposure_time*3600
        return df

    def _convert_to_seconds(self, df):
        df[self.col_los] = df[self.col_los] * self._second_conversion_constant(self.unit_los)
        if not self.offset_calc:
            df[self.col_time] = df[self.col_time] * self._second_conversion_constant(self.unit_offset)
        return df

    def _second_conversion_constant(self, unit):
        d = {
            'milisecond': 0.001,
            'second': 1,
            'minute': 60,
            'hour': 3600,
            'day': 3600*24,
        }
        try:
            return d[unit]
        except KeyError:
            raise ValueError(
                f'unit should be one of {[*d.keys()]}, not {unit}')

    def _offset_calc(self, df):
        if self.offset_calc:
            return compute_offset(df,
                                  col_intime=self.col_admittime,
                                  col_measuretime=self.col_time)
        else:
            return df

    def _get_ingredients(self, df):
        pieces = []
        for ingredient, labels in self.new_med_mapping.items():

            ingredient_adm = (df.loc[df[self.col_med].isin(labels)]
                                .copy()
                                .rename(columns={self.col_med:
                                                 'original_drugname'}))
            ingredient_adm['label'] = ingredient
            pieces.append(ingredient_adm)
        return pd.concat(pieces).reset_index(drop=True)

    def run(self, df):

        med = (df.merge(self.labels, on=self.col_pid)
                 .pipe(self._offset_calc)
                 .pipe(self._convert_to_seconds)
                 .pipe(self._clip_time)
                 .pipe(self._get_ingredients)
                 .pipe(self._add_dummy_value)
                 .pipe(self._start_and_end))

        return med
