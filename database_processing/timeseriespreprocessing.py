"""
This class is common to the four source databases and to the BlendedICU 
dataset.
It contains useful functions to convert the format of data, harmonize units and 
labels and save the timeseries variables as parquet files.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import numpy as np
from natsort import natsorted

from database_processing.dataprocessor import DataProcessor


class TimeseriesPreprocessing(DataProcessor):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.ts_variables_pth = self.user_input_pth+'timeseries_variables.csv'
        self.ts_variables = self._load_tsvariables()

        self.kept_ts = self.ts_variables[self.dataset].to_list()

        self.kept_med = self._kept_meds()
        self.index = None

        d = ({'time':{'blended':'time', 'eicu': 'time', 'mimic':'time', 'mimic3': 'time', 'amsterdam': 'time', 'hirid': 'time', 'is_numeric':1,'agg_method': 'last'}, 
             'hour':{'blended':'hour', 'eicu': 'hour', 'mimic':'hour', 'mimic3': 'time','amsterdam': 'hour', 'hirid': 'hour', 'is_numeric':1,'agg_method': 'last'}}
             | {med: {'blended':med, 'eicu': med, 'mimic':med, 'mimic3': med,'amsterdam': med, 'hirid': med, 'is_numeric':1, 'agg_method':'last', 'categories':'drug'} for med in self.kept_med})
        self.med_time_hour = pd.DataFrame(d).T
        
        self.cols = self._cols()

        self.col_mapping = self.cols.set_index(self.dataset, drop=False)['blended'].to_dict()
        self.omop_mapping = self.cols['hirid'].to_dict()
        self.aggregates_blended = self.cols['agg_method']
        self.aggregates = self.cols.set_index(self.dataset)['agg_method']
        self.dtypes = self._get_dtypes()
        self.is_measured = self.ts_variables.set_index('blended').notna()
        self.numeric_ts = self._get_numeric_cols()
        self.pyarrow_schema_dict = self._pyarrow_schema_dict()
        self.column_template = self._column_template()

        self.harmonizer = {
            'amsterdam': self._harmonize_amsterdam,
            'hirid': self._harmonize_hirid,
            'eicu': self._harmonize_eicu,
            'mimic': self._harmonize_mimic,
            'mimic3': self._harmonize_mimic3,
            'blended': (lambda x: x)
        }[self.dataset]

    def process_tables(self,
                       ts_ver=None,
                       ts_hor=None,
                       med=None,
                       admission_hours=None,
                       stop_at_first_chunk=False):
        '''
        time should be an int in hour
        patient a unique patient identifier.

        ts_ver : columns should be :  'patient', 'time', 'variable', 'value'
        ts_hor : columns should be :  'patient', 'time', var1, var2,...
        admission_hours : index should be 'patient', column should be 'hour'
          if no admission hour is provided, midnight will be taken as admission
          for all patients.
          
         stop_at_first_chunk:
             if True, the code will stop after processing of the first chunk.
             This can be useful for debugging or testing the pipeline on a 
             single chunk
        '''
        t_max = self.upper_los*24*3600

        timeseries = self.format_raw_data(ts_ver=ts_ver,
                                          ts_hor=ts_hor,
                                          med=med)

        time = timeseries.index.get_level_values(1)
        timeseries = timeseries.loc[(time >= 0) & (time < t_max)]
        med = med.loc[(med.start >= 0) & (med.end < t_max)]

        self.index = self._build_index(timeseries,
                                       cols_index=['patient', 'time'])

        self.index_start1 = self.index[self.index.get_level_values(1) > 0]

        self.med_mask = self._make_med_mask(med)
        self.ts = timeseries

        dropcols = [c for c in timeseries.columns if c not in self.cols['blended'].values]
        self.chunk = (timeseries.drop(columns=dropcols)
                                .pipe(self._resampling)
                                .pipe(self._mask)
                                .join(self.med_mask,
                                      how='outer')
                                .pipe(self._add_hour,
                                      admission_hours=admission_hours)
                                .reindex(self.index_start1)
                                .reset_index())

        self.chunk = (pd.concat([self.chunk, self.column_template])
                      .astype(self.dtypes))

        if stop_at_first_chunk:
            raise Exception('Stopped after 1st chunk, deactivate '
                            '"stop_at_first_chunk" to keep running.')
        self.save_timeseries(self.chunk, self.partiallyprocessed_ts_dir)

    def filter_tables(self,
                      table,
                      kept_variables=None,
                      col_id=None,
                      col_time=None,
                      col_var=None,
                      col_value=None):
        """
        This function harmonizes the column names in the input table.
        It produces a unique stay id for the blendedICU dataset by appending
        the name of the source database to the original stay ids.
        if `kept_variable` is specified, it also filters a set of variables.
        """
        table = table.rename(columns={col_id: self.idx_col,
                                      col_var: 'variable',
                                      col_time: self.time_col,
                                      col_value: 'value'})

        table[self.idx_col] = table[self.idx_col].apply(lambda x: f'{self.dataset}-{x}')

        if kept_variables is None:
            return table

        return table.loc[table['variable'].isin(kept_variables)]

    def format_raw_data(self,
                        ts_ver=None,
                        ts_hor=None,
                        med=None):
        """
        This function creates the data for the raw blendedICU dataset. 
        ts_ver:  
            DataFrame in long format: all variable names are in a single
            column. 
        ts_hor: 
            DataFrame in wide format: all variables have a dedicated column
        med: 
            Medication DataFrame produced by the medicationprocessor.
            
        All timeseries are converted to wide format and concatenated in a 
        single DataFrame.
        A parquet file per patient is saved into the formatted_timeseries
        directory. 
        
        The medication DataFrame is already properly formatted. A parquet file
        per patient is saved into the formatted_medications directory.
        """
        ts_ver = self._numericize_cols(ts_ver)

        cols_index = [self.idx_col, self.time_col]

        if ts_hor is None:
            ts_hor = pd.DataFrame(columns=cols_index)
        if ts_ver is None:
            ts_ver = pd.DataFrame(columns=cols_index+['variable', 'value'])

        ver_variables = ts_ver['variable'].drop_duplicates()
        ver_variables = ver_variables.loc[ver_variables.isin(self.cols[self.dataset])]

        ts_ver = ts_ver.set_index(cols_index)
        ts_hor = ts_hor.set_index(cols_index)

        ver_patients = ts_ver.index.get_level_values(0)
        hor_patients = ts_hor.index.get_level_values(0)
        med_patients = med.patient

        unique_patients = (ver_patients.union(hor_patients)
                                       .union(med_patients)
                                       .drop_duplicates()
                                       .sort_values())

        ts_ver, ts_hor = self._add_missing_patients(ts_ver,
                                                    ts_hor,
                                                    unique_patients,
                                                    cols_index)

        ts_hor = ts_hor.groupby(level=[0, 1]).mean()

        ts_ver = self._extract_variables(ts_ver, ver_variables)

        timeseries = (ts_ver.join(ts_hor, how='outer')
                            .pipe(self._convert_col_dtypes)
                            .pipe(self._clip_user_minmax)
                            .rename(columns=self.col_mapping)
                            .pipe(self.harmonizer)
                            .pipe(self._compute_GCS)
                            .reset_index())

        dtypes = ({v: tpe for v, tpe in self.dtypes.items() if v in timeseries.columns}
                  | {'time': np.float32, 'patient': str})

        timeseries = timeseries.astype(dtypes)
        
        med = (med.astype({'variable': 'category',
                           'start': int,
                           'end': int,
                           'value': int})
               .merge(timeseries.patient.drop_duplicates(),
                      on='patient', how='outer'))

        ts_schema = pa.schema({k:v for k,v in self.pyarrow_schema_dict.items() 
                                    if k in timeseries.columns})

        med_schema = pa.schema({'patient': pa.string(),
                                'original_drugname': pa.string(),
                                'variable': pa.string(),
                                'start': pa.float32(),
                                'end': pa.float32(),
                                'value': pa.float32()})

        self.save_timeseries(timeseries,
                             self.formatted_ts_dir, 
                             pyarrow_schema=ts_schema)

        self.save_timeseries(med,
                             self.formatted_med_dir,
                             pyarrow_schema=med_schema)

        return timeseries.set_index(cols_index)


    def _cols(self):
        """
        The `cols` variable contains the timeseries variables and medications.
        It lists the aggregation methods that should be used for each variable,
        the min and max admissible values for each variable, and the type of
        variable (numeric or str)
        """
        cols = (pd.concat([self.ts_variables, self.med_time_hour])
                .set_index('blended', drop=False))
        cols.loc[self.med_concept_id.index, 'concept_id'] = self.med_concept_id
        cols = cols.fillna({'concept_id': 0}).astype({'concept_id': int})
        return cols

    def _pyarrow_schema_dict(self):
        """
        Prepares the pyarrow schema for saving timeseries variables to parquet.
        """
        mp = {1: pa.float32(), 0: pa.string()}
        dtypes_var = {col: mp[isnum]
                      for col, isnum in self.cols.is_numeric.to_dict().items()}
        return dtypes_var | {'patient': pa.string()}

    def _get_numeric_cols(self):
        """
        returns the list of numeric cols (drugs are not included in the
        numeric cols.)
        """
        idx = (self.cols.is_numeric == 1) & (self.cols.categories != 'drug')
        return self.cols.loc[idx].index.to_list()

    def _get_dtypes(self):
        '''
        Time is int, non numeric are object, the rest is float32.
        '''
        df = self.cols.loc[self.cols[self.dataset].isin(self.kept_ts),
                           'is_numeric']
        df = df.replace({1: np.float32, 0: str})
        return df.to_dict()

    def _load_tsvariables(self):
        """
        Loads the timeseries_variables.csv file.
        """
        df = pd.read_csv(self.ts_variables_pth, sep=';')
        if (df.loc[df.is_numeric == 0, 'agg_method'] != 'last').any():
            raise ValueError('aggregates column should be "last" for '
                             'nonnumeric values.')
        return df

    def _column_template(self):
        """
        create an empty DataFrame with a column for each variable. This 
        ensures that all chunks have the same columns, even if 
        a medication or timeseries variables was not found in some chunks.
        """
        cols = pd.concat([self.cols['blended'], pd.Series(['patient'])])
        return pd.DataFrame(columns=cols).set_index('patient')

    def ls(self, pth, sort=True, aslist=True):
        """
        list the content of a directory.
        if aslist is False, the output is an iterable
        """
        iterdir = Path(pth).iterdir()
        if sort:
            return natsorted(iterdir)
        if aslist:
            return list(iterdir)
        return iterdir

    def _celsius_to_farenheit(self, series):
        return (series-32)/1.8

    def _harmonize_amsterdam(self, df):
        df['O2_arterial_saturation'] = df['O2_arterial_saturation']*100
        df['hemoglobin'] = df['hemoglobin']*1.613

        # The numericitems table indicates that all
        # tidal volume settings are in mL.
        # however there are inconsistent units in the data.
        idx_mult1000 = df['tidal_volume_setting'] < 1
        idx_nan = ((df['tidal_volume_setting'] > 1)
                   & (df['tidal_volume_setting'] < 3))
        df.loc[idx_mult1000, 'tidal_volume_setting'] *= 1000
        df.loc[idx_nan, 'tidal_volume_setting'] = np.nan
        return df

    def _harmonize_hirid(self, df):
        df['hemoglobin'] = df['hemoglobin']/10
        return df

    def _harmonize_eicu(self, df):
        """
        Conversion constants were taken from:
        https://www.wiv-isp.be/qml/uniformisation-units/conversion-table/tableconvertion-chimie.pdf
        """
        df['calcium'] = df['calcium']*0.25
        df['magnesium'] = df['magnesium']*0.411
        df['blood_glucose'] = df['blood_glucose']*0.0555
        df['creatinine'] = df['creatinine']*88.40
        df['bilirubine'] = df['bilirubine']*17.39
        df['albumin'] = df['albumin']*10
        df['blood_urea_nitrogen'] = df['blood_urea_nitrogen']*0.357
        df['phosphate'] = df['phosphate']*0.323
        # In ~2% cases, temperature is in farenheit.
        temp_idx = (df.temperature > 80, 'temperature')
        df.loc[temp_idx] = self._celsius_to_farenheit(df.loc[temp_idx])
        return df

    def _harmonize_mimic(self, df):
        """
        Unit conversion was necessary mostly for lab variables.
        """
        df['temperature'] = self._celsius_to_farenheit(df['temperature'])
        df['blood_glucose'] = df['blood_glucose']*0.0555
        df['magnesium'] = df['magnesium']*0.411
        df['creatinine'] = df['creatinine']*88.40
        df['calcium'] = df['calcium']*0.25
        df['bilirubine'] = df['bilirubine']*17.39
        df['albumin'] = df['albumin']*10.
        df['blood_urea_nitrogen'] = df['blood_urea_nitrogen']*0.357
        df['phosphate'] = df['phosphate']*0.323
        return df

    def _harmonize_mimic3(self, df):
        """
        Unit conversion was necessary mostly for lab variables.
        Same as mimic: TO check.
        """
        df['temperature'] = self._celsius_to_farenheit(df['temperature'])
        df['blood_glucose'] = df['blood_glucose']*0.0555
        df['magnesium'] = df['magnesium']*0.411
        df['creatinine'] = df['creatinine']*88.40
        df['calcium'] = df['calcium']*0.25
        df['bilirubine'] = df['bilirubine']*17.39
        df['albumin'] = df['albumin']*10.
        df['blood_urea_nitrogen'] = df['blood_urea_nitrogen']*0.357
        df['phosphate'] = df['phosphate']*0.323
        return df

    def _resampling(self, df):
        """
        Resamples the input dataframe and applies the aggregate functions 
        that are specified in the input timeseries_variables.csv file.
        """
        aggregates = self.aggregates_blended.loc[df.columns]
        df = self.df_resampler.join(df, how='outer')

        df['new_time'] = df.groupby(level=0).new_time.ffill()
        
        df = (df.droplevel(1)
                .rename(columns={'new_time': self.time_col})
                .set_index(self.time_col, append=True)
                .groupby(level=[0, 1])
                .agg(aggregates))
        return df

    def _fillna(self, df):
        """
        Fill nans with the medians from each variables. This is used in 
        the blendedICU prepocessing pipeline. 
        This step is skipped when TS_FILL_MEDIAN is False
        """
        if self.TS_FILL_MEDIAN:
            return self._fillna_with_computed_medians(df)
        else:
            return df
        
    def _fillna_with_computed_medians(self, df):
        """
        The medians should be computed before using this functionm this is 
        done in the clipping and normalization step.
        """
        if self.clipping_quantiles is None: 
            raise ValueError('Medians should be computed before nafilling.'
                             'Please set "recompute_quantiles" to True in the' 
                             'clipping and normalization step.')
        self.medians = self.clipping_quantiles[1]
        return df.fillna(self.medians)

    def _make_med_mask(self, med):
        """
        Produces a mask of drug exposure from the list of starts and ends of 
        medications.
        """
        self.med = med

        med_starts = med.set_index(['patient', 'start'])
        med_ends = med.set_index(['patient', 'end'])
        med_ends_resampled = (med_ends.pipe(self._extract_variables,
                                            kept_variables=self.kept_med)
                              .rename_axis([self.idx_col, self.time_col])
                              .pipe(self._resampling)
                              .multiply(0))
        
        med_starts_resampled = (med_starts.pipe(self._extract_variables,
                                                kept_variables=self.kept_med)
                                .rename_axis([self.idx_col, self.time_col])
                                .pipe(self._resampling))

        return (pd.concat([med_starts_resampled, med_ends_resampled])
                .groupby(level=[0, 1]).max()
                .groupby(level=0).ffill().fillna(0))

    def _clip_user_minmax(self, df):
        """
        Clips the values to the user-specified min and max in the 
        timeseries_variables.csv file.
        It is only applicable to numerical columns.
        """
        num_cols = list(set(df.columns).intersection(self.numeric_cols))
        cols = self.cols.set_index(self.dataset)
        lower = cols.user_min.loc[num_cols].values
        upper = cols.user_max.loc[num_cols].values
        df[num_cols] = df[num_cols].clip(lower=lower, upper=upper)
        return df

    def _convert_col_dtypes(self, df):
        """
        Convert numeric cols to float, a this point every value in the numeric
        columns is a float, but the column dtype in the dataframe may still be 
        'object'
        """
        numeric_cols = set(df.columns).intersection(self.numeric_cols)
        return df.astype({c: float for c in numeric_cols})

    def _numericize_cols(self, ts_ver):
        """
        Convert to numeric all columns that were specified as numeric in the 
        timeseries_variable.csv file. 
        Values that are not castable to float by pd.to_numeric will be dropped.
        """
        idx_numcols = self.cols['is_numeric'] == 1
        self.numeric_cols = self.cols.loc[idx_numcols, self.dataset].values

        idx_numeric = ts_ver.variable.isin(self.numeric_cols)
        ts_ver.loc[idx_numeric, 'value'] = (pd.to_numeric(ts_ver.loc[idx_numeric, 'value'],
                                                          errors='coerce'))  
        ts_ver = ts_ver.dropna()
        return ts_ver

    def _compute_GCS(self, df):
        """
        Some databases report the three components of the GCS score but do not 
        have a variable for the total GCS score. This function computes the 
        total GCS score for every hour where all three scores were measured.
        This function is applied before resampling to ensure that all 
        components were measured at the same time.
        
        It also adds empty columns for each component of the GCS score if
        they are not provided in the input dataframe.
        """
        glasgow_cols = ['glasgow_coma_score_eye',
                        'glasgow_coma_score_motor',
                        'glasgow_coma_score_verbal']
        if 'glasgow_coma_score' not in df.columns:
            df['glasgow_coma_score'] = df[glasgow_cols].sum(axis=1,
                                                            min_count=3)
        for glasgow_col in glasgow_cols:
            if glasgow_col not in df.columns:
                df[glasgow_col] = np.nan
        return df

    def _forward_fill(self, df):
        """
        Forward filling is optional. If FORWARD_FILL is set to 0 in the 
        config.json file, the tables will be resampled but missing values will
        be preserved.
        """
        if self.FORWARD_FILL:
            return df.groupby('patient').ffill()
        return df

    def _add_missing_patients(self,
                              ts_ver,
                              ts_hor,
                              unique_patients,
                              cols_index):
        '''
        Adds a line of nan at t=-1 for every patient.
        this ensures that every patientid appears in the dataframe.
        '''
        index = pd.MultiIndex.from_tuples([(p, -1) for p in unique_patients],
                                          names=cols_index)

        df_patients = pd.DataFrame(index=index)

        ts_ver = ts_ver.join(df_patients, how='outer')
        ts_hor = ts_hor.join(df_patients, how='outer')

        return ts_ver, ts_hor

    def save_timeseries(self,
                        timeseries,
                        ts_savepath,
                        pyarrow_schema=None):
        """
        Saves a table to parquet. The user may specify a pyarrow schema.
        """
        Path(ts_savepath).mkdir(exist_ok=True, parents=True)

        patient_ts = (timeseries.reset_index(drop=True)
                                .set_index('patient')
                                .groupby(level=0))

        for patient, df in patient_ts:
            self.save(df,
                      f'{ts_savepath}{patient}.parquet',
                      pyarrow_schema=pyarrow_schema)

    def _kept_meds(self):
        """convenience function to get the list of included medications."""
        return [*self.ohdsi_med.keys()]

    def _build_index(self, timeseries, cols_index, freq=3600):
        """
        observation times should be a dataframe with two columns :
            patient and time.

        It should contain the times at which values are available.
        """
        observation_times = pd.DataFrame(timeseries.index.tolist(),
                                         columns=cols_index)

        # get the timestep of the last value
        self.last_sample = (observation_times.groupby('patient')['time']
                                             .max().astype(int)
                                             .clip(lower=freq+1))

        tuples = [(i, t) for i, T in self.last_sample.items()
                  for t in np.arange(0, T, freq).astype(int)]
        self.mux = pd.MultiIndex.from_tuples(tuples, names=cols_index)

        self.df_resampler = pd.DataFrame(self.mux.get_level_values(1).values,
                                         index=self.mux,
                                         columns=['new_time'])
        return self.mux

    def generate_patient_chunks(self, unique_patients):
        """
        From a pd.Series of unique patients, creates an array of patient chunks
        of size self.n_patient_chunk.
        """
        patients = unique_patients.to_list()
        np.random.shuffle(patients)
        return np.split(patients, np.arange(self.n_patient_chunk,
                                            len(patients),
                                            self.n_patient_chunk))

    def _extract_variables(self, ts_ver, kept_variables):
        """
        ts_ver should have [patient, time] as multiindex
                           [variable, value] as columns

        The variables in kept_variables are queried in the variable column

        A dataframe created from 'index' is filled with the values.
        """
        series = []
        for var in kept_variables:
            self.var = var
            
            df = (ts_ver.loc[ts_ver['variable'] == var, 'value']
                  .groupby(level=[0, 1])
                  .agg(self.aggregates.loc[var])
                  .rename(var))
            series.append(df)
        if series:
            return pd.concat(series, axis=1)
        else:
            return pd.DataFrame(index=self.mux)

    def _apply_mask_decay(self, mask_bool, decay_rate=4/3):
        """
        This decaying mask can be used in a model to get an idea of the
        staleness of a measurement. Its value is 1 at the time of measures
        and decays towards zero as the measure gets older.
        """
        mask = mask_bool.astype(int).replace({0: np.nan})
        inv_mask_bool = ~mask_bool
        count_non_measurements = inv_mask_bool.cumsum() \
            - inv_mask_bool.cumsum().where(mask_bool).ffill().fillna(0)
        decay_mask = (mask.ffill().fillna(0)
                      / (count_non_measurements * decay_rate).replace(0, 1))
        return decay_mask

    def _mask(self, data):
        """
        Apply a decaying mask for all timeseries variables.
        """
        mask = (data.notna()
                    .groupby('patient')
                    .apply(self._apply_mask_decay)
                    .add_suffix('_mask')
                    .droplevel(0))
        return pd.concat([data, mask], axis=1)

    def _add_hour(self, timeseries, admission_hours):
        '''
        timeseries : index should be multiindex(patient, time)

        admission_hours : pd.Series :
                                index shoud be patient,
                                value shoud be the 'hour', the admission hour.
        if no admission hours are given, it is filled with 0.
        It also creates a 'hour' variable showing the hour at each timestamp. 
        In the current version, this variable is not included of the BlendedICU
        dataset because the admission hours are not specified in several
        source databases.
        '''
        hours_since_admission = timeseries.index.get_level_values(1)

        if admission_hours is None:
            # Hour is artificially added.
            timeseries['hour_admitted'] = 0
        else:
            admission_hours = (admission_hours.fillna(0)
                                              .add_suffix('_admitted'))
            timeseries = timeseries.merge(admission_hours,
                                          left_on='patient',
                                          right_index=True)

        timeseries['hour'] = ((timeseries['hour_admitted']+hours_since_admission) % 24)/24

        return timeseries.drop(columns='hour_admitted')
