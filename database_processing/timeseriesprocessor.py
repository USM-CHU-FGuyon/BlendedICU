"""
This class is common to the four source databases and to the BlendedICU 
dataset.
It contains useful functions to convert the format of data, harmonize units and 
labels and save the timeseries variables as parquet files.
"""
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import numpy as np
from natsort import natsorted

from database_processing.dataprocessor import DataProcessor


class TimeseriesProcessor(DataProcessor):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.ts_variables_pth = self.user_input_pth+'timeseries_variables.csv'
        self.ts_variables = self._load_tsvariables()

        self.kept_ts = self.ts_variables[self.dataset].dropna().to_list()

        self.kept_med = self._kept_meds()
        self.index = None

        d = ({'time':{'blended':'time',
                      'eicu': 'time',
                      'mimic4':'time',
                      'mimic3': 'time',
                      'amsterdam': 'time',
                      'hirid': 'time',
                      'is_numeric':1,
                      'agg_method': 'last',
                      'unit_concept_id': 8505}, 
             'hour':{'blended':'hour',
                     'eicu': 'hour',
                     'mimic4':'hour',
                     'mimic3': 'hour',
                     'amsterdam': 'hour',
                     'hirid': 'hour',
                     'is_numeric':1,
                     'agg_method': 'last',
                     'unit_concept_id': 8505}}
             | {med: {'blended':med,
                      'eicu': med,
                      'mimic4':med,
                      'mimic3': med,
                      'amsterdam': med,
                      'hirid': med,
                      'is_numeric':1,
                      'agg_method':'last',
                      'categories':'drug',
                      'unit_concept_id': 0}
                for med in self.kept_med})
        self.med_time_hour = pd.DataFrame(d).T
        
        self.cols = self._cols()

        self.col_mapping = self.cols.set_index(self.dataset, drop=False)['blended'].to_dict()
        self.omop_mapping = self.cols['hirid'].to_dict()
        self.aggregates_blended = pd.concat([self.cols['agg_method'], pd.Series({'patient': 'first'})])
        self.aggregates = self.cols.set_index(self.dataset)['agg_method']
        self.dtypes = self._get_dtypes()
        self.is_measured = self.ts_variables.set_index('blended').notna()
        self.numeric_ts = self._get_numeric_cols()
        self.cols_minmax = self._get_cols_minmax()
        self.pyarrow_schema_dict = self._pyarrow_schema_dict()
        self.column_template = self._column_template()
        
        self.pl_harmonizer = {
            'amsterdam': self._pl_harmonize_amsterdam,
            'eicu': self._pl_harmonize_eicu,
            'hirid': self._pl_harmonize_hirid,
            'mimic4': self._pl_harmonize_mimic4,
            'mimic3': self._pl_harmonize_mimic3,
            'blended': (lambda x: x)
            }[self.dataset]

    def _get_cols_minmax(self):
        df = self.cols.dropna(subset=self.dataset).set_index(self.dataset)
        df = df.loc[df.is_numeric.astype(bool), ['user_min', 'user_max']].replace(np.nan, None)
        df = df.to_dict('index')
        return df
        
    @staticmethod
    def collect_if_lazy(df):
        if isinstance(df, pl.lazyframe.frame.LazyFrame):
            df = df.collect()
        return df.to_pandas()
    
    
    def newprocess_tables(self,
                         timeseries,
                         med=None,
                         admission_hours=None,
                         stop_at_first_chunk=False,
                         chunk_number=None):
        import time
        t0 = time.time()
        print('filtering...', end=" ")
        
        timeseries = (timeseries
                      .filter(
                          (pl.col(self.time_col)>=0),
                          )
                      )

        timeseries = self.collect_if_lazy(timeseries)
        med = self.collect_if_lazy(med)
        
        timeseries = timeseries.set_index([self.idx_col_int, self.time_col])
        self.ts = timeseries

        print('done.', time.time()-t0)
        print('build index', end=' ')
        self.index = self._build_index(timeseries,
                                       cols_index=[self.idx_col_int, self.time_col])
 
        self.index_start1 = self.index[self.index.get_level_values(1) > 0]
        print('done', time.time()-t0)

        self.ts = timeseries

        dropcols = [c for c in timeseries.columns if c not in list(self.cols['blended'].values) + [self.idx_col]]
        print('resampling', end=' ')
        
        #TODO : replace resampling using cross and outer join coalesce
        #make index using polars 
        #llf.join(idx, on=['patient_int', 'time'], how='outer_coalesce')
        self.chunk = (timeseries
                      .drop(columns=dropcols)
                      .pipe(self._resampling)
                      .pipe(self._add_hour,
                            admission_hours=admission_hours)
                      .reindex(self.index_start1)
                      .reset_index())
        print('done', time.time()-t0)
        self.chunk = (pd.concat([self.chunk, self.column_template])
                      .astype(self.dtypes))

        if stop_at_first_chunk:
            raise Exception('Stopped after 1st chunk, deactivate '
                            '"stop_at_first_chunk" to keep running.')
        ts_savepath = f'{self.partiallyprocessed_ts_dir}/{self.dataset}_{chunk_number}.parquet'
        self.save(self.chunk, ts_savepath)

    def harmonize_columns(self,
                              table,
                              col_id=None,
                              col_time=None,
                              col_var=None,
                              col_value=None):
        
        mapping = {k: v for k, v in {col_id: self.idx_col_int,
                                     col_var: self.col_variable,
                                     col_time: self.time_col,
                                     col_value: self.col_value}.items()
                   if k is not None}
        
        return table.rename(mapping)


    def add_prefixed_pid(self, lf):
        lf = lf.with_columns(
            (self.dataset+'-'+pl.col(self.idx_col_int).cast(pl.String)).alias(self.idx_col)
            )
        return lf
            

    def filter_tables(self,
                      table,
                      kept_variables=None,
                      kept_stays=None
                      ):
        """
        This function harmonizes the column names in the input table.
        It produces a unique stay id for the blendedICU dataset by appending
        the name of the source database to the original stay ids.
        if `kept_variable` is specified, it also filters a set of variables.
        """
        
        def _filters(kept_variables, kept_stays):
            
            filters = []
            
            if kept_variables is not None:
                filters.append(pl.col(self.col_variable).is_in(kept_variables))
            if kept_stays is not None:
                filters.append(pl.col(self.idx_col_int).is_in(kept_stays))
            return filters
        
        table = (table
                 .filter(_filters(kept_variables, kept_stays))
                 )

        return table

    @staticmethod
    def _set_patient_time_index(ts, cols_index):
        """
        'time' colum is a number of seconds. This function rounds the 'time'
        column to an int buts keeps a float dtype. This avoids unnecessary
        precision on certain measurements (especially for HiRID).
        Then sets patient and time columns as index.
        """
        ts['time'] = ts['time'].round()
        ts = ts.set_index(cols_index)
        return ts
        
    def _pl_get_variables_from_long_format(self, lf_tsver):
        ver_variables = (lf_tsver
                         .select(pl.col('variable').unique())
                         .filter(pl.col('variable').is_in(self.cols[self.dataset]))
                         .collect().to_numpy().flatten()
                         .tolist())
        return ver_variables
    
    def _get_variables_from_long_format(self, ts_ver):
        """
        Returns the list of variables that are found in ts_ver AND in 
        the included timeseries variables listed in self.cols.
        """
        ver_variables = (ts_ver['variable']
                         .drop_duplicates()
                         .reset_index(drop=True))
        ver_variables = ver_variables.loc[ver_variables.isin(self.cols[self.dataset])]
        return ver_variables
    
    @staticmethod
    def _get_unique_patients(ts_ver, ts_hor, med):
        """
        Returns all patients that are found in either long format or wide
        format timeseries or medication data.
        """
        ver_patients = ts_ver.index.get_level_values(0)
        hor_patients = ts_hor.index.get_level_values(0)
        med_patients = med.patient

        unique_patients = (ver_patients.union(hor_patients)
                                       .union(med_patients)
                                       .drop_duplicates()
                                       .sort_values())
        return unique_patients
    
    
    def pl_format_meds(self, med):
        med = (med
               .pipe(self.add_prefixed_pid)
               )
        med_savepath = self.formatted_med_dir + self.dataset + ".parquet"
        self.save(med, med_savepath)
        return med
    
    def pl_format_timeseries(self,
                             lf_tsver=None,
                             lf_tshor=None,
                             chunk_number=None):
        
        '''There is a much better way to do what format_raw_data does.'''
        
        cols_index = {self.idx_col_int: int, self.time_col: int}
        pl_cols_index = [pl.col(self.idx_col_int), pl.col(self.time_col)]
        if lf_tshor is None:
            lf_tshor = pl.LazyFrame(schema=cols_index|{self.idx_col: str})
        if lf_tsver is None:
            lf_tsver = pl.LazyFrame(schema=cols_index | {'variable':str, 'value': str, self.idx_col: str})

        lf_tsver_pivoted = self.pl_lazypivot(lf_tsver,
                                             index=pl_cols_index,
                                             columns=pl.col('variable'),
                                             values=pl.col('value'),
                                             unique_column_names=self.kept_ts)

        variables = set(lf_tsver_pivoted.columns).union(set(lf_tshor.columns)) - {self.idx_col, self.time_col}

        colsminmax = {k: v for k, v in self.cols_minmax.items() if k in variables}

        df_ts = (lf_tsver_pivoted
                  .join(lf_tshor, on=pl_cols_index, how='outer_coalesce')
                  .with_columns(
                      [pl.col(variable).clip(lower_bound=colsminmax[variable]['user_min'],
                                             upper_bound=colsminmax[variable]['user_max'])
                       for variable in colsminmax]
                      )
                  .rename({old: new for old, new in self.col_mapping.items() if old in variables})
                  .pipe(self.pl_harmonizer)
                  .pipe(self._pl_compute_GCS)
                  .pipe(self.add_prefixed_pid)
                  )

        if chunk_number is None:
            ts_savepath = self.formatted_ts_dir + self.dataset + ".parquet"
        else :
            print('Collecting...')
            df_ts = df_ts.collect(streaming=True)
            ts_savepath = self.formatted_ts_dir + self.dataset + f"_{chunk_number}.parquet"
            
        self.save(df_ts, ts_savepath)
        return df_ts

    
    @staticmethod
    def pl_lazypivot(lf, index, columns, values, unique_column_names):

        lf = (lf
              .group_by(index)
              .agg(
                     values.filter(columns==col).mean().alias(col) for col in unique_column_names
                  )
              )
        
        return lf
        

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

    def _column_template(self, meds=False):
        """
        create an empty DataFrame with a column for each variable. This 
        ensures that all chunks have the same columns, even if 
        a medication or timeseries variables was not found in some chunks.
        """
        cols = self.cols if meds else self.cols.loc[self.cols.categories!='drug']
        cols = pd.concat([cols['blended'], pd.Series(['patient'])])
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


    def _pl_harmonize_hirid(self, lf):
        lf = lf.with_columns(
            pl.col('hemoglobin').mul(10)
            )
        return lf

    def _pl_harmonize_amsterdam(self, lf):
        """
        Conversion constants were taken from:
        https://www.wiv-isp.be/qml/uniformisation-units/conversion-table/tableconvertion-chimie.pdf
        """
        # The numericitems table indicates that all
        # tidal volume settings are in mL.
        # however there are inconsistent units in the data.
        lf = lf.with_columns(
            pl.col('O2_arterial_saturation').mul(100),
            pl.col('hemoglobin').mul(1.613),
            (pl.when(pl.col('tidal_volume_setting')<1)
             .then(pl.col('tidal_volume_setting').mul(1000))
             .otherwise(pl.col('tidal_volume_setting'))),
            (pl.when((pl.col('tidal_volume_setting')>1) 
                     | (pl.col('tidal_volume_setting')<3)  )
             .then(pl.lit(None))
             .otherwise(pl.col('tidal_volume_setting')))
            )
        return lf


    def _pl_harmonize_eicu(self, lf):

        lf = lf.with_columns(
            pl.when(pl.col('temperature')>80)
            .then((pl.col('temperature')-32).truediv(1.8))
            .otherwise(pl.col('temperature'))
            .alias('temperature'),
            pl.col('calcium').mul(0.25),
            pl.col('magnesium').mul(0.411),
            pl.col('blood_glucose').mul(0.0555),
            pl.col('creatinine').mul(88.40),
            pl.col('bilirubine').mul(17.39),
            pl.col('albumin').mul(10),
            pl.col('blood_urea_nitrogen').mul(0.357),
            pl.col('phosphate').mul(0.323)
            )

        return lf

    def _pl_harmonize_mimic3(self, lf):
        lf = lf.with_columns(
            (pl.col('temperature')-32).truediv(1.8),
            pl.col('blood_glucose').mul(0.0555),
            pl.col('magnesium').mul(0.411),
            pl.col('creatinine').mul(88.40),
            pl.col('calcium').mul(0.25),
            pl.col('bilirubine').mul(17.39),
            pl.col('albumin').mul(10.),
            pl.col('blood_urea_nitrogen').mul(0.357),
            pl.col('phosphate').mul(0.323))  
        return lf

    def _pl_harmonize_mimic4(self, lf):
        lf = lf.with_columns(
            (pl.col('temperature')-32).truediv(1.8),
            pl.col('blood_glucose').mul(0.0555),
            pl.col('magnesium').mul(0.411),
            pl.col('creatinine').mul(88.40),
            pl.col('calcium').mul(0.25),
            pl.col('bilirubine').mul(17.39),
            pl.col('albumin').mul(10.),
            pl.col('blood_urea_nitrogen').mul(0.357),
            pl.col('phosphate').mul(0.323))        
        
        return lf

    def _resampling(self, df):
        """
        Resamples the input dataframe and applies the aggregate functions 
        that are specified in the input timeseries_variables.csv file.
        """
        
        aggregates = self.aggregates_blended.loc[df.columns]

        df = self.df_resampler.join(df, how='outer')

        df['new_time'] = df.groupby(level=0).new_time.ffill()
        
        df = (df.droplevel(self.time_col)
                .rename(columns={'new_time': self.time_col})
                .set_index(self.time_col, append=True)
                .groupby(level=[self.idx_col_int, self.time_col])
                .agg(aggregates)
                )
        if self.idx_col in df.columns:
            df[self.idx_col] = df[self.idx_col].bfill()
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


    def _pl_compute_GCS(self, lf):
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

        if 'glasgow_coma_score' not in lf.columns:
            
            lf = (lf
                  .with_columns(
                      pl.when(~pl.all_horizontal(glasgow_cols).is_null())
                      .then(pl.sum_horizontal(glasgow_cols))
                      .otherwise(pl.lit(None))
                      .alias('glasgow_coma_score')
                      )
                  )
        
        lf = lf.with_columns(
                [pl.lit(None).alias(col) for col in glasgow_cols if col not in lf.columns]
                )
        return lf 

    def _forward_fill(self, df):
        """
        Forward filling is optional. If FORWARD_FILL is set to 0 in the 
        config.json file, the tables will be resampled but missing values will
        be preserved.
        """
        if self.FORWARD_FILL:
            return df.groupby('patient').ffill()
        return df


    def save_timeseries(self,
                        timeseries,
                        ts_savepath,
                        pyarrow_schema=None):
        """
        Saves a table to parquet. The user may specify a pyarrow schema.
        """
        def _save_index(patient_pths, ts_savepath):
            index_savepath = ts_savepath+'/index.csv'
            (pd.Series(patient_pths, name='ts_pth')
             .rename_axis('patient')
             .to_csv(index_savepath, sep=';'))
            print(f'  -> saved {index_savepath}')
            
        Path(ts_savepath).mkdir(exist_ok=True, parents=True)
        print(ts_savepath)
        patient_ts = (timeseries.reset_index(drop=True)
                                .set_index('patient')
                                .groupby(level=0))
        patient_pths = {}
        for patient, df in patient_ts:
            patient_pths[patient] = f'{ts_savepath}{patient}.parquet'
            self.save(df,
                      patient_pths[patient],
                      pyarrow_schema=pyarrow_schema)
        self._save_index(patient_pths, ts_savepath)


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
        self.last_sample = (observation_times.groupby(self.idx_col_int)['time']
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

    def _extract_variables(self, ts_ver, kept_variables, join_index=(0,1)):
        """
        ts_ver should have [patient, time] as multiindex
                           [variable, value] as columns

        The variables in kept_variables are queried in the variable column

        A dataframe created from 'index' is filled with the values.
        """
        series = []
        for var in kept_variables:
            self.var = var
            
            df = (ts_ver.loc[ts_ver['variable']==var, 'value']
                  .groupby(level=join_index)
                  .agg(self.aggregates.loc[var])
                  .rename(var))
            series.append(df)
        if series:
            return pd.concat(series, axis=1)
        else:
            return pd.DataFrame(index=self.mux)

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

    def get_stay_chunks(self, add_prefix=False, n_patient_chunk=None):
        n_patients = self.n_patient_chunk if n_patient_chunk is None else n_patient_chunk
       
        stays = (np.array([f'{self.dataset}-{s}' for s in self.stays])
                 if add_prefix
                 else self.stays)
        return np.array_split(stays, self.stays.size//n_patients)
        