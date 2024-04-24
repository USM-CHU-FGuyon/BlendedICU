from functools import partial
from pathlib import Path

import pandas as pd
import polars as pl

from database_processing.medicationprocessor import MedicationProcessor
from database_processing.datapreparator import DataPreparator

class mimic3Preparator(DataPreparator):
    def __init__(self,
                 chartevents_pth,):
        super().__init__(dataset='mimic3', col_stayid='ICUSTAY_ID')
        self.chartevents_pth = self.source_pth+chartevents_pth
        self.labevents_pth = f'{self.savepath}labevents.parquet'
        self.outputevents_pth = f'{self.savepath}outputevents.parquet'
        self.admissions_pth = f'{self.savepath}admissions.parquet'
        self.inputevents_mv_pth = f'{self.savepath}inputevents_mv.parquet'
        self.inputevents_cv_pth = f'{self.savepath}inputevents_cv.parquet'
        self.icustays_pth = f'{self.savepath}icustays.parquet'
        self.patients_pth = f'{self.savepath}patients.parquet'
        self.ditems_pth = f'{self.savepath}d_items.parquet'
        self.dlabitems_pth = f'{self.savepath}d_labitems.parquet'
        self.tslab_savepath = f'{self.savepath}/timeserieslab.parquet'
        self.ts_savepath = f'{self.savepath}/timeseries/'
        self.outputevents_savepath = f'{self.savepath}/timeseriesoutputs.parquet'
        self.col_los = 'LOS'
        self.unit_los = 'day'
        
    def gen_icustays(self):
        admissions = pl.scan_parquet(self.admissions_pth)
        icustays = pl.scan_parquet(self.icustays_pth)
        
        df_icustays = (icustays
                       .with_columns(
                           pl.col('INTIME').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                       )
                       .join(admissions.select('HADM_ID',
                                               'ETHNICITY',
                                               'ADMISSION_LOCATION',
                                               'INSURANCE',
                                               'DISCHARGE_LOCATION',
                                               'HOSPITAL_EXPIRE_FLAG'),
                             on='HADM_ID',
                             how='left')
                       .collect())
        return df_icustays
    
    
    def load_raw_tables(self):
        """
        To make processing faster, a set of tables are converted to .parquet 
        It is only useful to run this function once.
        """
        tables = [
            'LABEVENTS',
            'ADMISSIONS',
            'D_ITEMS',
            'D_LABITEMS',
            'INPUTEVENTS_CV',
            'INPUTEVENTS_MV',
            'OUTPUTEVENTS',
            'ICUSTAYS',
            'PATIENTS',
        ]

        for table in tables:
            print(table)
            table_pth = Path(table)
            pth_csv = f'{self.source_pth}/{table_pth}'
            pth_pqt = f'{self.savepath}/{table_pth.name}'
            self.save(pd.read_csv(f'{pth_csv}.csv.gz'), f'{pth_pqt}.parquet')

    def _fetch_heights_weights(self):
        """
        admission heights and weights, available at self.flat_hr_from_adm
        are fetched to fill the flat and labels tables.
        They can be found under several itemids depending on the unit in 
        which they are measured. Every value is converted to the metric system.
        """
        print('Fetching heights and weights in the chartevents table...'
              'this may take several minutes.')
        icustays = self.icustays.to_pandas()
        itemids = {'weight_kg_2': 224639,
                   'weight_kg': 226512,
                   'weight_lbs': 226531,
                   'height_inch': 226707,
                   'height_cm': 226730}

        keepids = [*itemids.values()]

        chartevents = pd.read_csv(self.chartevents_pth,
                                  chunksize=self.chunksize,
                                  usecols=['ICUSTAY_ID',
                                           'ITEMID',
                                           'VALUENUM',
                                           'CHARTTIME'])

        dfs_hw = []
        for i, df in enumerate(chartevents):
            print(f'Read {(i+1)*self.chunksize} lines from chartevents table...')
            df = df.merge(icustays[['ICUSTAY_ID', 'INTIME']], on='ICUSTAY_ID')

            df['measuretime'] = ((pd.to_datetime(df['CHARTTIME'])
                                 - pd.to_datetime(df['INTIME']))
                                 .astype('timedelta64[s]'))

            df = df.loc[(df.ITEMID.isin(keepids))
                        & (df.measuretime < self.flat_hr_from_adm)]

            dfs_hw.append(df.drop(columns=['measuretime', 'INTIME']))

        df_hw = pd.concat(dfs_hw)

        inch_idx = df_hw.ITEMID == itemids['height_inch']
        lbs_idx = df_hw.ITEMID == itemids['weight_lbs']
        df_hw.loc[inch_idx, 'VALUENUM'] *= self.inch_to_cm
        df_hw.loc[lbs_idx, 'VALUENUM'] *= self.lbs_to_kg

        heights = df_hw.loc[df_hw.ITEMID.isin([itemids['height_inch'],
                                               itemids['height_cm']]),
                            ['ICUSTAY_ID', 'VALUENUM']]
        weights = df_hw.loc[df_hw.ITEMID.isin([itemids['weight_kg_2'],
                                               itemids['weight_kg'],
                                               itemids['weight_lbs']]),
                            ['ICUSTAY_ID', 'VALUENUM']]

        heights = (heights.rename(columns={'VALUENUM': 'height'})
                          .groupby('ICUSTAY_ID')
                          .mean())
        weights = (weights.rename(columns={'VALUENUM': 'weight'})
                          .groupby('ICUSTAY_ID')
                          .mean())
        return heights, weights

    def gen_flat(self):
        """
        TODO : to polars.
        """
        icustays = self.icustays.to_pandas()
        print('o Flat Features')
        patients = pd.read_parquet(self.patients_pth,
                                   columns=['SUBJECT_ID',
                                            'GENDER',
                                            'DOB'])

        self.heights, self.weights = self._fetch_heights_weights()

        df_flat = (icustays
                           .merge(patients, on='SUBJECT_ID', how='left')
                           .merge(self.heights, on='ICUSTAY_ID', how='left')
                           .merge(self.weights, on='ICUSTAY_ID', how='left')
                           .sort_values('ICUSTAY_ID')
                           .rename(columns={'ANCHOR_AGE': 'age'}))
        self.df_flat = df_flat
        df_flat['hour'] = pd.to_datetime(df_flat['INTIME']).dt.hour

        df_flat = df_flat.drop(columns=['SUBJECT_ID',
                                        'HADM_ID',
                                        'LAST_CAREUNIT',
                                        'INTIME',
                                        'OUTTIME',
                                        'LOS'])

        return self.save(df_flat, self.flat_savepath)


    def gen_labels(self):
        print('o Labels')

        icustays = self.icustays.lazy()

        hospital_mortality = (icustays
                              .group_by("HADM_ID")
                              .agg(pl.col("HOSPITAL_EXPIRE_FLAG").max()))

        self.labels = (icustays.select('SUBJECT_ID',
                                             'HADM_ID',
                                             'ICUSTAY_ID',
                                             'LOS',
                                             'INTIME',
                                             'DISCHARGE_LOCATION',
                                             'FIRST_CAREUNIT')
                        .join(hospital_mortality, on='HADM_ID')
                        .with_columns(
                            pl.lit('Beth Israel Deaconess Medical Center').alias('care_site')
                            )
                        .sort('ICUSTAY_ID'))
        
        labels = self.labels.collect()
        self.save(labels, self.labels_savepath)

    def _load_inputevents(self):

        inputevents_mv = (pl.scan_parquet(self.inputevents_mv_pth)
                          .select('ICUSTAY_ID',
                                'STARTTIME',
                                'ITEMID')
                          .rename({'STARTTIME': 'CHARTTIME'}))
        
        inputevents_cv = (pl.scan_parquet(self.inputevents_cv_pth)
                          .select('ICUSTAY_ID',
                                'CHARTTIME',
                                'ITEMID'))
        
        d_items = pl.scan_parquet(self.ditems_pth)

        inputevents = pl.concat([inputevents_cv, inputevents_mv])

        df_inputevents = (inputevents
                          .join(d_items.select('ITEMID', 'LABEL'), on='ITEMID')
                          .drop('ITEMID')
                          .rename({'CHARTTIME': 'time'})
                          .drop_nulls('ICUSTAY_ID')
                          .with_columns(
                              pl.col('ICUSTAY_ID').cast(pl.Int32())
                              )
                          )
        
        return df_inputevents


    def gen_medication(self):
        """
        Medication can be found in the inputevents table.
        """
        inputevents = self._load_inputevents().collect().to_pandas()

        icustays = pd.read_parquet(self.icustays_pth,
                                   columns=['ICUSTAY_ID', 'INTIME', 'LOS'])

        self.mp = MedicationProcessor('mimic3',
                                      icustays,
                                      col_pid='ICUSTAY_ID',
                                      col_los='LOS',
                                      col_med='LABEL',
                                      col_time='time',
                                      col_admittime='INTIME',
                                      offset_calc=True,
                                      unit_offset='second',
                                      unit_los='day')
        
        self.med = self.mp.run(inputevents)
        return self.save(self.med, self.med_savepath)
    
    
    def gen_timeseriesoutputs(self):
        self.get_labels(lazy=True)
        ditems = pl.scan_parquet(self.ditems_pth)
        
        outputevents = pl.scan_parquet(self.outputevents_pth)
        
        df_outputs = (outputevents
                      .select('HADM_ID',
                              'CHARTTIME',
                              'ITEMID',
                              'VALUE')
                      .with_columns(
                          pl.col('CHARTTIME').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                          pl.col('HADM_ID').cast(pl.Int64)
                          )
                      .pipe(self.pl_prepare_tstable,
                           col_measuretime='CHARTTIME',
                            col_intime='INTIME',
                            col_variable='ITEMID',
                            col_mergestayid='HADM_ID',
                            col_value='VALUE',
                            unit_los='day'
                            )
                      .join(ditems.select('ITEMID', 'LABEL'), on='ITEMID')
                      .drop('HADM_ID', 'ITEMID')
                      .collect()
                      )
        
        self.save(df_outputs, self.outputevents_savepath)

    
    def gen_timeserieslab(self):
        self.get_labels(lazy=True)
        dlabitems = pl.scan_parquet(self.dlabitems_pth)
        
        print('o Timeseries Lab')
        labevents = pl.scan_parquet(self.labevents_pth)
        
        self.df_lab = (labevents
                       .select('HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM')
                       .drop_nulls()
                       .with_columns(
                           pl.col('CHARTTIME').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                           pl.col('HADM_ID').cast(pl.Int64)
                           )
                       .pipe(self.pl_prepare_tstable,
                             col_measuretime='CHARTTIME',
                             col_intime='INTIME',
                             col_variable='ITEMID',
                             col_mergestayid='HADM_ID',
                             col_value='VALUENUM',
                             unit_los='day')
                       .join(dlabitems.select('ITEMID', 'LABEL'), on='ITEMID')
                       .drop(columns=['HADM_ID', 'ITEMID'])
                       .collect()
                       )
        
        self.save(self.df_lab, self.tslab_savepath)


    def gen_timeseries(self):
        '''
        Polars 0.20 does not support scanning or chunking csv.gz
        We load thorugh pandas and convert chunks to LazyFrames
        '''
        self.get_labels(lazy=True)

        ditems = pl.scan_parquet(self.ditems_pth)

        chartevents = pd.read_csv(self.chartevents_pth,
                                  chunksize=self.chunksize,
                                  usecols=['ICUSTAY_ID',
                                           'CHARTTIME',
                                           'ITEMID',
                                           'VALUENUM'])
        for i, df in enumerate(chartevents):
            
            lf = pl.LazyFrame(df)

            ts = (lf.drop_nulls()
                    .with_columns(
                        pl.col('CHARTTIME').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                        pl.col('ICUSTAY_ID').cast(pl.Int64)
                        )
                  .pipe(self.pl_prepare_tstable,
                        col_measuretime='CHARTTIME',
                        col_intime='INTIME',
                        col_variable='ITEMID',
                        col_value='VALUENUM',
                        unit_los='day')
                  .join(ditems.select('ITEMID', 'LABEL'), on='ITEMID')
                  .drop('ITEMID')
                  .collect())
                  
            self.save(ts, self.ts_savepath+f'{i}.parquet')
            
            
        
        
        
