from functools import partial
from pathlib import Path

import pandas as pd

from database_processing.medicationprocessor import MedicationProcessor
from database_processing.datapreparator import DataPreparator

class mimic3Preparator(DataPreparator):
    def __init__(self,
                 chartevents_pth,):
        super().__init__(dataset='mimic3', col_stayid='ICUSTAY_ID')
        self.chartevents_pth = self.source_pth+chartevents_pth
        self.labevents_pth = f'{self.parquet_pth}labevents.parquet'
        self.outputevents_pth = f'{self.parquet_pth}outputevents.parquet'
        self.admissions_pth = f'{self.parquet_pth}admissions.parquet'
        self.inputevents_mv_pth = f'{self.parquet_pth}inputevents_mv.parquet'
        self.inputevents_cv_pth = f'{self.parquet_pth}inputevents_cv.parquet'
        self.icustays_pth = f'{self.parquet_pth}icustays.parquet'
        self.patients_pth = f'{self.parquet_pth}patients.parquet'
        self.ditems_pth = f'{self.parquet_pth}d_items.parquet'
        self.dlabitems_pth = f'{self.parquet_pth}d_labitems.parquet'
        self.tslab_savepath = f'{self.parquet_pth}/timeserieslab.parquet'
        self.ts_savepath = f'{self.parquet_pth}/timeseries.parquet'
        self.outputevents_savepath = f'{self.parquet_pth}/timeseriesoutputs.parquet'
        self.col_los = 'LOS'
        self.unit_los = 'day'
        self.icustays = self._icustays()
        
    def _icustays(self):
        admissions = pd.read_parquet(self.admissions_pth,
                                     columns=['HADM_ID',
                                              'ETHNICITY',
                                              'ADMISSION_LOCATION',
                                              'INSURANCE',
                                              'DISCHARGE_LOCATION',
                                              'HOSPITAL_EXPIRE_FLAG'])
        
        icustays = pd.read_parquet(self.icustays_pth)
        
        return icustays.merge(admissions, on='HADM_ID', how='left')
    
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
            pth_pqt = f'{self.parquet_pth}/{table_pth.name}'
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
            df = df.merge(self.icustays[['ICUSTAY_ID', 'INTIME']], on='ICUSTAY_ID')

            df['measuretime'] = ((pd.to_datetime(df['CHARTTIME'])
                                 - pd.to_datetime(df['INTIME']))
                                 .astype('timedelta64[h]'))

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
        print('o Flat Features')
        patients = pd.read_parquet(self.patients_pth,
                                   columns=['SUBJECT_ID',
                                            'GENDER',
                                            'DOB'])

        self.heights, self.weights = self._fetch_heights_weights()

        df_flat = (self.icustays
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
        labels = self.icustays.loc[:, ['SUBJECT_ID', 'HADM_ID',
                                       'ICUSTAY_ID', 'LOS', 'INTIME',
                                       'DISCHARGE_LOCATION', 'FIRST_CAREUNIT']]

        hospital_mortality = (self.icustays.groupby('HADM_ID')
                                           .HOSPITAL_EXPIRE_FLAG.max())

        labels = labels.merge(hospital_mortality, on='HADM_ID')
        labels['care_site'] = 'Beth Israel Deaconess Medical Center'

        labels = labels.sort_values('ICUSTAY_ID')
        self.labels = labels
        self.save(labels, self.labels_savepath)


    def _load_inputevents(self):
        inputevents_mv = pd.read_parquet(self.inputevents_mv_pth,
                                      columns=['HADM_ID',
                                               'ICUSTAY_ID',
                                               'STARTTIME',
                                               'ITEMID'])
        inputevents_cv = pd.read_parquet(self.inputevents_mv_pth,
                                      columns=['HADM_ID',
                                               'ICUSTAY_ID',
                                               'STARTTIME',
                                               'ITEMID'])
        inputevents = pd.concat([inputevents_mv, inputevents_cv])

        d_items = pd.read_parquet(self.ditems_pth, columns=['ITEMID', 'LABEL'])

        return (inputevents.merge(d_items[['ITEMID', 'LABEL']], on='ITEMID')
                .drop(columns=['HADM_ID', 'ITEMID'])
                .rename(columns={'STARTTIME': 'time'})
                .dropna(subset='ICUSTAY_ID')
                .astype({'ICUSTAY_ID': int}))

    def gen_medication(self):
        """
        Medication can be found in the inputevents table.
        """
        inputevents = self._load_inputevents()

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
        """
        The output table is small enough to be processed all at once.
        """
        self.get_labels()
        ditems = self.load(self.ditems_pth, columns=['ITEMID', 'LABEL'])

        outputevents = self.load(self.outputevents_pth,
                                 columns=['HADM_ID',
                                          'CHARTTIME',
                                          'ITEMID',
                                          'VALUE'])

        df_outputs = (outputevents.pipe(self.prepare_tstable,
                                        col_offset='CHARTTIME',
                                        col_intime='INTIME',
                                        col_variable='ITEMID',
                                        col_mergestayid='HADM_ID')
                      .merge(ditems, on='ITEMID')
                      .drop(columns=['HADM_ID', 'ITEMID'])
                      .rename(columns={'CHARTTIME': 'offset'}))

        self.save(df_outputs, self.outputevents_savepath)

    def gen_timeserieslab(self):
        """
        This timeserieslab table does not fit in memory, it is processed by 
        chunks. The processed table is smaller so it is saved to a single file.
        """
        self.get_labels()

        dlabitems = pd.read_parquet(self.dlabitems_pth,
                                    columns=['LABEL', 'ITEMID'])

        print('o Timeseries Lab')
        labevents = pd.read_parquet(self.labevents_pth,
                                columns=['HADM_ID',
                                         'ITEMID',
                                         'CHARTTIME',
                                         'VALUENUM'])
        
        self.df_lab = (labevents.pipe(self.prepare_tstable,
                                      col_offset='CHARTTIME',
                                      col_intime='INTIME',
                                      col_variable='ITEMID',
                                      col_mergestayid='HADM_ID')
                         .merge(dlabitems, on='ITEMID')
                         .drop(columns=['HADM_ID', 'ITEMID'])
                         .rename(columns={'CHARTTIME': 'offset'}))

        self.save(self.df_lab, self.tslab_savepath)
        
    def gen_timeseries(self):
        """
        This timeseries table does not fit in memory, it is processed by 
        chunks. The processed table is smaller so it is saved to a single file.
        """
        self.get_labels()
        ditems = pd.read_parquet(self.ditems_pth, columns=['ITEMID', 'LABEL'])

        chartevents = pd.read_csv(self.chartevents_pth,
                                  chunksize=self.chunksize,
                                  usecols=['ICUSTAY_ID',
                                           'CHARTTIME',
                                           'ITEMID',
                                           'VALUENUM'])

        print('o Timeseries')
        prepare_tslab = partial(self.prepare_tstable,
                                col_offset='CHARTTIME',
                                col_intime='INTIME',
                                col_variable='ITEMID',
                                )

        self.df_ts = (pd.concat(map(prepare_tslab, chartevents))
                        .merge(ditems, on='ITEMID')
                        .drop(columns='ITEMID'))

        self.save(self.df_ts, self.ts_savepath)
