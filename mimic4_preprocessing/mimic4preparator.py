from functools import partial
from pathlib import Path

import pandas as pd
import polars as pl

from database_processing.medicationprocessor import MedicationProcessor
from database_processing.datapreparator import DataPreparator


class mimic4Preparator(DataPreparator):
    def __init__(self,
                 chartevents_pth,
                 labevents_pth):
        super().__init__(dataset='mimic4', col_stayid='stay_id')
        self.chartevents_pth = self.source_pth+chartevents_pth
        self.labevents_pth = self.source_pth+labevents_pth
        self.diagnoses_pth = f'{self.savepath}diagnoses_icd.parquet'
        self.outputevents_pth = f'{self.savepath}outputevents.parquet'
        self.admissions_pth = f'{self.savepath}admissions.parquet'
        self.inputevents_pth = f'{self.savepath}inputevents.parquet'
        self.icustays_pth = f'{self.savepath}icustays.parquet'
        self.patients_pth = f'{self.savepath}patients.parquet'
        self.ditems_pth = f'{self.savepath}d_items.parquet'
        self.dlabitems_pth = f'{self.savepath}d_labitems.parquet'
        self.ddiagnoses_pth = f'{self.savepath}d_icd_diagnoses.parquet'
        self.tslab_savepath = f'{self.savepath}/timeserieslab.parquet'
        self.ts_savepath = f'{self.savepath}/timeseries/'
        self.outputevents_savepath = f'{self.savepath}/timeseriesoutputs.parquet'
        self.col_los = 'los'
        self.unit_los = 'day'


    def gen_icustays(self):
        
        admissions = pl.scan_parquet(self.admissions_pth)
        icustays = pl.scan_parquet(self.icustays_pth)
        
        df_icustays = (admissions
                       .select(['hadm_id',
                                'race',
                                'admission_location',
                                'insurance',
                                'discharge_location',
                                'hospital_expire_flag'])
                       .join(icustays, on='hadm_id')
                       .with_columns(
                           pl.col('intime').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                           pl.col('outtime').str.to_datetime("%Y-%m-%d %H:%M:%S")
                           )
                       .collect())
        
        return df_icustays
        

    def load_raw_tables(self):
        """
        To make processing faster, a set of tables are converted to .parquet 
        It is only useful to run this function once.
        """
        tables = [
            'hosp/d_labitems',
            'hosp/admissions',
            'hosp/diagnoses_icd',
            'hosp/d_icd_diagnoses',
            'icu/d_items',
            'icu/inputevents',
            'icu/outputevents',
            'icu/icustays',
            'hosp/patients',
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
        
        TODO : convert to polars
        
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
                                  usecols=['stay_id',
                                           'itemid',
                                           'valuenum',
                                           'charttime'])

        dfs_hw = []
        for i, df in enumerate(chartevents):
            print(f'Read {(i+1)*self.chunksize} lines from chartevents table...')
            df = df.merge(icustays[['stay_id', 'intime']], on='stay_id')
                    
            df['measuretime'] = ((pd.to_datetime(df['charttime'])
                                 - pd.to_datetime(df['intime']))
                                 .astype('timedelta64[s]'))

            df = df.loc[(df.itemid.isin(keepids))
                        & (df.measuretime < self.flat_hr_from_adm)]

            dfs_hw.append(df.drop(columns=['measuretime', 'intime']))

        df_hw = pd.concat(dfs_hw)

        inch_idx = df_hw.itemid == itemids['height_inch']
        lbs_idx = df_hw.itemid == itemids['weight_lbs']
        df_hw.loc[inch_idx, 'valuenum'] *= self.inch_to_cm
        df_hw.loc[lbs_idx, 'valuenum'] *= self.lbs_to_kg

        heights = df_hw.loc[df_hw.itemid.isin([itemids['height_inch'],
                                               itemids['height_cm']]),
                            ['stay_id', 'valuenum']]
        weights = df_hw.loc[df_hw.itemid.isin([itemids['weight_kg_2'],
                                               itemids['weight_kg'],
                                               itemids['weight_lbs']]),
                            ['stay_id', 'valuenum']]

        heights = (heights.rename(columns={'valuenum': 'height'})
                          .groupby('stay_id')
                          .mean())
        weights = (weights.rename(columns={'valuenum': 'weight'})
                          .groupby('stay_id')
                          .mean())
        return heights, weights

    def gen_flat(self):
        print('o Flat Features')
        icustays = self.icustays.to_pandas()
        patients = pd.read_parquet(self.patients_pth,
                                   columns=['subject_id',
                                            'gender',
                                            'anchor_age'])

        self.heights, self.weights = self._fetch_heights_weights()

        df_flat = (icustays
                           .merge(patients, on='subject_id', how='left')
                           .merge(self.heights, on='stay_id', how='left')
                           .merge(self.weights, on='stay_id', how='left')
                           .sort_values('stay_id')
                           .rename(columns={'anchor_age': 'age'}))
        self.df_flat = df_flat
        df_flat['hour'] = pd.to_datetime(df_flat.intime).dt.hour

        df_flat = df_flat.drop(columns=['subject_id',
                                        'hadm_id',
                                        'last_careunit',
                                        'intime',
                                        'outtime',
                                        'los'])

        return self.save(df_flat, self.flat_savepath)

    def _load_inputevents(self):
        inputevents = pl.scan_parquet(self.inputevents_pth)
        d_items = pl.scan_parquet(self.ditems_pth)
        
        df_inputevents = (inputevents
                          .select(['stay_id',
                                   'starttime',
                                   'itemid'])
                          .join(d_items.select(['itemid', 'label']), on='itemid')
                          .drop('itemid')
                          .rename({'starttime': 'time'})
                          .collect())
        return df_inputevents
        
    def gen_diagnoses(self):
        icustays = self.icustays.lazy().select('stay_id', 'hadm_id')
        diagnoses = pl.scan_parquet(self.diagnoses_pth)
        d_diagnoses = pl.scan_parquet(self.ddiagnoses_pth)
    
        df_diagnoses = (diagnoses
                        .join(icustays,
                              left_on='hadm_id',
                              right_on='hadm_id')
                        .select(['subject_id',
                                 'stay_id',
                                 'icd_code',
                                 'icd_version'])
                        .join(d_diagnoses.select(['icd_code',
                                                  'icd_version',
                                                  'long_title']),
                              on=['icd_code', 'icd_version'])
                        .collect())
        return self.save(df_diagnoses, self.diag_savepath)
    
        
    def gen_medication(self):
        """
        Medication can be found in the inputevents table.
        """
        inputevents = self._load_inputevents().to_pandas()
        icustays = pd.read_parquet(self.icustays_pth,
                                   columns=['stay_id', 'intime', 'los'])

        self.mp = MedicationProcessor('mimic4',
                                      icustays,
                                      col_pid='stay_id',
                                      col_los='los',
                                      col_med='label',
                                      col_time='time',
                                      col_admittime='intime',
                                      offset_calc=True,
                                      unit_offset='second',
                                      unit_los='day')
        self.med = self.mp.run(inputevents)
        return self.save(self.med, self.med_savepath)

    def gen_labels(self):
        print('o Labels')
        
        icustays = self.icustays.lazy()

        hospital_mortality = (icustays
                              .group_by("hadm_id")
                              .agg(pl.col("hospital_expire_flag").max()))
        
        self.labels = (icustays.select('subject_id', 'hadm_id',
                                       'stay_id', 'los', 'intime',
                                       'discharge_location', 'first_careunit')
                       
                       .join(hospital_mortality, on='hadm_id')
                        .with_columns(
                            care_site=pl.lit('Beth Israel Deaconess Medical Center')
                            )
                        .sort('stay_id')
                        .collect())

        self.save(self.labels, self.labels_savepath)

    def gen_timeseriesoutputs(self):
        """
        The output table is small enough to be processed all at once.
        """
        self.get_labels(lazy=True)
        ditems = pl.scan_parquet(self.ditems_pth)
        
        outputevents = pl.scan_parquet(self.outputevents_pth)
        
        df_outputs = (outputevents
                      .select('hadm_id',
                               'charttime',
                               'itemid',
                               'value')
                      .with_columns(
                          pl.col('charttime').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                          pl.col('hadm_id').cast(pl.Int64)
                          )
                      .pipe(self.pl_prepare_tstable,
                            col_measuretime='charttime',
                            col_intime='intime',
                            col_variable='itemid',
                            col_mergestayid='hadm_id',
                            unit_los='day',
                            col_value='value'
                            )
                      .join(ditems.select('itemid', 'label'), on='itemid')
                      .drop('hadm_id', 'itemid')
                      .collect()
                      )

        self.save(df_outputs, self.outputevents_savepath)

    def gen_timeserieslab(self):
        """
        This timeserieslab table does not fit in memory, it is processed by 
        chunks. The processed table is smaller so it is saved to a single file.
        """
        self.get_labels(lazy=True)

        dlabitems = pl.scan_parquet(self.dlabitems_pth)
        
        print('o Timeseries Lab')
        labevents = pl.read_csv(self.labevents_pth,
                                columns=['hadm_id',
                                         'itemid',
                                         'charttime',
                                         'valuenum']).lazy()

        keepvars = self._lab_keepvars()

        keepitemids = dlabitems.filter(pl.col('label').is_in(keepvars)).select('itemid').collect()

        self.df_lab = (labevents
                       .select('hadm_id', 'itemid', 'charttime', 'valuenum')
                       .drop_nulls()
                       .with_columns(
                           pl.col('charttime').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                           pl.col('hadm_id').cast(pl.Int64)
                           )
                       .pipe(self.pl_prepare_tstable,
                             keepvars=keepitemids,
                             col_measuretime='charttime',
                             col_intime='intime',
                             col_variable='itemid',
                             col_mergestayid='hadm_id',
                             unit_los='day',
                             col_value='valuenum')
                       .join(dlabitems.select('itemid', 'label'), on='itemid')
                       .drop(['hadm_id', 'itemid'])
                       .collect()
                       )

        self.save(self.df_lab, self.tslab_savepath)

    def gen_timeseries(self):
        self.get_labels(lazy=True)
        ditems = pl.scan_parquet(self.ditems_pth)

        chartevents = pd.read_csv(self.chartevents_pth,
                                  chunksize=self.chunksize,
                                  usecols=['stay_id',
                                           'charttime',
                                           'itemid',
                                           'valuenum'])

        print('o Timeseries')
        keepvars = self._timeseries_keepvars()

        keepitemids = ditems.filter(pl.col('label').is_in(keepvars)).select('itemid').collect().to_numpy().flatten()

        for i, df in enumerate(chartevents):
            
            lf = pl.LazyFrame(df)
            
            ts = (lf.drop_nulls()
                    .with_columns(
                        pl.col('charttime').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                        pl.col('stay_id').cast(pl.Int64)
                        )
                  .pipe(self.pl_prepare_tstable,
                        keepvars=keepitemids,
                        col_measuretime='charttime',
                        col_intime='intime',
                        col_variable='itemid',
                        col_value='valuenum',
                        unit_los='day')
                  .join(ditems.select('itemid', 'label'), on='itemid')
                  .drop('itemid')
                  .collect())
                  
            self.save(ts, self.ts_savepath+f'{i}.parquet')

    @staticmethod
    def _lab_keepvars():
        return ['MCV', 'Phosphate', 'Hemoglobin', 'PTT', 'Platelet Count',
                    'RDW',
                    'Red Blood Cells', 'Magnesium', 'Creatinine',
                    'White Blood Cells',
                    'RDW-SD', 'INR(PT)', 'PT', 'Chloride', 'MCH',
                    'Calcium, Total',
                    'Bicarbonate', 'Anion Gap', 'Hematocrit', 'Potassium',
                    'Urea Nitrogen', 'Sodium', 'MCHC', 'Glucose', 'pCO2',
                    'Base Excess', 'Potassium, Whole Blood', 'pO2', 'pH',
                    'Calculated Total CO2', 'Sodium, Whole Blood', 'Lactate',
                    'Hematocrit, Calculated', 'Free Calcium',
                    'Oxygen Saturation',
                    'Temperature', 'Alanine Aminotransferase (ALT)',
                    'Alkaline Phosphatase', 'Asparate Aminotransferase (AST)',
                    'Bilirubin, Total', 'H', 'I', 'L', 'Albumin']
    
    @staticmethod
    def _timeseries_keepvars():
        return ['Hemoglobin', 'Potassium (whole blood)',
                    'Glucose (whole blood)',
                    'Lactic Acid', 'Ionized Calcium', 'WBC',
                    'Hematocrit (serum)',
                    'Platelet Count', 'PH (Arterial)', 'Arterial O2 pressure',
                    'Arterial CO2 Pressure', 'Arterial Base Excess',
                    'TCO2 (calc) Arterial', 'Fspn High', 'Plateau Pressure',
                    'Total PEEP Level', 'Ventilator Tank #1',
                    'Ventilator Tank #2',
                    'Ventilator Type', 'O2 saturation pulseoxymetry',
                    'Vti High',
                    'Apnea Interval', 'Tidal Volume (set)',
                    'Tidal Volume (observed)',
                    'Minute Volume', 'Respiratory Rate (Set)',
                    'Inspiratory Ratio',
                    'Respiratory Rate (spontaneous)',
                    'Respiratory Rate (Total)',
                    'Peak Insp. Pressure', 'Mean Airway Pressure',
                    'Expiratory Ratio',
                    'Inspiratory Time', 'Minute Volume Alarm - Low',
                    'Minute Volume Alarm - High', 'PEEP set',
                    'Inspired O2 Fraction',
                    'Paw High', 'Ventilator Mode',
                    'Arterial Blood Pressure diastolic',
                    'Heart Rate', 'Arterial Blood Pressure systolic',
                    'Arterial Blood Pressure mean', 'Respiratory Rate',
                    'Alarms On',
                    'Heart rate Alarm - High', 'Heart Rate Alarm - Low',
                    'Arterial Blood Pressure Alarm - Low',
                    'Arterial Blood Pressure Alarm - High',
                    'O2 Saturation Pulseoxymetry Alarm - High',
                    'O2 Saturation Pulseoxymetry Alarm - Low',
                    'Parameters Checked',
                    'SpO2 Desat Limit', 'Temperature Fahrenheit',
                    'Chloride (serum)',
                    'Creatinine (serum)', 'Sodium (serum)', 'BUN',
                    'Anion gap',
                    'Potassium (serum)', 'HCO3 (serum)',
                    'Non Invasive Blood Pressure diastolic',
                    'Non Invasive Blood Pressure systolic',
                    'Non Invasive Blood Pressure mean',
                    'GCS - Motor Response',
                    'GCS - Verbal Response', 'GCS - Eye Opening',
                    'Activity / Mobility (JH-HLM)',
                    'Arterial Line placed in outside facility',
                    'Arterial Line Dressing Occlusive',
                    '18 Gauge Dressing Occlusive',
                    '18 Gauge placed in outside facility',
                    '18 Gauge placed in the field',
                    '20 Gauge placed in the field',
                    '20 Gauge placed in outside facility',
                    '20 Gauge Dressing Occlusive', 'PSV Level',
                    'Subglottal Suctioning', 'Tidal Volume (spontaneous)',
                    'O2 Flow',
                    'Glucose finger stick (range 70-100)',
                    'Secondary diagnosis',
                    'Mental status', 'Gait/Transferring', 'IV/Saline lock',
                    'Ambulatory aid', 'History of falling (within 3 mnths)',
                    'Pain Level', 'Resp Alarm - High', 'Resp Alarm - Low',
                    'ST Segment Monitoring On', 'Richmond-RAS Scale',
                    'Goal Richmond-RAS Scale', 'Orientation to Person',
                    'Orientation to Place', 'Orientation to Time',
                    'Strength L Arm',
                    'Strength L Leg', 'Strength R Leg', 'Braden Mobility',
                    'Braden Nutrition', 'Strength R Arm', 'CAM-ICU MS Change',
                    'Braden Friction/Shear', 'Braden Activity',
                    'Braden Sensory Perception', 'Braden Moisture',
                    'Glucose (serum)',
                    'Magnesium', 'Daily Weight', 'Current Dyspnea Assessment',
                    'High risk (>51) interventions',
                    'Non-Invasive Blood Pressure Alarm - High',
                    'Non-Invasive Blood Pressure Alarm - Low',
                    'Calcium non-ionized',
                    'Phosphorous']
