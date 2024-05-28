from pathlib import Path

import polars as pl

from database_processing.medicationprocessor import MedicationProcessor
from database_processing.datapreparator import DataPreparator


class mimic4Preparator(DataPreparator):
    def __init__(self,
                 chartevents_pth,
                 labevents_pth,
                 d_labitems_pth,
                 admissions_pth,
                 diagnoses_pth,
                 d_diagnoses_pth,
                 d_items_pth,
                 inputevents_pth,
                 outputevents_pth,
                 icustays_pth,
                 patients_pth):
        
        super().__init__(dataset='mimic4', col_stayid='stay_id')
        self.chartevents_pth = self.source_pth + chartevents_pth
        self.labevents_pth = self.source_pth + labevents_pth
        self.d_labitems_pth = self.source_pth + d_labitems_pth
        self.diagnoses_pth = self.source_pth + diagnoses_pth
        self.admissions_pth = self.source_pth + admissions_pth
        self.d_diagnoses_pth = self.source_pth + d_diagnoses_pth
        self.d_items_pth = self.source_pth + d_items_pth
        self.outputevents_pth = self.source_pth + outputevents_pth
        self.icustays_pth = self.source_pth + icustays_pth
        self.patients_pth = self.source_pth + patients_pth
        self.inputevents_pth = self.source_pth + inputevents_pth

        self.diagnoses_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(diagnoses_pth)
        self.outputevents_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(outputevents_pth)
        self.admissions_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(admissions_pth)
        self.inputevents_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(inputevents_pth)
        self.icustays_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(icustays_pth)
        self.patients_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(patients_pth)
        self.d_items_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(d_items_pth)
        self.d_labitems_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(d_labitems_pth)
        self.d_diagnoses_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(d_diagnoses_pth)
        self.labevents_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(labevents_pth)
        self.chartevents_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(chartevents_pth)

        self.outputevents_savepath = self.savepath + 'timeseriesoutputs.parquet'
        self.lab_savepath = self.savepath + 'timeserieslab.parquet'
        self.flat_savepath = self.savepath + 'flat_features.parquet'
        self.ts_savepath = self.savepath + 'timeseries.parquet'
        
        self.col_los = 'los'
        self.unit_los = 'day'

        
    def raw_tables_to_parquet(self):
        """
        Writes initial csv.gz files to parquet files. This operations 
        needs only to be done once and allows further methods to be 
        done laziy using polars.
        """
        for i, src_pth in enumerate([
                self.admissions_pth,
                self.icustays_pth,
                self.patients_pth,
                self.d_labitems_pth,
                self.d_diagnoses_pth,
                self.d_items_pth,
                self.diagnoses_pth,
                self.outputevents_pth,
                self.inputevents_pth,
                self.labevents_pth,
                self.chartevents_pth,
                ]):
            tgt = self.raw_as_parquet_pth + self._get_name_as_parquet(src_pth)
            if Path(tgt).is_file() and i==0:
                inp = input('Some parquet files already exist, skip conversion to parquet ?[n], y')
                if inp.lower() == 'y':
                    break
            
            self.write_as_parquet(src_pth, tgt)



    def gen_icustays(self):
        
        admissions = pl.scan_parquet(self.admissions_parquet_pth)
        icustays = pl.scan_parquet(self.icustays_parquet_pth)
        
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


    def _fetch_heights_weights(self):
        """
        admission heights and weights, available at self.flat_hr_from_adm
        are fetched to fill the flat and labels tables.
        They can be found under several itemids depending on the unit in 
        which they are measured. Every value is converted to the metric system.
        """
        icustays = self.icustays.lazy()
        itemids = {'weight_kg_2': 224639,
                   'weight_kg': 226512,
                   'weight_lbs': 226531,
                   'height_inch': 226707,
                   'height_cm': 226730}

        keepids = [*itemids.values()]

        df = (pl.scan_parquet(self.chartevents_parquet_pth)
                .select('stay_id', 'itemid', 'valuenum', 'charttime')
                .with_columns(
                    pl.col('charttime').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                    )
                .join(icustays.select('stay_id', 'intime'), on='stay_id')
                .with_columns(
                    (pl.col('charttime') - pl.col('intime')).alias('measuretime')
                    )
                .filter(
                    pl.col('itemid').is_in(keepids),
                    pl.col('measuretime').le(pl.duration(hours=self.flat_hr_from_adm_int))
                    )
                .cast({'itemid': pl.String})
                .drop('measuretime', 'intime')
                
                .collect(streaming=True)
                .pivot(index=['stay_id', 'charttime'], columns='itemid', values='valuenum')
                .rename({str(v): k for k, v in itemids.items()})
                .with_columns(
                    pl.col('height_inch').mul(self.inch_to_cm).alias('height_inch_in_cm'),
                    pl.col('weight_lbs').mul(self.lbs_to_kg).alias('weight_lbs_in_kg')
                    )
                .with_columns(
                    pl.concat_list(pl.col('height_cm', 'height_inch_in_cm' )).list.mean().alias('height'),
                    pl.concat_list(pl.col('weight_kg', 'weight_lbs_in_kg', 'weight_kg_2')).list.mean().alias('weight')
                    )
                .select('stay_id', 'charttime', 'height', 'weight')
                .select(pl.all()
                        .sort_by('charttime')
                        .forward_fill()
                        .over('stay_id')
                        .sort_by('stay_id'))
                .group_by('stay_id')
                .last()
                .drop('charttime')
                .lazy())
        
        return df

    def gen_flat(self):
        print('o Flat Features')
        icustays = self.icustays.lazy()
        patients = (pl.scan_parquet(self.patients_parquet_pth)
                    .select('subject_id', 'gender', 'anchor_age'))

        self.heights_weights = self._fetch_heights_weights()

        df_flat = (icustays
                   .join(patients, on='subject_id')
                   .join(self.heights_weights, on='stay_id')
                   .select(pl.all().sort_by('stay_id'))
                   .rename({'anchor_age': 'age'})
                   .with_columns(
                       hour=pl.col('intime').dt.hour()
                       )
                   .drop('subject_id',
                         'hadm_id',
                         'last_careunit',
                         'intime',
                         'outtime',
                         'los')
                   .collect())

        return self.save(df_flat, self.flat_savepath)

    def _load_inputevents(self):
        print('o Inputevents')
        inputevents = pl.scan_parquet(self.inputevents_parquet_pth)
        d_items = pl.scan_parquet(self.d_items_parquet_pth)
        
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
        print('o Diagnoses')
        icustays = self.icustays.lazy().select('stay_id', 'hadm_id')
        diagnoses = pl.scan_parquet(self.diagnoses_parquet_pth)
        d_diagnoses = pl.scan_parquet(self.d_diagnoses_parquet_pth)
    
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
        print('o Medication')
        inputevents = self._load_inputevents().to_pandas()
        icustays = self.icustays.to_pandas()
    
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
        print('o Outputevents')
        self.get_labels(lazy=True)
        ditems = pl.scan_parquet(self.d_items_parquet_pth)
        
        outputevents = pl.scan_parquet(self.outputevents_parquet_pth)
        
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
        print('o Laboratory')
        self.get_labels(lazy=True)

        dlabitems = pl.scan_parquet(self.d_labitems_parquet_pth)
        
        print('o Timeseries Lab')
        labevents = pl.scan_parquet(self.labevents_parquet_pth)

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
                       .collect())

        self.save(self.df_lab, self.lab_savepath)

    def gen_timeseries(self):
        self.get_labels(lazy=True)
        ditems = pl.scan_parquet(self.d_items_parquet_pth)

        print('o Timeseries')
        keepvars = self._timeseries_keepvars()

        keepitemids = (ditems
                       .filter(pl.col('label').is_in(keepvars))
                       .select('itemid')
                       .collect()
                       .to_numpy()
                       .flatten())

        chartevents = pl.scan_parquet(self.chartevents_parquet_pth)

        ts = (chartevents
              .select('stay_id',
                      'charttime',
                      'itemid',
                      'valuenum')
             .drop_nulls()
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
             .collect(streaming=True))
        
        self.save(ts, self.ts_savepath)

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
