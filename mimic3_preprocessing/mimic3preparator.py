from pathlib import Path

import polars as pl

from database_processing.newmedicationprocessor import NewMedicationProcessor
from database_processing.datapreparator import DataPreparator

class mimic3Preparator(DataPreparator):
    def __init__(self,
                 chartevents_pth,
                 labevents_pth,
                 d_labitems_pth,
                 admissions_pth,
                 d_items_pth,
                 outputevents_pth,
                 icustays_pth,
                 patients_pth,
                 inputevents_mv_pth,
                 inputevents_cv_pth,
                 ):
        super().__init__(dataset='mimic3', col_stayid='ICUSTAY_ID')
        
        self.chartevents_pth = self.source_pth + chartevents_pth
        self.labevents_pth = self.source_pth + labevents_pth
        self.d_labitems_pth = self.source_pth + d_labitems_pth
        self.admissions_pth = self.source_pth + admissions_pth
        self.d_items_pth = self.source_pth + d_items_pth
        self.outputevents_pth = self.source_pth + outputevents_pth
        self.icustays_pth = self.source_pth + icustays_pth
        self.patients_pth = self.source_pth + patients_pth
        self.inputevents_mv_pth = self.source_pth + inputevents_mv_pth
        self.inputevents_cv_pth = self.source_pth + inputevents_cv_pth

        self.outputevents_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(outputevents_pth)
        self.admissions_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(admissions_pth)
        self.inputevents_mv_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(inputevents_mv_pth)
        self.inputevents_cv_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(inputevents_cv_pth)
        self.icustays_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(icustays_pth)
        self.patients_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(patients_pth)
        self.d_items_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(d_items_pth)
        self.d_labitems_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(d_labitems_pth)
        self.labevents_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(labevents_pth)
        self.chartevents_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(chartevents_pth)

        self.outputevents_savepath = self.savepath + 'timeseriesoutputs.parquet'
        self.lab_savepath = self.savepath + 'timeserieslab.parquet'
        self.ts_savepath = self.savepath + 'timeseries.parquet'
        
        self.col_los = 'LOS'
        
    def gen_icustays(self):
        admissions = pl.scan_parquet(self.admissions_parquet_pth)
        icustays = pl.scan_parquet(self.icustays_parquet_pth)
        
        df_icustays = (icustays
                       .join(admissions.select('HADM_ID',
                                               'ETHNICITY',
                                               'ADMISSION_LOCATION',
                                               'INSURANCE',
                                               'DISCHARGE_LOCATION',
                                               'HOSPITAL_EXPIRE_FLAG'),
                             on='HADM_ID',
                             how='left')
                       .with_columns(
                           pl.col('INTIME').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                           pl.duration(seconds=pl.col('LOS').mul(self.seconds_in_a_day)).alias('LOS'),
                           pl.col('ICUSTAY_ID').cast(pl.Int32),
                           pl.col('HADM_ID').cast(pl.Int32),
                       ))
        return df_icustays
    
    
    def raw_tables_to_parquet(self):
        """
        Writes initial csv.gz files to parquet files. This operations 
        needs only to be done once and allows further methods to be 
        done laziy using polars.
        """
        for i, src_pth in enumerate([
                self.chartevents_pth,
                self.admissions_pth,
                self.icustays_pth,
                self.patients_pth,
                self.d_labitems_pth,
                self.d_items_pth,
                self.outputevents_pth,
                self.inputevents_cv_pth,
                self.inputevents_mv_pth,
                self.labevents_pth,
                
                ]):
            tgt = self.raw_as_parquet_pth + self._get_name_as_parquet(src_pth)
            if Path(tgt).is_file() and i==0:
                inp = input('Some parquet files already exist, skip conversion to parquet ?[n], y')
                if inp.lower() == 'y':
                    break
            
            self.write_as_parquet(src_pth,
                                  tgt,
                                  astype_dic={'HADM_ID': float,
                                              'ICUSTAY_ID': float,
                                              'VALUENUM':float,
                                              'VALUEUOM': str,
                                              'AMOUNTUOM': str,
                                              'RATEUOM': str,
                                              'CGID': float,
                                              'ORIGINALRATEUOM':str,
                                              'ORIGINALSITE': str,
                                              'ORIGINALAMOUNTUOM': str,
                                              'RESULTSTATUS': str,
                                              'STOPPED': str,
                                              'VALUE': str,
                                              'WARNING': str,
                                              'ERROR': str,
                                              'RESULTSTATUS': str},
                                  chunksize=1e7)
    

    def _fetch_heights_weights(self):
        """
        admission heights and weights, available at self.flat_hr_from_adm
        are fetched to fill the flat and labels tables.
        They can be found under several itemids depending on the unit in 
        which they are measured. Every value is converted to the metric system.
        """
        print('Fetching heights and weights in timeseries, this takes'
              'minutes')
        icustays = self.icustays.lazy()
        itemids = {'weight_kg_2': 224639,
                   'weight_kg': 226512,
                   'weight_lbs': 226531,
                   'height_inch': 226707,
                   'height_cm': 226730}

        keepids = [*itemids.values()]

        df = (pl.scan_parquet(self.chartevents_parquet_pth)
                  .select('ICUSTAY_ID', 'ITEMID', 'VALUENUM', 'CHARTTIME')
                .with_columns(
                    pl.col('CHARTTIME').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                    pl.col('ICUSTAY_ID').cast(pl.Int32, strict=False),
                    )
                .join(icustays.select('ICUSTAY_ID', 'INTIME'), on='ICUSTAY_ID')
                .with_columns(
                    (pl.col('CHARTTIME') - pl.col('INTIME')).alias('measuretime')
                    )
                .filter(
                    pl.col('ITEMID').is_in(keepids),
                    pl.col('measuretime').le(pl.duration(hours=self.flat_hr_from_adm_int))
                    )
                .cast({'ITEMID': pl.String})
                .drop('measuretime', 'INTIME')
                .collect(streaming=True)
                .group_by(['ICUSTAY_ID', 'CHARTTIME', 'ITEMID']).first()
                .pivot(index=['ICUSTAY_ID', 'CHARTTIME'], columns='ITEMID', values='VALUENUM')
                .rename({str(v): k for k, v in itemids.items()})
                .with_columns(
                    pl.col('height_inch').mul(self.inch_to_cm).alias('height_inch_in_cm'),
                    pl.col('weight_lbs').mul(self.lbs_to_kg).alias('weight_lbs_in_kg')
                    )
                .with_columns(
                    pl.concat_list(pl.col('height_cm', 'height_inch_in_cm' )).list.mean().alias('height'),
                    pl.concat_list(pl.col('weight_kg', 'weight_lbs_in_kg', 'weight_kg_2')).list.mean().alias('weight')
                    )
                .select('ICUSTAY_ID', 'CHARTTIME', 'height', 'weight')
                .select(pl.all()
                        .sort_by('CHARTTIME')
                        .forward_fill()
                        .over('ICUSTAY_ID')
                        .sort_by('ICUSTAY_ID'))
                .group_by('ICUSTAY_ID')
                .last()
                .drop('CHARTTIME')
                .lazy())
        return df


    def gen_flat(self):
        icustays = self.icustays.lazy()
        print('o Flat Features')
        patients = (pl.scan_parquet(self.patients_parquet_pth)
                    .select('SUBJECT_ID',
                            'GENDER',
                            'DOB'))

        self.heights_weights = self._fetch_heights_weights()

        df_flat = (icustays
                   .join(patients, on='SUBJECT_ID')
                   .join(self.heights_weights, on='ICUSTAY_ID')
                   .select(pl.all().sort_by('ICUSTAY_ID'))
                   .with_columns(
                       pl.col('DOB').str.to_datetime("%Y-%m-%d %H:%M:%S")
                       )
                   .with_columns(
                       hour=pl.col('INTIME').dt.hour(),
                       age= pl.col('INTIME') - pl.col('DOB')
                       )
                   .drop('SUBJECT_ID',
                         'HADM_ID',
                         'LAST_CAREUNIT',
                         'INTIME',
                         'OUTTIME',
                         'LOS')
                   .collect())

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

        inputevents_mv = (pl.scan_parquet(self.inputevents_mv_parquet_pth)
                          .select('ICUSTAY_ID',
                                'STARTTIME',
                                'ENDTIME',
                                'ITEMID',
                                'AMOUNT',
                                'AMOUNTUOM')
                          .rename({'STARTTIME': 'start',
                                   'ENDTIME': 'end'}))
        
        inputevents_cv = (pl.scan_parquet(self.inputevents_cv_parquet_pth)
                          .select('ICUSTAY_ID',
                                  'CHARTTIME',
                                  'ITEMID',
                                  'AMOUNT',
                                  'AMOUNTUOM',
                                  'ORIGINALROUTE')
                          .rename({'CHARTTIME': 'start'}))
        
        d_items = pl.scan_parquet(self.d_items_parquet_pth)

        inputevents = pl.concat([inputevents_cv, inputevents_mv], how='diagonal')

        df_inputevents = (inputevents
                          .join(d_items.select('ITEMID', 'LABEL'), on='ITEMID')
                          .drop('ITEMID')
                          .drop_nulls('ICUSTAY_ID')
                          .with_columns(
                              pl.col('ICUSTAY_ID').cast(pl.Int32),
                              pl.col('start').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                              pl.col('end').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                              )
                          )
        
        return df_inputevents

    def gen_medication(self):
        inputevents = self._load_inputevents()
        icustays = (self.icustays.lazy()
                    .select('ICUSTAY_ID', 'INTIME', 'LOS'))
        
        dose_unit_conversions = {
            'gm': {"omop_code": "mg",
                    "mul": 1e3},
            'grams': {"omop_code": "mg",
                      "mul": 1e3},
            'mcg': {'omop_code': 'mg',
                    'mul': 0.001},
            'L': {'omop_code': 'mL',
                  'mul': 1e3},
            'mL': {'omop_code': 'mL',
                  'mul': 1},
            'uL': {'omop_code': 'mL',
                  'mul': 1e3},
            'cc': {'omop_code': 'mL',
                   'mul': 1},
            'ml': {'omop_code': 'mL',
                   'mul': 1},
            'mEq': {'omop_code': '10*-3.eq',
                    'mul': 1},
            'mEq.': {'omop_code': '10*-3.eq',
                    'mul': 1},
            'mEQ': {'omop_code': '10*-3.eq',
                    'mul': 1},
            }
        
        
        self.nmp = NewMedicationProcessor(self.dataset,
                                          lf_med=inputevents,
                                          lf_labels=icustays,
                                          col_pid='ICUSTAY_ID',
                                          col_med='LABEL',
                                          col_start='start',
                                          col_end='end',
                                          col_los='LOS',
                                          col_dose='AMOUNT',
                                          col_dose_unit='AMOUNTUOM',
                                          col_route='ORIGINALROUTE',
                                          col_admittime='INTIME',
                                          offset_calc=True,
                                          dose_unit_conversion_dic=dose_unit_conversions
                                        )
        med = self.nmp.run()
        self.save(med, self.med_savepath)
        

    def gen_timeseriesoutputs(self):
        self.get_labels(lazy=True)
        ditems = pl.scan_parquet(self.d_items_parquet_pth)
        
        outputevents = pl.scan_parquet(self.outputevents_parquet_pth)
        
        df_outputs = (outputevents
                      .select('HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE')
                      .with_columns(
                          pl.col('CHARTTIME').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                          pl.col('HADM_ID').cast(pl.Int32)
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
                      .collect())
        
        self.save(df_outputs, self.outputevents_savepath)

    
    def gen_timeserieslab(self):
        print('o Timeseries Lab')
        self.get_labels(lazy=True)
        dlabitems = pl.scan_parquet(self.d_labitems_parquet_pth)
        labevents = pl.scan_parquet(self.labevents_parquet_pth)
        
        self.df_lab = (labevents
                       .select('HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM')
                       .drop_nulls()
                       .with_columns(
                           pl.col('CHARTTIME').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                           pl.col('HADM_ID').cast(pl.Int32)
                           )
                       .pipe(self.pl_prepare_tstable,
                             col_measuretime='CHARTTIME',
                             col_intime='INTIME',
                             col_variable='ITEMID',
                             col_mergestayid='HADM_ID',
                             col_value='VALUENUM')
                       .join(dlabitems.select('ITEMID', 'LABEL'), on='ITEMID')
                       .drop(columns=['HADM_ID', 'ITEMID'])
                       .collect())
        
        self.save(self.df_lab, self.lab_savepath)


    def gen_timeseries(self):
        self.get_labels(lazy=True)

        ditems = pl.scan_parquet(self.d_items_parquet_pth)
        chartevents = pl.scan_parquet(self.chartevents_parquet_pth)

        ts = (chartevents
              .select('ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUENUM')
              .drop_nulls()
              .with_columns(
                  pl.col('CHARTTIME').str.to_datetime("%Y-%m-%d %H:%M:%S"),
                  pl.col('ICUSTAY_ID').cast(pl.Int32)
                  )
              .pipe(self.pl_prepare_tstable,
                    col_measuretime='CHARTTIME',
                    col_intime='INTIME',
                    col_variable='ITEMID',
                    col_value='VALUENUM',
                    unit_los='day')
              .join(ditems.select('ITEMID', 'LABEL'), on='ITEMID')
              .drop('ITEMID')
              #.collect(streaming=True)
              )

        self.save(ts, self.ts_savepath)
