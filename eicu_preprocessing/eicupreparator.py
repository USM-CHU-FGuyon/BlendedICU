from pathlib import Path

import pandas as pd
import polars as pl

from database_processing.datapreparator import DataPreparator
from database_processing.medicationprocessor import MedicationProcessor


class eicuPreparator(DataPreparator):
    def __init__(self,
                 lab_pth,
                 diag_pth,
                 pasthistory_pth,
                 respiratorycharting_pth,
                 admissiondx_pth,
                 medication_pth,
                 infusiondrug_pth,
                 admissiondrug_pth,
                 patient_pth,
                 physicalexam_pth,
                 periodic_pth,
                 aperiodic_pth,
                 nursecharting_pth,
                 intakeoutput_pth):

        super().__init__(dataset='eicu', col_stayid='patientunitstayid')
        self.patient_pth = self.source_pth+patient_pth
        self.lab_pth = self.source_pth+lab_pth
        self.physicalexam_pth = self.source_pth+physicalexam_pth
        self.respiratorycharting_pth = self.source_pth+respiratorycharting_pth
        self.medication_pth = self.source_pth+medication_pth
        self.infusiondrug_pth = self.source_pth+infusiondrug_pth
        self.admissiondrug_pth = self.source_pth+admissiondrug_pth
        self.diag_pth = self.source_pth+diag_pth
        self.pasthistory_pth = self.source_pth+pasthistory_pth
        self.admissiondx_pth = self.source_pth+admissiondx_pth
        self.nursecharting_pth = self.source_pth+nursecharting_pth
        self.periodic_pth = self.source_pth+periodic_pth
        self.aperiodic_pth = self.source_pth+aperiodic_pth
        self.intakeoutput_pth = self.source_pth+intakeoutput_pth
        
        self.patient_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(patient_pth)
        self.lab_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(lab_pth)
        self.intakeoutput_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(intakeoutput_pth)
        self.respiratorycharting_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(respiratorycharting_pth)
        self.nursecharting_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(nursecharting_pth)
        self.aperiodic_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(aperiodic_pth)
        self.periodic_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(periodic_pth)
        self.admissiondrug_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(admissiondrug_pth)
        self.infusiondrug_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(infusiondrug_pth)
        self.medication_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(medication_pth)
        
        self.intakeoutput_savepath = f'{self.savepath}/tsintakeoutput.parquet'
        self.lab_savepath = f'{self.savepath}/lab.parquet'
        self.aperiodic_savepath = f'{self.savepath}/tsaperiodic.parquet'
        self.nursecharting_savepath = f'{self.savepath}/tsnurse.parquet'
        self.tsresp_savepath = f'{self.savepath}/tsresp.parquet'
        self.tsperiodic_savepath = f'{self.savepath}/tsperiodic.parquet'

        self.col_los = 'unitdischargeoffset'
        self.unit_los = 'minute'

        self.patient = pl.scan_parquet(self.patient_parquet_pth)



    def raw_tables_to_parquet(self):
        """
        Writes initial csv.gz files to parquet files. This operations 
        needs only to be done once and allows further methods to be 
        done laziy using polars.
        """
        for i, src_pth in enumerate([
                self.patient_pth,
                self.lab_pth,
                self.physicalexam_pth,
                self.respiratorycharting_pth,
                self.medication_pth,
                self.infusiondrug_pth,
                self.admissiondrug_pth,
                self.diag_pth,
                self.pasthistory_pth,
                self.admissiondx_pth,
                self.nursecharting_pth,
                self.periodic_pth,
                self.aperiodic_pth,
                self.intakeoutput_pth
                ]):
            tgt = self.raw_as_parquet_pth + self._get_name_as_parquet(src_pth)
            
            if Path(tgt).is_file() and i==0:
                inp = input('Some parquet files already exist, skip conversion to parquet ?[n], y')
                if inp.lower() == 'y':
                    break
            
            self.write_as_parquet(src_pth,
                                  tgt,
                                  astype_dic={'respchartvalue': str,
                                              'loadingdose': str,
                                              'drugrate': str})
            

    def _load_los(self, col_id, col_los):
        try:
            return self.load(self.labels_savepath, columns=[col_id, col_los])
        except FileNotFoundError:
            return

    def gen_labels(self):
        patient = (self.patient
                   .select('uniquepid',
                           'patienthealthsystemstayid',
                           'patientunitstayid',
                           'unitvisitnumber',
                           'unitdischargelocation',
                           'unitdischargestatus',
                           'unitdischargeoffset',
                           'hospitalid',
                           'unittype')
                   .filter(~pl.all_horizontal(pl.col('unitdischargestatus').is_null()))
                   .sort('patientunitstayid')
                   .collect())

        self.save(patient, self.labels_savepath)
        self.labels = patient

    def gen_medication(self):
        """
        Medication can be found in three separate tables. 
        """
        print('o Medication')
        self.get_labels(lazy=True)
        
        admissiondrug = (pl.scan_parquet(self.admissiondrug_parquet_pth)
                         .select('patientunitstayid',
                                 'drugoffset',
                                 'drugname'))
        
        infusiondrug = (pl.scan_parquet(self.infusiondrug_parquet_pth)
                        .select('patientunitstayid',
                                'infusionoffset',
                                'drugname')
                        .rename({"infusionoffset": "drugoffset"}))

        medication = (pl.scan_parquet(self.medication_parquet_pth)
                      .select('patientunitstayid',
                              'drugstartoffset',
                              'drugname')
                      .rename({'drugstartoffset': 'drugoffset'}))
        
        medication_in = (pl.concat([infusiondrug,
                                   medication,
                                   admissiondrug])
                         .collect()
                         .to_pandas())

        self.mp = MedicationProcessor(self.dataset,
                                      self.labels.collect().to_pandas(),
                                      col_med='drugname',
                                      col_time='drugoffset',
                                      col_pid='patientunitstayid',
                                      col_los='unitdischargeoffset',
                                      unit_offset='minute',
                                      unit_los='minute')

        self.med = self.mp.run(medication_in)
        return self.save(self.med, self.med_savepath)

    def gen_flat(self):
        print('o Flat features')
        flat = (self.patient
                .select('patientunitstayid',
                        'gender',
                        'age',
                        'ethnicity',
                        'admissionheight',
                        'admissionweight',
                        'apacheadmissiondx',
                        'unittype',
                        'unitadmitsource',
                        'unitvisitnumber',
                        'unitstaytype')
                .sort('patientunitstayid')
                .collect())

        return self.save(flat, self.flat_savepath)

    def gen_diagnoses(self):
        """
        The eICU database offers a diagnosis table. This extracts the data from
        the diagnoses, pasthistory and admission diagnosis, however it was not
        used in the BlendedICU dataset as the same data is not easily available
        on other sources.
        The variable self.flat_hr_from_adm (default 5h) specifies the time
        since admission where the admission diagnosis should be available to
        be included.
        """
        print('o Diagnoses')
        diagnosis = pd.read_csv(self.diag_pth,
                                usecols=['patientunitstayid',
                                         'diagnosisoffset',
                                         'diagnosisstring'])
        pasthistory = pd.read_csv(self.pasthistory_pth,
                                  usecols=['patientunitstayid',
                                           'pasthistoryoffset',
                                           'pasthistorypath'])
        admissiondx = pd.read_csv(self.admissiondx_pth,
                                  usecols=['patientunitstayid',
                                           'admitdxenteredoffset',
                                           'admitdxpath'])

        pasthistory = pasthistory.rename(columns={
            'pasthistoryoffset': 'diagnosisoffset',
            'pasthistorypath': 'diagnosisstring'})

        admissiondx = admissiondx.rename(columns={
            'admitdxenteredoffset': 'diagnosisoffset',
            'admitdxpath': 'diagnosisstring'})

        diagnoses = pd.concat([diagnosis, pasthistory, admissiondx], axis=0)

        diagnoses['diagnosisoffset'] = pd.to_timedelta(diagnoses.diagnosisoffset,
                                                       unit='min')
        idx_adm = diagnoses.diagnosisoffset < self.flat_hr_from_adm

        diagnoses = (diagnoses.loc[idx_adm]
                              .drop(columns=['diagnosisoffset'])
                              .sort_values('patientunitstayid'))

        return self.save(diagnoses, self.diag_savepath)


    def gen_timeserieslab(self):
        print('o Timeserieslab')
        self.get_labels(lazy=True)
        lab = pl.scan_parquet(self.lab_parquet_pth)
        
        keepvars = [
            'PT - INR', 'magnesium', 'PT', 'pH', 'MCH', 'BUN', 'HCO3',
            'lactate', 'PTT', 'FiO2', '-lymphs', 'chloride', 'troponin - I',
            'paO2', '-eos', 'platelets x 1000', 'anion gap', 'MCV', 'paCO2',
            'RBC', 'RDW', 'MCHC', 'alkaline phos.', 'WBC x 1000', 'creatinine',
            'calcium', 'Hgb', 'bicarbonate', 'potassium', '-polys', '-monos',
            'total bilirubin', '-basos', 'phosphate', 'ALT (SGPT)', 'Hct',
            'AST (SGOT)', 'glucose', 'total protein', 'sodium', 'albumin',
            'bedside glucose', 'urinary specific gravity', 'Base Excess',
            'O2 Sat (%)', 'MPV']
        
        df = (lab
              .select(['patientunitstayid',
                          'labname',
                          'labresultoffset',
                          'labresult'])
              .pipe(self.pl_prepare_tstable,
                    keepvars=keepvars,
                    col_offset='labresultoffset',
                    col_variable='labname',
                    unit_offset='minute',
                    unit_los='minute',
                    col_value='labresult')
              .collect())
        self.save(df, self.lab_savepath)


    def gen_timeseriesintakeoutput(self):
        print('o Timeseries Intakeoutput')
        self.get_labels(lazy=True)
        intakeoutput = pl.scan_parquet(self.intakeoutput_parquet_pth)
        
        self.intakeout = (intakeoutput.select(['patientunitstayid',
                                               'celllabel',
                                               'intakeoutputoffset',
                                               'cellvaluenumeric'])
                                      .pipe(self.pl_prepare_tstable,
                                            col_offset='intakeoutputoffset',
                                            col_variable='celllabel',
                                            unit_offset='minute',
                                            unit_los='minute',
                                            col_value='cellvaluenumeric')
                                      .collect())
        
        self.save(self.intakeout, self.intakeoutput_savepath)
        
    def gen_timeseriesresp(self):
        self.get_labels(lazy=True)
        print('o Timeseriesresp')
        respiratorycharting = pl.scan_parquet(self.respiratorycharting_parquet_pth)
        
        keepvars = ['FiO2', 'Total RR', 'Vent Rate', 'Tidal Volume (set)',
                    'TV/kg IBW', 'Mechanical Ventilator Mode',
                    'PEEP', 'Plateau Pressure', 'LPM O2', 'Pressure Support',
                    'Peak Insp. Pressure', 'RR (patient)',
                    'Exhaled TV (patient)',
                    'Mean Airway Pressure', 'Exhaled MV', 'SaO2']
        
        tsresp = (respiratorycharting
                  .select('patientunitstayid',
                          'respchartoffset',
                          'respchartvaluelabel',
                          'respchartvalue')
                  .pipe(self.pl_prepare_tstable,
                        keepvars=keepvars,
                        col_offset='respchartoffset',
                        col_variable='respchartvaluelabel',
                        unit_offset='minute',
                        unit_los='minute',
                        cast_to_float=False,
                        additional_expr=[(pl.col('respchartvalue')
                                          .str.replace('%', '')
                                          .cast(pl.Float32(), strict=False))])
                  .collect())
        
        self.save(tsresp, self.tsresp_savepath)


    def gen_timeseriesnurse(self):
        self.get_labels(lazy=True)
        print('o Timeseriesnurse')
        nursecharting = pl.scan_parquet(self.nursecharting_parquet_pth)
        
        keepvars = ['Non-Invasive BP', 'Heart Rate', 'Pain Score/Goal',
                    'Respiratory Rate', 'O2 Saturation', 'Temperature',
                    'Glasgow coma score', 'Invasive BP', 'Bedside Glucose',
                    'O2 L/%',
                    'O2 Admin Device', 'Sedation Scale/Score/Goal',
                    'Delirium Scale/Score']
    
        df = (nursecharting
              .select('patientunitstayid',
                       'nursingchartoffset',
                       'nursingchartcelltypevallabel',
                       'nursingchartvalue')
              .pipe(self.pl_prepare_tstable,
                    keepvars=keepvars,
                    col_offset='nursingchartoffset',
                    col_variable='nursingchartcelltypevallabel',
                    unit_offset='minute',
                    unit_los='minute',
                    col_value='nursingchartvalue')
              .collect())
        
        self.save(df, self.nursecharting_savepath)


    def gen_timeseriesaperiodic(self):
        """
        This table is already in wide format, but this does not change this
        processing step.
        The table is processed all at once and saved by chunks, see tslab.
        """
        self.get_labels(lazy=True)
        print('o Timeseriesaperiodic')
        vitalaperiodic = pl.scan_parquet(self.aperiodic_parquet_pth)
        numeric_cols =['noninvasivesystolic',
                       'noninvasivediastolic',
                       'noninvasivemean']
        
        df = (vitalaperiodic.select(['patientunitstayid',
                                     'observationoffset',
                                     *numeric_cols])
             .pipe(self.pl_prepare_tstable,
                   col_offset='observationoffset',
                   unit_offset='minute',
                   cast_to_float=False,
                   additional_expr=[pl.col(col).cast(pl.Float32, strict=False)
                                    for col in numeric_cols])
             .collect())
        
        self.save(df, self.aperiodic_savepath)


    def gen_timeseriesperiodic(self):
        '''
        As of polars 0.20 there is no support for reading csv.gz by chunks.
        We load the files in pandas and convert them to lazyframes for 
        unified processing. 
        
        This should be changed to full polars when the feature is up.
        
        '''
        self.get_labels(lazy=True)
        print('o Timeseriesperiodic')
        numeric_cols = ['temperature',
                        'sao2',
                        'heartrate',
                        'respiration',
                        'cvp',
                        'systemicsystolic',
                        'systemicdiastolic',
                        'systemicmean',
                        'st1', 'st2', 'st3']
        
        vitalperiodic = pl.scan_parquet(self.periodic_parquet_pth)
        
        lf = (vitalperiodic
              .select('patientunitstayid',
                      'observationoffset',
                      *numeric_cols)
              .pipe(self.pl_prepare_tstable,
                    col_offset='observationoffset',
                    unit_offset='minute',
                    unit_los='minute',
                    cast_to_float=False,
                    additional_expr=[pl.col(col).cast(pl.Float32, strict=False)
                                     for col in numeric_cols])
              .collect(streaming=True))
        
        self.save(lf, self.tsperiodic_savepath)
    