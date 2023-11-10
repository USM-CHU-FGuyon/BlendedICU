from functools import partial

import pandas as pd

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
        
        suffix = f'_{self.n_patient_chunk}_patient_chunks/'

        self.diag_savepath = f'{self.parquet_pth}/diagnoses.parquet'
        self.lab_savepath = f'{self.parquet_pth}/tslab{suffix}'
        self.resp_savepath = f'{self.parquet_pth}/tsresp{suffix}'
        self.nurse_savepath = f'{self.parquet_pth}/tsnurse{suffix}'
        self.periodic_savepath = f'{self.parquet_pth}/tsperiodic{suffix}'
        self.aperiodic_savepath = f'{self.parquet_pth}/tsaperiodic{suffix}'
        self.intakeoutput_savepath = f'{self.parquet_pth}/tsintakeoutput{suffix}'

        self.col_los = 'unitdischargeoffset'
        self.unit_los = 'minute'

        self.patient = pd.read_csv(self.patient_pth)

    def _load_los(self, col_id, col_los):
        try:
            return self.load(self.labels_savepath, columns=[col_id, col_los])
        except FileNotFoundError:
            return

    def gen_labels(self):
        patient = self.patient.loc[:, ['uniquepid',
                                       'patienthealthsystemstayid',
                                       'patientunitstayid',
                                       'unitvisitnumber',
                                       'unitdischargelocation',
                                       'unitdischargestatus',
                                       'unitdischargeoffset',
                                       'hospitalid',
                                       'unittype']]
        patient = (patient.dropna(subset=['unitdischargestatus'])
                          .sort_values('patientunitstayid'))

        self.save(patient, self.labels_savepath)
        self.labels = patient

    def gen_medication(self):
        """
        Medication can be found in three separate tables. 
        """
        print('o Medication')
        self.get_labels()
        self.admissiondrug = pd.read_csv(self.admissiondrug_pth,
                                         usecols=['patientunitstayid',
                                                  'drugoffset',
                                                  'drugname'])
        self.infusiondrug = pd.read_csv(self.infusiondrug_pth,
                                        usecols=['patientunitstayid',
                                                 'infusionoffset',
                                                 'drugname'])

        self.medication_table = pd.read_csv(self.medication_pth,
                                            usecols=['patientunitstayid',
                                                     'drugstartoffset',
                                                     'drugname'])

        self.medication_in = pd.concat([self.infusiondrug
                                        .rename(columns={'infusionoffset':
                                                         'drugoffset'}),
                                        self.medication_table
                                        .rename(columns={'drugstartoffset':
                                                         'drugoffset'}),
                                        self.admissiondrug])

        self.mp = MedicationProcessor(self.dataset,
                                      self.labels,
                                      col_med='drugname',
                                      col_time='drugoffset',
                                      col_pid='patientunitstayid',
                                      col_los='unitdischargeoffset',
                                      unit_offset='minute',
                                      unit_los='minute')

        self.med = self.mp.run(self.medication_in)
        return self.save(self.med, self.med_savepath)

    def gen_flat(self):
        print('o Flat features')
        flat = self.patient.loc[:, ['patientunitstayid',
                                    'gender',
                                    'age',
                                    'ethnicity',
                                    'admissionheight',
                                    'admissionweight',
                                    'apacheadmissiondx',
                                    'unitadmittime24',
                                    'unittype',
                                    'unitadmitsource',
                                    'unitvisitnumber',
                                    'unitstaytype', ]]

        flat['hour'] = pd.to_datetime(flat['unitadmittime24']).dt.hour
        flat = (flat.sort_values('patientunitstayid')
                    .drop(columns=['unitadmittime24']))

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
        """
        The lab table fits in memory so it is processed all at once.
        To ease further processing, the processed table are saved into chunks 
        of 1_000 patients (by default).
        A set of variables, listed in keepvars are given for reducing memory
        usage.
        """
        print('o Timeserieslab')
        self.get_labels()
        lab = pd.read_csv(self.lab_pth,
                          usecols=['patientunitstayid',
                                   'labname',
                                   'labresultoffset',
                                   'labresult'])

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

        self.tslab = self.prepare_tstable(lab,
                                          keepvars=keepvars,
                                          col_offset='labresultoffset',
                                          col_variable='labname',
                                          unit_offset='minute')

        self.split_and_save_chunks(self.tslab, self.lab_savepath)

    def gen_timeseriesintakeoutput(self):
        """
        The table is processed all at once and saved by chunks, see tslab.
        """
        print('o Timeseries Intakeoutput')
        self.get_labels()
        intakeoutput = pd.read_csv(self.intakeoutput_pth,
                                   usecols=['patientunitstayid',
                                            'celllabel',
                                            'intakeoutputoffset',
                                            'cellvaluenumeric'])

        self.intakeout = self.prepare_tstable(intakeoutput,
                                              col_offset='intakeoutputoffset',
                                              col_variable='celllabel',
                                              unit_offset='minute')

        self.split_and_save_chunks(self.intakeout, self.intakeoutput_savepath)

    def gen_timeseriesresp(self):
        """
        The table is processed all at once and saved by chunks, see tslab.
        Some percentages (eg. FiO2) are often written '80%', the % sign is 
        removed before conversion to numeric.
        """
        self.get_labels()
        print('o Timeseriesresp')
        respiratorycharting = pd.read_csv(self.respiratorycharting_pth,
                                          usecols=['patientunitstayid',
                                                   'respchartoffset',
                                                   'respchartvaluelabel',
                                                   'respchartvalue'])

        keepvars = ['FiO2', 'Total RR', 'Vent Rate', 'Tidal Volume (set)',
                    'TV/kg IBW', 'Mechanical Ventilator Mode',
                    'PEEP', 'Plateau Pressure', 'LPM O2', 'Pressure Support',
                    'Peak Insp. Pressure', 'RR (patient)',
                    'Exhaled TV (patient)',
                    'Mean Airway Pressure', 'Exhaled MV', 'SaO2']

        self.tsresp = respiratorycharting.pipe(self.prepare_tstable,
                                               keepvars=keepvars,
                                               col_offset='respchartoffset',
                                               col_variable='respchartvaluelabel',
                                               unit_offset='minute')
                       
        self.tsresp['respchartvalue'] = (self.tsresp['respchartvalue']
                                         .str.replace('%', '')
                                         .pipe(pd.to_numeric,
                                               errors='coerce')
                                         .fillna(self.tsresp['respchartvalue'])
                                         .astype(str))

        self.split_and_save_chunks(self.tsresp,  self.resp_savepath)

    def gen_timeseriesnurse(self):
        """
        The table is processed all at once and saved by chunks, see tslab.
        """
        self.get_labels()
        print('o Timeseriesnurse')
        nursecharting = pd.read_csv(self.nursecharting_pth,
                                    usecols=['patientunitstayid',
                                             'nursingchartoffset',
                                             'nursingchartcelltypevallabel',
                                             'nursingchartvalue'])

        keepvars = ['Non-Invasive BP', 'Heart Rate', 'Pain Score/Goal',
                    'Respiratory Rate', 'O2 Saturation', 'Temperature',
                    'Glasgow coma score', 'Invasive BP', 'Bedside Glucose',
                    'O2 L/%',
                    'O2 Admin Device', 'Sedation Scale/Score/Goal',
                    'Delirium Scale/Score']

        tsnurse = (nursecharting.pipe(self.prepare_tstable,
                                      keepvars=keepvars,
                                      col_offset='nursingchartoffset',
                                      col_variable='nursingchartcelltypevallabel',
                                      unit_offset='minute')
                   .astype({'nursingchartvalue': str}))

        self.split_and_save_chunks(tsnurse, self.nurse_savepath)

    def gen_timeseriesaperiodic(self):
        """
        This table is already in wide format, but this does not change this
        processing step.
        The table is processed all at once and saved by chunks, see tslab.
        """
        self.get_labels()
        print('o Timeseriesaperiodic')
        vitalaperiodic = pd.read_csv(self.aperiodic_pth,
                                     usecols=['patientunitstayid',
                                              'observationoffset',
                                              'noninvasivesystolic',
                                              'noninvasivediastolic',
                                              'noninvasivemean'])

        tsaperiodic = self.prepare_tstable(vitalaperiodic,
                                           col_offset='observationoffset',
                                           unit_offset='minute')

        self.split_and_save_chunks(tsaperiodic, self.aperiodic_savepath)

    def gen_timeseriesperiodic(self):
        """
        This table is too large to be processed at once. It is processed and 
        saved by chunks of 1000 patients (by default).
        """
        self.get_labels()
        print('o Timeseriesperiodic')
        vitalperiodic = pd.read_csv(self.periodic_pth,
                                    chunksize=self.chunksize,
                                    usecols=['patientunitstayid',
                                             'observationoffset',
                                             'temperature',
                                             'sao2',
                                             'heartrate',
                                             'respiration',
                                             'cvp',
                                             'systemicsystolic',
                                             'systemicdiastolic',
                                             'systemicmean',
                                             'st1', 'st2', 'st3'])

        prepare_tstable = partial(self.prepare_tstable,
                                  col_offset='observationoffset',
                                  unit_offset='minute')

        df_tsp = pd.concat(map(prepare_tstable, vitalperiodic))

        self.split_and_save_chunks(df_tsp, self.periodic_savepath)
