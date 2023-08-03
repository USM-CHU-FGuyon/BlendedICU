"""
This code extracts the data from the Amsterdam dataset 
('eicu_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('eicu' in paths.json). 
"""
from eicu_preprocessing.eicupreparator import eicuPreparator

eicu_prep = eicuPreparator(
    physicalexam_pth='physicalExam.csv.gz',
    diag_pth='diagnosis.csv.gz',
    pasthistory_pth='pastHistory.csv.gz',
    admissiondx_pth='admissionDx.csv.gz',
    medication_pth='medication.csv.gz',
    infusiondrug_pth='infusionDrug.csv.gz',
    admissiondrug_pth='admissionDrug.csv.gz',
    patient_pth='patient.csv.gz',
    lab_pth='lab.csv.gz',
    respiratorycharting_pth='respiratoryCharting.csv.gz',
    nursecharting_pth='nurseCharting.csv.gz',
    periodic_pth='vitalPeriodic.csv.gz',
    aperiodic_pth='vitalAperiodic.csv.gz',
    intakeoutput_pth='intakeOutput.csv.gz')

eicu_prep.gen_labels()
eicu_prep.gen_flat()
eicu_prep.gen_medication()
eicu_prep.gen_timeseriesintakeoutput()
eicu_prep.gen_timeseriesresp()
eicu_prep.gen_timeserieslab()
eicu_prep.gen_timeseriesnurse()
eicu_prep.gen_timeseriesaperiodic()
eicu_prep.gen_timeseriesperiodic()
