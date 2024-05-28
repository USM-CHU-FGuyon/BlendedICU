"""
This code extracts the data from the MIMIC-IV dataset 
('mimic_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('mimic' in paths.json). 
Approximate running time: 20min.
"""
from mimic3_preprocessing.mimic3preparator import mimic3Preparator


mimic3_prep = mimic3Preparator(chartevents_pth='CHARTEVENTS.csv.gz',
                               d_labitems_pth='D_LABITEMS.csv.gz',
                               d_items_pth='D_ITEMS.csv.gz',
                               outputevents_pth='OUTPUTEVENTS.csv.gz',
                               icustays_pth='ICUSTAYS.csv.gz',
                               patients_pth='PATIENTS.csv.gz',
                               inputevents_mv_pth='INPUTEVENTS_MV.csv.gz',
                               inputevents_cv_pth='INPUTEVENTS_CV.csv.gz',
                               labevents_pth='LABEVENTS.csv.gz',
                               admissions_pth='ADMISSIONS.csv.gz')

mimic3_prep.raw_tables_to_parquet()

mimic3_prep.icustays = mimic3_prep.gen_icustays()
mimic3_prep.gen_labels()
mimic3_prep.gen_flat()
mimic3_prep.gen_medication()
mimic3_prep.gen_timeseriesoutputs()
mimic3_prep.gen_timeserieslab()
mimic3_prep.gen_timeseries()
