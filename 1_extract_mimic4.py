"""
This code extracts the data from the MIMIC-IV dataset 
('mimic_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('mimic' in paths.json). 
Approximate running time: 
    * raw_tables_to_parquet()  19min #only run once. csv.gz -> parquet with no data changes.
    * gen_* : 1min
"""
from mimic4_preprocessing.mimic4preparator import mimic4Preparator

mimic4_prep = mimic4Preparator(
    chartevents_pth='/icu/chartevents.csv.gz',                         
    labevents_pth='/hosp/labevents.csv.gz',
    d_labitems_pth='hosp/d_labitems.csv.gz',
    admissions_pth='hosp/admissions.csv.gz',
    diagnoses_pth='hosp/diagnoses_icd.csv.gz',
    d_diagnoses_pth='hosp/d_icd_diagnoses.csv.gz',
    d_items_pth='icu/d_items.csv.gz',
    inputevents_pth='icu/inputevents.csv.gz',
    outputevents_pth='icu/outputevents.csv.gz',
    icustays_pth='icu/icustays.csv.gz',
    patients_pth='hosp/patients.csv.gz',
    )

mimic4_prep.raw_tables_to_parquet()

mimic4_prep.icustays = mimic4_prep.gen_icustays()
mimic4_prep.gen_labels()
mimic4_prep.gen_flat()
mimic4_prep.gen_medication()
mimic4_prep.gen_timeseriesoutputs()
mimic4_prep.gen_timeserieslab()
mimic4_prep.gen_timeseries()
mimic4_prep.gen_diagnoses()
