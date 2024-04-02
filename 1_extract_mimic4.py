"""
This code extracts the data from the MIMIC-IV dataset 
('mimic_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('mimic' in paths.json). 
Approximate running time: 20min.
"""
from mimic4_preprocessing.mimic4preparator import mimic4Preparator

mimic4_prep = mimic4Preparator(
    chartevents_pth='/icu/chartevents.csv.gz',                         
    labevents_pth='/hosp/labevents.csv.gz')

mimic4_prep.load_raw_tables()

mimic4_prep.icustays = mimic4_prep.gen_icustays()
mimic4_prep.gen_labels()
mimic4_prep.gen_flat()
mimic4_prep.gen_medication()
mimic4_prep.gen_timeseriesoutputs()
mimic4_prep.gen_timeserieslab()
mimic4_prep.gen_timeseries()
mimic4_prep.gen_diagnoses()
