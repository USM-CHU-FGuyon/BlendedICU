"""
This code extracts the data from the Amsterdam dataset 
('mimic_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('mimic' in paths.json). 
"""
from mimic_preprocessing.mimicpreparator import mimicPreparator

mimic_prep = mimicPreparator(
    chartevents_pth='/icu/chartevents.csv.gz',                         
    labevents_pth='/hosp/labevents.csv.gz')

mimic_prep.load_raw_tables()

mimic_prep.gen_labels()
mimic_prep.gen_flat()
mimic_prep.gen_medication()
mimic_prep.gen_timeseriesoutputs()
mimic_prep.gen_timeserieslab()
mimic_prep.gen_timeseries()
