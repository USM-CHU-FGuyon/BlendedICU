"""
This code extracts the data from the MIMIC-IV dataset 
('mimic_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('mimic' in paths.json). 
Approximate running time: 20min.
"""
from mimic3_preprocessing.mimic3preparator import mimic3Preparator


mimic3_prep = mimic3Preparator(chartevents_pth='CHARTEVENTS.csv.gz')

mimic3_prep.load_raw_tables()

mimic3_prep.icustays = mimic3_prep.gen_icustays()
mimic3_prep.gen_labels()
mimic3_prep.gen_flat()
mimic3_prep.gen_medication()
mimic3_prep.gen_timeseriesoutputs()
mimic3_prep.gen_timeserieslab()
mimic3_prep.gen_timeseries()
