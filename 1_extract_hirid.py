"""
This code extracts the data from the Amsterdam dataset 
('hirid_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('hirid' in paths.json). 
"""
from hirid_preprocessing.HiridPreparator import hiridPreparator

hirid_prep = hiridPreparator(
    variable_ref_path='hirid_variable_reference_v1.csv',
    raw_ts_path='raw_stage/observation_tables_parquet.tar.gz',
    raw_pharma_path='raw_stage/pharma_records_parquet.tar.gz',
    admissions_path='reference_data.tar.gz',
    imputedstage_path='imputed_stage/imputed_stage_parquet.tar.gz',
    untar=True)

hirid_prep.gen_labels()
hirid_prep.gen_medication()
hirid_prep.gen_timeseries()
