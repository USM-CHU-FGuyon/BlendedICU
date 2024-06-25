"""
This code extracts the data from the Amsterdam dataset 
('hirid_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('hirid' in paths.json). 

"""
from hirid_preprocessing.HiridPreparator import hiridPreparator

hirid_prep = hiridPreparator(
    variable_ref_path='hirid_variable_reference_v1.csv',
    ts_path='observation_tables/parquet/',
    pharma_path='pharma_records/parquet/',
    admissions_path='reference_data/general_table.csv',
    imputedstage_path='imputed_stage/parquet/')

hirid_prep.raw_tables_to_parquet()

hirid_prep.init_gen()
hirid_prep.gen_labels()
hirid_prep.gen_medication()
hirid_prep.gen_timeseries()
