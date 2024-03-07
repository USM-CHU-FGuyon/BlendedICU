"""
At this point, the dataset is ready to be written to the OMOP format.
Most tables are very quickly written. Only the drug_exposure and measurement 
variables require a lot of time. 

To reduce the memory overhead, the timeseries data is once again processed by 
chunks. We provide the option to generate a csv dataset, however this is 
inconvenient due to the size of the data. 
The preffered way is to output .parquet dataset, written using the OMOP 
specifications.
"""
from blended_preprocessing.omop_conversion import OMOP_converter

self = OMOP_converter(initialize_tables=True,
                      recompute_index=False)

self.measurement_table(start_chunk=0)

self.drug_exposure_table(start_chunk=0)
