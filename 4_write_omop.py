"""
At this point, the dataset is ready to be written to the OMOP format.
Most tables are very quickly written. Only the drug_exposure and measurement 
variables require a lot of time. 
"""
from blended_preprocessing.omop_conversion import OMOP_converter

self = OMOP_converter(initialize_tables=True)

self.observation_period_table()
self.measurement_table()
self.drug_exposure_table()

