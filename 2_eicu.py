"""
This script lauches the timeseriesprocessing (tsp) and the flat and labels 
processing (flp) for the eicu database. 
Note that this produces the 'raw' data of the BlendedICU dataset.
The preprocessed BlendedICU dataset will then be obtained with 3_blendedICU.py
Approximate running time: 8h.
"""
from eicu_preprocessing.flat_and_labels import eicu_FLProcessor
from eicu_preprocessing.timeseries import eicuTSP

tsp = eicuTSP(
    lab_pth='lab.parquet',
    resp_pth='tsresp.parquet',
    nurse_pth='tsnurse/',
    aperiodic_pth='tsaperiodic.parquet',
    periodic_pth='tsperiodic/',
    inout_pth='tsintakeoutput.parquet')

tsp.run(reset_dir=False)

flp = eicu_FLProcessor()

flp.run_labels()
