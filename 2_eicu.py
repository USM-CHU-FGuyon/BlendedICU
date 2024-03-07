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
    lab_pth='tslab_1000_patient_chunks/',
    resp_pth='tsresp_1000_patient_chunks/',
    nurse_pth='tsnurse_1000_patient_chunks/',
    aperiodic_pth='tsperiodic_1000_patient_chunks/',
    periodic_pth='tsaperiodic_1000_patient_chunks/',
    inout_pth='tsintakeoutput_1000_patient_chunks/')

tsp.run()

flp = eicu_FLProcessor()

flp.run_labels()
