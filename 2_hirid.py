"""
This script lauches the timeseriesprocessing (tsp) and the flat and labels 
processing (flp) for the hirid database. 
Note that this produces the 'raw' data of the BlendedICU dataset.
The preprocessed BlendedICU dataset will then be obtained with 3_blendedICU.py
"""
from hirid_preprocessing.flat_and_labels import Hir_FLProcessing
from hirid_preprocessing.timeseries import hiridTSP

tsp = hiridTSP(
    ts_chunks='timeseries_1000_patient_chunks/',
    pharma_chunks='pharma_1000_patient_chunks/')

tsp.run()

flp = Hir_FLProcessing()

flp.run_labels()
