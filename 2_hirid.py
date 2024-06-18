"""
This script lauches the timeseriesprocessing (tsp) and the flat and labels 
processing (flp) for the hirid database. 
Note that this produces the 'raw' data of the BlendedICU dataset.
The preprocessed BlendedICU dataset will then be obtained with 3_blendedICU.py
Approximate running time: 7h.
"""
from hirid_preprocessing.flat_and_labels import Hir_FLProcessing
from hirid_preprocessing.timeseries import hiridTSP
import polars as pl
self = hiridTSP(
    ts_chunks='timeseries.parquet',
    pharma_chunks='pharma_1000_patient_chunks/')

self.run(reset_dir=False)

flp = Hir_FLProcessing()

flp.run_labels()
