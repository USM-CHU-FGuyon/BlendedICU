"""
This script lauches the timeseriesprocessing (tsp) and the flat and labels 
processing (flp) for the amsterdam database. 
Note that this produces the 'raw' data of the BlendedICU dataset.
The preprocessed BlendedICU dataset will then be obtained with 3_blendedICU.py
Approximate running time: 2min
"""
from amsterdam_preprocessing.timeseries import amsterdamTSP
from amsterdam_preprocessing.flat_and_labels import Ams_FLProcessor
import polars as pl

tsp = amsterdamTSP(
    ts_chunks='numericitems.parquet',
    listitems_pth='listitems.parquet',
    gcs_scores_pth='glasgow_coma_scores.parquet')

tsp.run_harmonization()

flp = Ams_FLProcessor()

flp.run_labels()
