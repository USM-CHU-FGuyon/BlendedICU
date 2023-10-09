"""
This script lauches the timeseriesprocessing (tsp) and the flat and labels 
processing (flp) for the mimic database. 
Note that this produces the 'raw' data of the BlendedICU dataset.
The preprocessed BlendedICU dataset will then be obtained with 3_blendedICU.py
"""
from mimic_preprocessing.flat_and_labels import mimic_FLProcessor
from mimic_preprocessing.timeseries import mimicTSP

tsp = mimicTSP(
    med_pth='medication.parquet',
    ts_pth='timeseries.parquet',
    tslab_pth='timeserieslab.parquet',
    outputevents_pth='timeseriesoutputs.parquet')

tsp.run()

flp = mimic_FLProcessor()

flp.run_labels()
