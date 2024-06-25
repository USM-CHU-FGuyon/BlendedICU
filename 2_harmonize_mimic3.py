"""
This script lauches the timeseriesprocessing (tsp) and the flat and labels 
processing (flp) for the mimic database. 
Note that this produces the 'raw' data of the BlendedICU dataset.
The preprocessed BlendedICU dataset will then be obtained with 3_blendedICU.py
Approximate running time: <1min.
"""
from mimic3_preprocessing.flat_and_labels import mimic3_FLProcessor
from mimic3_preprocessing.timeseries import mimic3TSP

tsp = mimic3TSP(
    med_pth='medication.parquet',
    ts_pth='timeseries.parquet',
    tslab_pth='timeserieslab.parquet',
    outputevents_pth='timeseriesoutputs.parquet')

tsp.run_harmonization()

flp = mimic3_FLProcessor()

flp.run_labels()
