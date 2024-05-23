"""
This script lauches the timeseriesprocessing (tsp) and the flat and labels 
processing (flp) for the mimic database. 
Note that this produces the 'raw' data of the BlendedICU dataset.
The preprocessed BlendedICU dataset will then be obtained with 3_blendedICU.py
Approximate running time: 2h.
"""
from mimic4_preprocessing.flat_and_labels import mimic4_FLProcessor
from mimic4_preprocessing.diagnoses import mimic4_DiagProcessor
from mimic4_preprocessing.timeseries import mimic4TSP

tsp = mimic4TSP(
    med_pth='medication.parquet',
    ts_pth='timeseries.parquet',
    tslab_pth='timeserieslab.parquet',
    outputevents_pth='timeseriesoutputs.parquet')

tsp.run(reset_dir=False)

flp = mimic4_FLProcessor()

flp.run_labels()

dp = mimic4_DiagProcessor()

dp.run()
