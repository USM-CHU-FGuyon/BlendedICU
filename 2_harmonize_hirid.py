"""
This script lauches the timeseriesprocessing (tsp) and the flat and labels 
processing (flp) for the hirid database. 
Note that this produces the 'raw' data of the BlendedICU dataset.
The preprocessed BlendedICU dataset will then be obtained with 3_blendedICU.py
Approximate running time: 2min
"""
from hirid_preprocessing.flat_and_labels import Hir_FLProcessing
from hirid_preprocessing.timeseries import hiridTSP

tsp = hiridTSP(ts='timeseries.parquet',
                pharma='medication.parquet')

tsp.run_harmonization()

flp = Hir_FLProcessing()

flp.run_labels()
