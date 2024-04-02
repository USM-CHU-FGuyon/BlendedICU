"""
Processing of the BlendedICU dataset. 
The raw timeseries data of the blendedICU dataset is already computed at step 2
Only the labels processing needs to be done to get the complete raw dataset.

blendedicuTSP creates the processed blendedICU dataset by adding resampling, 
missing values imputation and clipping&normalization steps. All these steps 
are optional and customizable using the config.json file.
"""
from blended_preprocessing.timeseries import blendedicuTSP
from blended_preprocessing.flat_and_labels import blended_FLProcessor
from blended_preprocessing.diagnoses import blended_DiagProcessor

flp = blended_FLProcessor(datasets=['mimic4',
                                    'mimic3',
                                    'hirid',
                                    'amsterdam',
                                    'eicu'])
flp.run_flat_and_labels()

dp = blended_DiagProcessor(datasets=['mimic4'])

dp.run()

tsp = blendedicuTSP(compute_index=False)

tsp.run(reset_dir=False)


