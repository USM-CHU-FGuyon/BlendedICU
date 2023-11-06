from pathlib import Path
from functools import reduce
import operator
import json
import shutil

import pandas as pd


class DataProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.SEED = 974
        self.n_patient_chunk = 1000
        self.pth_dic = self._read_json('paths.json')
        self.config = self._read_json('config.json')

        self.aux_pth = self.pth_dic['auxillary_files']
        self.savepath = self.pth_dic[self.dataset]
        self.voc_pth = self.pth_dic['vocabulary']
        self.user_input_pth = self.pth_dic['user_input']
        self.parquet_pth = f'{self.savepath}{self.dataset}_parquet/'
        Path(self.parquet_pth).mkdir(exist_ok=True, parents=True)
        try:
            self.source_pth = self.pth_dic[f'{self.dataset}_source_path']
        except KeyError:
            self.source_pth = None
        
        self.med_file = self.aux_pth+'medications_v10.json'
        self.unittype_file = self.user_input_pth+'unit_type_v2.json'
        self.dischargeloc_file = self.user_input_pth+'discharge_location_v2.json'
        self.admissionorigin_file = self.user_input_pth+'admission_origins_v2.json'

        self.datasets = ('eicu', 'mimic', 'mimic3', 'hirid', 'amsterdam')
        
        self.ohdsi_med = self._read_json(self.med_file)
        self.med_concept_id = self._med_concept_id_mapping()
        self.formatted_ts_dir_blended = self.pth_dic['blended']+'formatted_timeseries/'

        blended_formatted_ts = self.pth_dic["blended"] + '/formatted_timeseries/'
        blended_formatted_med = self.pth_dic["blended"] + '/formatted_medications/'

        self.formatted_ts_dirs = ({d: f'{blended_formatted_ts}/{d}/' for d in self.datasets}
                                  | {'blended': blended_formatted_ts})
        
        self.formatted_med_dirs = ({d: f'{blended_formatted_med}/{d}/' for d in self.datasets}
                                   | {'blended': blended_formatted_med})
        
        self.formatted_ts_dir = self.formatted_ts_dirs[self.dataset]
        self.formatted_med_dir = self.formatted_med_dirs[self.dataset]
        self.preprocessed_ts_dir = self.savepath+'preprocessed_timeseries/'
        self.preprocessed_ts_dirs = {d: self.pth_dic[d]+'preprocessed_timeseries/' for d in self.datasets}
        self.partiallyprocessed_ts_dir = self.savepath+'partially_processed_timeseries/'
        self.partiallyprocessed_ts_dirs = {d: self.pth_dic[d]+'partially_processed_timeseries/' for d in self.datasets}

        self.time_col = 'time'
        self.idx_col = 'patient'
        self.mor_col = 'mortality'
        self.los_col = 'lengthofstay'

        self.lower_los = self.config['lower_los']['value']
        self.upper_los = self.config['upper_los']['value']
        self.preadm_anteriority = self.config['preadm_anteriority']['value']
        self.drug_exposure_time = self.config['drug_exposure_time']['value']
        self.flat_hr_from_adm = self.config['flat_hr_from_adm']['value']
        self.TS_FILL_MEDIAN = self.config['TS_FILL_MEDIAN']['value']
        self.TS_NORMALIZE = self.config['TS_NORMALIZE']['value']
        self.TS_CLIP = self.config['TS_CLIP']['value']
        self.FLAT_FILL_MEDIAN = self.config['FLAT_FILL_MEDIAN']['value']
        self.FLAT_NORMALIZE = self.config['FLAT_NORMALIZE']['value']
        self.FLAT_CLIP = self.config['FLAT_CLIP']['value']
        self.FORWARD_FILL = self.config['FORWARD_FILL']['value']

        self.admission_origins = self._load_mapping(self.admissionorigin_file)
        self.discharge_locations = self._load_mapping(self.dischargeloc_file)
        self.unit_types = self._load_mapping(self.unittype_file)
        self.med_mapping = self._load_med_mapping()
        self.clipping_quantiles = None
        self.labels = None
        self.med_savepath = f'{self.parquet_pth}/medication.parquet'
        self.labels_savepath = f'{self.parquet_pth}/labels.parquet'
        self.flat_savepath = f'{self.parquet_pth}/flat_features.parquet'

    def _concat(self, df1, df2):
        return ([df1.copy()] if df2.empty 
                else [df2.copy()] if df1.empty
                else [pd.concat([df1, df2])])
    
    def concat(self, df_list):
        if not isinstance(df_list, list):
            raise ValueError('Argument should be a list of DataFrames')
        while len(df_list)>1:
            df_list = self._concat(*df_list[:2]) + df_list[2:]
        return df_list[0]

    def _med_concept_id_mapping(self):
        dic = {ing: m['blended'] for ing, m in self.ohdsi_med.items()}
        return pd.Series(dic)

    def _read_json(self, pth):
        return json.load(open(pth, 'r'))

    def load(self, pth, verbose=True, **kwargs):
        """
        alias for pd.read_parquet
        """
        if verbose:
            print(f'Loading {pth}')
        return pd.read_parquet(pth, **kwargs)

    def save(self, df, savepath, pyarrow_schema=None):
        """
        convenience function: save safely a file to parquet by creating the 
        parent directory if it does not exist.
        """
        Path(savepath).parents[0].mkdir(parents=True, exist_ok=True)
        print(f'   saving {savepath}')
        df.to_parquet(savepath, schema=pyarrow_schema)
        return df

    def rmdir(self, pth):
        """
        alias for shutil.rmtree that does not raise an exception if the path
        to remove does not exist.
        """
        try:
            shutil.rmtree(pth)
        except FileNotFoundError:
            pass

    def rglob(self, pth, reg):
        """
        alias for using rglob.
        """
        return [*Path(pth).rglob(reg)]

    def reset_dir(self):
        pth_preprocessed_ts = Path(self.preprocessed_ts_dir)
        pth_partprocessed_ts = Path(self.partiallyprocessed_ts_dir)
        proceed = input(f'Delete contents of {pth_preprocessed_ts}  and '
                        f'{pth_partprocessed_ts} ? [n], y')
        if proceed == 'y':
            self.rmdir(pth_preprocessed_ts)
            self.rmdir(pth_partprocessed_ts)
            pth_preprocessed_ts.mkdir(parents=True)
            pth_partprocessed_ts.mkdir(parents=True)

    def _load_mapping(self, pth):
        """
        Loads the mappings from the user_input folder, and converts them to 
        a practical format for further usage.
        """
        jsonfile = self._read_json(pth)
        mapping = ({v: key for v in val} for key, val in jsonfile.items())
        return reduce(operator.ior, mapping)

    def _load_med_mapping(self):
        """
        Loads the medication mapping that was created by 0_prepare_files.py
        For source databases, each entry is a list of labels,
        The blendedicu entries are the unique concept_id as an int.
        """
        if self.dataset != 'blended':
            mapping = ({v: key for v in val[self.dataset]}
                       for key, val in self.ohdsi_med.items())
            return reduce(operator.ior, mapping)
        return {val[self.dataset]: key for key, val in self.ohdsi_med.items()}
        
    def reset_chunk_idx(self):
        self.chunk_idx = 0

    def compute_offset(self, df, col_measuretime, col_intime):
        """
        Some databases are already "offset" based: admission is taken as the 
        origin of times for each stay. 
        This functions is used to convert all databases to this format.
        """
        if col_intime is None:
            return df
        df[col_intime] = pd.to_datetime(df[col_intime])
        df[col_measuretime] = pd.to_datetime(df[col_measuretime])
        df[col_measuretime] = (df[col_measuretime] -
                               df[col_intime]).dt.total_seconds()
        return df.drop(columns=[col_intime])

    def medianfill(self, df, cols):
        """
        This function is used in the flat and labels processing as an option
        to impute missing data.
        """
        df[cols] = df[cols].fillna(df[cols].median())
        return df

    def compute_quantiles(self, df):
        """
        Computes the 5%, 95% quantiles and median of the input dataframe.
        This can used in the clipping step and in the missing values imputation
        step.
        """
        quantiles = df.quantile([0.05, 0.5, 0.95])

        mins = quantiles.loc[0.05]
        maxs = quantiles.loc[0.95]
        meds = quantiles.loc[0.5]

        consant_values_idx = mins == maxs

        mins[consant_values_idx] = meds[consant_values_idx]*0.5
        maxs[consant_values_idx] = meds[consant_values_idx]*2

        self.clipping_quantiles = mins, meds, maxs

    def clip_and_norm(self,
                      df,
                      cols=None,
                      clip=True,
                      normalize=False,
                      recompute_quantiles=False):
        '''
        Clipping and normalization step.
        applies clipping and/or normalization to a set of specified colums in 
        a dataframe.
        df:
            pd.DataFrame
        cols:
            list of columns from df
        clip:
            bool, whether to apply clipping for removing erroneous outliers.
        normalize:
            bool, whether to apply normalization
        recompute_quantiles:
            bool, wheter to compute the quantiles of the variables. When
            data by chunk,it may be useful to compute the quantiles in the 
            first chunk and use those quantiles in the processing of all
            chunks, which motivated this option.
        '''
        if cols is None:
            cols = df.columns

        if (self.clipping_quantiles is None) or recompute_quantiles:
            self.compute_quantiles(df[cols])

        mins, meds, maxs = self.clipping_quantiles

        if clip and normalize:
            df[cols] = (2 * (df[cols]-mins) / (maxs-mins) - 1).clip(lower=-4,
                                                                    upper=4)
            
            if recompute_quantiles:
                # if quantiles were recomputed and normalization is on: median should be rescaled for 
                # forward filling in the next step
                self.clipping_quantiles = mins, 2*(meds-mins) / (maxs-mins) - 1, maxs 

        elif clip:
            low = meds - 4*(meds - mins)
            up = meds + 4*(maxs - meds)
            df[cols] = df[cols].clip(lower=low, upper=up, axis=1)
        elif normalize:
            df[cols] = (2 * (df[cols] - mins) / (maxs - mins) - 1)
        return df
