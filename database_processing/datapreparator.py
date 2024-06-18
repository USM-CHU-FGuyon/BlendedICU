from pathlib import Path

import polars as pl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from database_processing.dataprocessor import DataProcessor


class DataPreparator(DataProcessor):
    def __init__(self, dataset, col_stayid):
        super().__init__(dataset)
        self.col_stayid = col_stayid

        self.chunksize = 10_000_000  # chunksize in csv reader
        self.inch_to_cm = 2.54
        self.lbs_to_kg = 0.454

    @staticmethod
    def _get_name_as_parquet(pth):
        pth = Path(pth).name
        extensions = "".join(Path(pth).suffixes)
        pth = pth.removesuffix(extensions)
        return pth + '.parquet'

    def _clip_time(self, df, col_offset='resultoffset'):
        idx = ((df[col_offset] < df[self.col_los])
               & (df[col_offset] > -self.preadm_anteriority*24*3600))
        return df.loc[idx].drop(columns=self.col_los)

    def _keepvars(self, df, col_variable, keepvars=None):
        if keepvars is None:
            return df
        return df.loc[df[col_variable].isin(keepvars)]

    def get_labels(self, lazy=False):
        if lazy:
            self.labels = pl.scan_parquet(self.labels_savepath)
            self.stays = self.labels.select(self.col_stayid).unique().collect().to_numpy()

        elif self.labels is None:
            self.labels = self.load(self.labels_savepath)
            self.stays = self.labels[self.col_stayid].unique()
        
        if self.labels is None:
            raise ValueError('Run gen_labels first !')


    @staticmethod
    def write_as_parquet(pth_src,
                         pth_tgt,
                         astype_dic={},
                         chunksize=1e8,
                         encoding=None,
                         sep=',',
                         src_is_multiple_parquet=False):
        print(f'Writing {pth_tgt}')
        Path(pth_tgt).parent.mkdir(exist_ok=True)
        
        def _read_chunks(pth_src, src_is_multiple_parquet):
            if src_is_multiple_parquet:
                return (pl.read_parquet(pth).to_pandas() for pth in Path(pth_src).iterdir())
            return pd.read_csv(pth_src,
                                chunksize=chunksize,
                                encoding=encoding,
                                sep=sep)
        
        df_chunks = _read_chunks(pth_src, src_is_multiple_parquet)


        for i, df in enumerate(df_chunks):
            astype_dic = {k:v for k, v in astype_dic.items() if k in df.columns}
            df = df.astype(astype_dic)
            table = pa.Table.from_pandas(df)
            if i == 0:
                pqwriter = pq.ParquetWriter(pth_tgt, table.schema)            
            pqwriter.write_table(table)
        
        if pqwriter:
            pqwriter.close()
        print('  -> Done')
        
    def _to_seconds(self, df, col, unit='second'):
        k = {'day': 86400,
             'hour': 3600,
             'minute': 60,
             'second': 1,
             'milisecond': 1/1000}

        df[col] = k[unit]*df[col]
        return df

    def split_and_save_chunks(self, df, savepath):
        self.chunk_idx = 0
        shuffled_stays = (self.labels[self.col_stayid]
                          .drop_duplicates()
                          .sample(frac=1, random_state=self.SEED))

        for start in range(0, len(shuffled_stays), self.n_patient_chunk):
            stay_chunk = shuffled_stays.iloc[start:start+self.n_patient_chunk]
            df_chunk = df.loc[df[self.col_stayid].isin(stay_chunk)]
            self.save_chunk(df_chunk, savepath)

    def save_chunk(self, chunk, savepath):
        Path(savepath).mkdir(exist_ok=True)
        chunk_savepath = f'{savepath}chunk_{self.chunk_idx}.parquet'
        self.save(chunk, chunk_savepath)
        self.chunk_idx += 1
    
    def pl_prepare_tstable(self,
                            lf,
                            itemid_label=None,
                            id_mapping={},
                            rename_dic={},
                            col_variable='variable',
                            col_intime=None,
                            col_measuretime=None,
                            col_offset=None,
                            col_mergestayid=None,
                            keepvars=None,
                            unit_offset='second',
                            unit_los='second',
                            cast_to_float=True,
                            col_value=None,
                            additional_expr=[]
                            ):
        
        if cast_to_float and col_value is None:
            raise RuntimeError('Specify col_value when ensure_float is True.')
        
        def _map_itemids(lf, itemid_label, id_mapping):
            if itemid_label:
                return lf.with_columns(
                    pl.col(itemid_label).replace(id_mapping).alias(col_variable)
                    )
            return lf
        
        def _clip_time(lf):
            lf = (lf.filter((pl.col(self.col_offset) < pl.col(self.col_los))
                      & (pl.col(self.col_offset) > pl.duration(days=-self.preadm_anteriority))
                      )
                  .drop(self.col_los)
                  )
            return lf
        
        def _compute_offset(lf, col_measuretime, col_intime):
            if col_measuretime:
                lf = (lf
                 .with_columns(
                    (pl.col(col_measuretime) - pl.col(col_intime)).alias(self.col_offset)
                    )
                .drop(col_measuretime, col_intime)
                )
            return lf
        
        def _offset_to_duration(lf, col_offset, alias, unit):
            if col_offset:
                t_mul = {
                    'millisecond': 1e-3,
                    'second': 1,
                    'minute': 60,
                    'hour': 60*60,
                    'day': 60*60*24
                    }
                
                lf = lf.with_columns(
                    pl.duration(seconds=pl.col(col_offset).mul(t_mul[unit])).alias(alias)
                    )
            return lf
        
        def _keepvars(lf, col_variable, keepvars=None):
            if keepvars is not None:
                return lf.filter(pl.col(col_variable).is_in(keepvars))

            return lf
        
        def _expressions(col_value, cast_to_float, additional_expr):
            '''
            Convert time to a number of seconds,
            Casts values to float32
            '''
            expr = [pl.col(self.col_offset).dt.seconds()] + additional_expr
            if cast_to_float:
                expr.append(pl.col(col_value).cast(pl.Float32, strict=False))
            return expr
        
        def _dropnull_values(lf, col_value):
            if col_value is not None:
                lf = lf.drop_nulls(col_value)
            return lf
        
        if col_mergestayid is None:
            col_mergestayid = self.col_stayid
        
        labels = self.labels.lazy()
        
        if not labels.columns:
            raise RuntimeError('labels is empty.')
        
        selec_cols = list({col for col in [col_mergestayid, self.col_stayid, self.col_los, col_intime]
                         if col in labels.columns})
        
        dropcols = [col_offset] if (col_offset!=self.col_offset) and (col_offset is not None) else []

        lf = (lf
              .rename(rename_dic)
              .pipe(_map_itemids,
                    itemid_label=itemid_label,
                    id_mapping=id_mapping)
              .join(labels.select(selec_cols), on=col_mergestayid, how='left')
              .pipe(_compute_offset,
                    col_measuretime=col_measuretime,
                    col_intime=col_intime)
              .pipe(_offset_to_duration,
                    col_offset=self.col_los,
                    alias=self.col_los,
                    unit=unit_los)
              .pipe(_offset_to_duration,
                    col_offset=col_offset,
                    alias=self.col_offset,
                    unit=unit_offset)
              .pipe(_clip_time)
              .pipe(_keepvars, col_variable=col_variable, keepvars=keepvars)
              .with_columns(
                  _expressions(col_value, cast_to_float, additional_expr)
                  )
              .pipe(_dropnull_values, col_value=col_value)
              .drop(dropcols)
            )
        
        return lf
    
