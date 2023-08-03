from pathlib import Path

from database_processing.dataprocessor import DataProcessor


class DataPreparator(DataProcessor):
    def __init__(self, dataset, col_stayid):
        super().__init__(dataset)
        self.col_stayid = col_stayid

        self.chunksize = 10_000_000  # chunksize in csv reader
        self.inch_to_cm = 2.54
        self.lbs_to_kg = 0.454

    def _clip_time(self, df, col_offset='resultoffset'):
        idx = ((df[col_offset] < df[self.col_los])
               & (df[col_offset] > -self.preadm_anteriority*24*3600))
        return df.loc[idx].drop(columns=self.col_los)

    def _keepvars(self, df, col_variable, keepvars=None):
        if keepvars is None:
            return df
        return df.loc[df[col_variable].isin(keepvars)]

    def get_labels(self):
        if self.labels is None:
            self.labels = self.load(self.labels_savepath)
        if self.labels is None:
            raise ValueError('Run gen_labels first !')

        self.stays = self.labels[self.col_stayid].unique()

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

    def prepare_tstable(self,
                        table,
                        col_offset,
                        col_variable='variable',
                        unit_offset='second',
                        keepvars=None,
                        col_intime=None,
                        col_mergestayid=None):

        if col_mergestayid is None:
            col_mergestayid = self.col_stayid

        keepcols = self.labels.columns.isin([col_mergestayid,
                                             self.col_stayid,
                                             self.col_los,
                                             col_intime])

        los = self.labels.loc[:, keepcols]
        return (table.merge(los,
                            on=col_mergestayid,
                            how='left')
                .pipe(self.compute_offset,
                      col_measuretime=col_offset,
                      col_intime=col_intime)
                .pipe(self._to_seconds,
                      col=self.col_los,
                      unit=self.unit_los)
                .pipe(self._to_seconds,
                      col=col_offset,
                      unit=unit_offset)
                .pipe(self._clip_time,
                      col_offset=col_offset)
                .pipe(self._keepvars,
                      col_variable=col_variable,
                      keepvars=keepvars)
                .dropna()
                .astype({self.col_stayid: int})
                .sort_values(self.col_stayid))
