from pathlib import Path

import numpy as np
import pandas as pd

from database_processing.timeseriesprocessor import TimeseriesProcessor


class blendedicuTSP(TimeseriesProcessor):
    def __init__(self, compute_index=False):
        '''
        Use compute_index=True if timeseries files may have changed location,
        or have been deleted or created between the 2_{dataset} step and the
        3_Blended step.
        if compute_index is False, index files from each directory in 
        partially_processed_timeseries will be read and concatenated.
        '''
        super().__init__(dataset='blended')
        self.ts_pths = self.get_ts_pths(self.partiallyprocessed_ts_dir,
                                         compute_index)

    
    def get_ts_pths(self, ts_dir, compute_index=False, sample=None):
        labels = self.load(self.savepath+'preprocessed_labels.parquet')
        
        if compute_index:
            index_df = self._build_full_index(ts_dir)
        else:
            index_df = self._read_full_index(ts_dir)

        kwargs = {'frac': 1} if sample is None else {'n': sample}
    
        ts_pths = (labels[['uniquepid', 'source_dataset']]
                   .join(index_df[['ts_pth']])
                   .dropna(subset='ts_pth')
                   .groupby('source_dataset')
                   .sample(**kwargs, random_state=self.SEED))
        index_pth = self._get_index_pth(ts_dir)
        
        print(f'Saving {index_pth}')
        ts_pths.to_csv(index_pth, sep=';')
        return ts_pths
        
    def _read_full_index(self, ts_dir):
        index_dfs = [self.read_index(p) for p in Path(ts_dir).iterdir() if p.is_dir()]
        index_df = pd.concat(index_dfs)
        return index_df

    def _build_full_index(self, ts_dir):
        index_dfs = [self.build_index(p) for p in Path(ts_dir).iterdir() if p.is_dir()]
        index_df = pd.concat(index_dfs)
        return index_df

    def _time_to_hours(self, timeseries):
        """
        The time column is initially in seconds.
        """
        timeseries['time'] = timeseries['time'].div(3600).astype(int)
        return timeseries

    def _shuffled_ts_pths(self):
        return (self.ts_pths.sample(frac=1, random_state=self.SEED)
                    .ts_pth.to_list())

    def _make_pth_chunks(self):
        pths = self._shuffled_ts_pths()
        return map(list, np.array_split(pths, 1+len(pths)/1000))

    def run(self):
        """
        Applies the processing pipeline to the 'partially_processed_timeseries'
        files. These files are already resampled 
        """
        scalecols = [c for c in self.numeric_ts if c not in ['time', 'hour']]

        self.reset_dir()

        self.pth_chunks = self._make_pth_chunks()

        for chunk_number, pths in enumerate(self.pth_chunks):
            comp_quantiles = chunk_number == 0
            timeseries = self.load(pths, verbose=False)
            timeseries = timeseries.loc[:, self.cols.index]

            timeseries = (timeseries.pipe(self.clip_and_norm,
                                          cols=scalecols,
                                          clip=self.TS_CLIP,
                                          normalize=self.TS_NORMALIZE,
                                          recompute_quantiles=comp_quantiles)
                                    .pipe(self._forward_fill)
                                    .pipe(self._fillna)
                                    .pipe(self._time_to_hours)
                                    .reset_index())

            ts_savepath = f'{self.preprocessed_ts_dir}/{chunk_number}/'
            self.save_timeseries(timeseries, ts_savepath)
