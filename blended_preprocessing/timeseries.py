from pathlib import Path

import numpy as np

from database_processing.timeseriespreprocessing import TimeseriesPreprocessing

np.random.seed(974)


class blendedicuTSP(TimeseriesPreprocessing):
    def __init__(self):
        super().__init__(dataset='blended')
        self.partially_processed_pths = self._get_ts_pths()
        self.labels = self.load(self.savepath+'preprocessed_labels.parquet')
        self.labels['ts_path'] = (self.labels.source_dataset.map(self.partiallyprocessed_ts_dirs)
                                  + self.labels.index
                                  + '.parquet')

    def _get_ts_pths(self):
        dirname = 'partially_processed_timeseries'
        return {d: Path(f'{self.pth_dic[d]}/{dirname}').iterdir()
                for d in self.datasets}

    def _make_pth_chunks(self):
        pths = (self.labels.sample(frac=1, random_state=self.SEED)
                    .ts_path.to_list())
        return map(list, np.array_split(pths, 1+len(pths)/1000))

    def run(self):
        """
        Applies the processing pipeline to the 'partially_processed_timeseries'
        files. These files are already resampled 
        """
        scalecols = [c for c in self.cols[self.dataset]
                     if c not in self.kept_med+['time', 'hour']]

        self.reset_dir()

        self.pth_chunks = self._make_pth_chunks()

        for i, pths in enumerate(self.pth_chunks):
            comp_quantiles = i == 0
            self.timeseries = self.load(pths)
            print(self.cols)
            timeseries = self.timeseries.loc[:, self.cols.index]

            timeseries = (timeseries.pipe(self.clip_and_norm,
                                          cols=scalecols,
                                          clip=self.TS_CLIP,
                                          normalize=self.TS_NORMALIZE,
                                          recompute_quantiles=comp_quantiles)
                                    .pipe(self._forward_fill)
                                    .pipe(self._fillna)
                                    .pipe(self._time_to_hours)
                                    .reset_index())

            self.save_timeseries(timeseries, self.preprocessed_ts_dir)

    def _time_to_hours(self, timeseries):
        """
        The time column is initially in seconds.
        """
        timeseries['time'] = (timeseries['time']/3600).astype(int)
        timeseries = timeseries.groupby(['patient', 'time']).mean()
        return timeseries
