import polars as pl

from database_processing.timeseriesprocessor import TimeseriesProcessor


class mimic4TSP(TimeseriesProcessor):
    """
    Our processing of the hirid database handles 
    * 3 long tables: timeseries, timeserieslab, outputevents
    * 1 medication table that was computed in 1_mimic.py
    """
    def __init__(self,
                 med_pth,
                 ts_pth,
                 tslab_pth,
                 outputevents_pth):
        super().__init__(dataset='mimic4')


        self.labels = self.load(self.labels_savepath)
        self.lf_medication = self.scan(self.savepath+med_pth)
        self.lf_timeseries = self.scan(self.savepath+ts_pth)
        self.lf_timeseries_lab = self.scan(self.savepath+tslab_pth)
        self.lf_outputevents = self.scan(self.savepath+outputevents_pth).select('stay_id',
                 'offset',
                 'value',
                 'label')

        self.colnames_med = {
            'col_id': 'stay_id',
            'col_var': 'label',
        }

        self.colnames_ts = {
            'col_id': 'stay_id',
            'col_var': 'label',
            'col_value': 'valuenum',
            'col_time': self.col_offset
        }

        self.colnames_lab = {
            'col_id': 'stay_id',
            'col_var': 'label',
            'col_value': 'valuenum',
            'col_time': self.col_offset
        }

        self.colnames_outputevents = {
            'col_id': 'stay_id',
            'col_var': 'label',
            'col_value': 'value',
            'col_time': self.col_offset
        }

    def get_stays(self):
        return self.labels.stay_id.unique()

    def run(self, reset_dir=None):
        self.reset_dir(reset_dir)
        
        self.lf_outputevents = self.harmonize_columns(self.lf_outputevents,
                                                   **self.colnames_outputevents)
        
        self.lf_timeseries_lab = self.harmonize_columns(self.lf_timeseries_lab,
                                                   **self.colnames_lab)
        
        self.lf_timeseries = self.harmonize_columns(self.lf_timeseries,
                                                   **self.colnames_ts)
        
        self.lf_medication = self.harmonize_columns(self.lf_medication,
                                                   **self.colnames_med)
        
        lf_ts = pl.concat([self.lf_timeseries,
                           self.lf_timeseries_lab,
                           self.lf_outputevents],
                             how='diagonal')

        self.stays = self.get_stays()
        self.stay_chunks = self.get_stay_chunks()

        for chunk_number, stay_chunk in enumerate(self.stay_chunks):
            
            med = (self.filter_tables(self.lf_medication,
                                    kept_variables=self.kept_med,
                                    kept_stays=stay_chunk)
                   .collect()
                   .to_pandas())
            
            ts = (self.filter_tables(lf_ts,
                                    kept_variables=self.kept_ts,
                                    kept_stays=stay_chunk)
                  .collect()
                  .to_pandas())

            self.process_tables(ts,
                                med=med,
                                chunk_number=chunk_number)
