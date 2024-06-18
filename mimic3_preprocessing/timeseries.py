import polars as pl

from database_processing.timeseriesprocessor import TimeseriesProcessor


class mimic3TSP(TimeseriesProcessor):
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
        super().__init__(dataset='mimic3')

        self.lf_medication = self.scan(self.savepath+med_pth)

        self.lf_timeseries = self.scan(self.savepath+ts_pth)
        self.lf_timeseries_lab = self.scan(self.savepath+tslab_pth)
        self.lf_outputevents = (self.scan(self.savepath+outputevents_pth)
                             .select('ICUSTAY_ID',
                                     'offset',
                                     'VALUE',
                                     'LABEL')
                             )

        self.colnames_med = {
            'col_id': 'ICUSTAY_ID',
            'col_var': 'label',
            'col_value': 'value',
        }

        self.colnames_ts = {
            'col_id': 'ICUSTAY_ID',
            'col_var': 'LABEL',
            'col_value': 'VALUENUM',
            'col_time': self.col_offset
        }

        self.colnames_lab = {
            'col_id': 'ICUSTAY_ID',
            'col_var': 'LABEL',
            'col_value': 'VALUENUM',
            'col_time': self.col_offset
        }

        self.colnames_outputevents = {
            'col_id': 'ICUSTAY_ID',
            'col_var': 'LABEL',
            'col_value': 'VALUE',
            'col_time': self.col_offset
        }

    def _get_stays(self):
        return (self.scan(self.labels_savepath)
                .select('ICUSTAY_ID')
                .unique()
                .collect().to_numpy().flatten())
        
    def run(self, reset_dir=None):
        self.reset_dir(reset_dir)
        self.stays = self._get_stays()
        self.stay_chunks = self.get_stay_chunks(n_patient_chunk=10_000)
        
        self.lf_medication = self.harmonize_columns(self.lf_medication,
                                                    **self.colnames_med)
        self.timeseries = self.harmonize_columns(self.lf_timeseries,
                                                 **self.colnames_ts)
        self.timeseries_lab = self.harmonize_columns(self.lf_timeseries_lab,
                                                     **self.colnames_lab)
        self.outputevents = self.harmonize_columns(self.lf_outputevents,
                                                   **self.colnames_outputevents)
        
        self.lf_timeser = pl.concat([self.timeseries,
                                     self.timeseries_lab,
                                     self.outputevents],
                                    how='diagonal')
        
        lf_med = self.filter_tables(self.lf_medication,
                                    kept_variables=self.kept_med)
        
        lf_ts = self.filter_tables(self.lf_timeser,
                                   kept_variables=self.kept_ts)
        
        lf_formatted_ts = self.pl_format_timeseries(lf_ts)
        lf_formatted_med = self.pl_format_meds(lf_med)
        
        for chunk_number, stay_chunk in enumerate(self.stay_chunks):
            
            lf_med_chunked = self.filter_tables(lf_formatted_med,
                                                kept_stays=stay_chunk)
            
            lf_formatted_ts_chunked = self.filter_tables(lf_formatted_ts,
                                                         kept_stays=stay_chunk)
            
            self.df_med_chunked = (lf_med_chunked
                      .collect())
            
            self.df_ts_chunked = (lf_formatted_ts_chunked
                             .collect())

            self.newprocess_tables(self.df_ts_chunked,
                                med=self.df_med_chunked,
                                chunk_number=chunk_number)
