import polars as pl

from database_processing.timeseriesprocessor import TimeseriesProcessor


class amsterdamTSP(TimeseriesProcessor):
    """
    Our processing of the amsterdam database handles 
    * 2 long tables: numericitems and listitems
    * 1 wide table: the gcs_score tables that was computed in 1_amsterdam.py
    * 1 medication table that was computed in 1_amsterdam.py
    """
    def __init__(self, ts_chunks, listitems_pth, gcs_scores_pth):
        super().__init__(dataset='amsterdam')
        self.lf_ts = self.scan(self.savepath+ts_chunks)
        self.lf_listitems = self.scan(self.savepath+listitems_pth)
        self.lf_medication = (self.scan(self.med_savepath)
                           .select('admissionid',
                                             'label',
                                             'original_drugname',
                                             'start',
                                             'end',
                                             'value'))
        print(self.savepath+gcs_scores_pth)
        self.gcs_scores = self.scan(self.savepath+gcs_scores_pth)
        
        self.colnames_med = {
            'col_id': 'admissionid',
            'col_var': 'label'}

        self.colnames_ts = {
            'col_id': 'admissionid',
            'col_var': 'item',
            'col_value': 'value',
            'col_time': self.col_offset}

        self.colnames_gcs = {
            'col_id': 'admissionid',
            'col_time': self.col_offset
            }

        self.tscols = self.colnames_ts.values()

        self.gcs_scores = self.harmonize_columns(self.gcs_scores, **self.colnames_gcs)

    def get_stays(self):
        self.stays = (self.scan(self.labels_savepath)
                      .select('admissionid')
                      .unique()
                      .collect()
                      .to_numpy()
                      .flatten())
        return self.stays
    
    def run(self, reset_dir=None):
        self.reset_dir(reset_dir)
        self.stays = self.get_stays()
        self.stay_chunks = self.get_stay_chunks()
        
        self.ts = self.harmonize_columns(self.lf_ts, **self.colnames_ts)
        self.listitems = self.harmonize_columns(self.lf_listitems, **self.colnames_ts)
        lf_med = self.harmonize_columns(self.lf_medication, **self.colnames_med)
        
        lf_ts = pl.concat([self.ts, self.listitems], how='diagonal_relaxed')
        
        for chunk_number, stays in enumerate(self.stay_chunks):

            self.ts_chunk = (self.filter_tables(lf_ts,
                                           kept_variables=self.kept_ts,
                                           kept_stays=stays)
                                  .collect(streaming=True)
                                  .to_pandas())
            
            med_chunk = (self.filter_tables(lf_med,
                                            kept_variables=self.kept_med,
                                            kept_stays=stays)
                                  .collect()
                                  .to_pandas())

            self.gcs_scores_chunk = (self.filter_tables(self.gcs_scores,
                                                        kept_stays=stays)
                                     .collect()
                                     .to_pandas())

            self.process_tables(ts_ver=self.ts_chunk,
                                ts_hor=self.gcs_scores_chunk,
                                med=med_chunk,
                                chunk_number=chunk_number)
