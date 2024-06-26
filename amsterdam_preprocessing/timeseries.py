from database_processing.timeseriesprocessor import TimeseriesProcessor


class amsterdamTSP(TimeseriesProcessor):
    """
    Our processing of the amsterdam database handles 
    * 2 long tables: numericitems and listitems
    * 1 wide table: the gcs_score tables that was computed in 1_amsterdam.py
    * 1 medication table that was computed in 1_amsterdam.py
    """
    def __init__(self, ts_pth, listitems_pth, gcs_scores_pth):
        super().__init__(dataset='amsterdam')
        self.lf_ts = self.scan(self.savepath+ts_pth)
        self.lf_listitems = self.scan(self.savepath+listitems_pth)
        self.lf_medication = self.scan(self.med_savepath)

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
    
    def run_harmonization(self):

        lf_ts = self.harmonize_columns(self.lf_ts, **self.colnames_ts)
        lf_med = self.harmonize_columns(self.lf_medication, **self.colnames_med)
        
        lf_ts = self.filter_tables(lf_ts, kept_variables=self.kept_ts,)

        self.timeseries_to_long(lf_long=lf_ts,
                                lf_wide=self.gcs_scores)
        self.medication_to_long(lf_med)
    
    def run_for_preprocessed(self, reset_dir=None):
        raise UserWarning("This function is not maintained. It should be replaced"
                          "by a cleaner/faster alternative in the future.\n"
                          "Contributions welcome.")
        
        
        self.reset_dir(reset_dir)

        lf_ts = self.harmonize_columns(self.lf_ts, **self.colnames_ts)
        lf_med = self.harmonize_columns(self.lf_medication, **self.colnames_med)
        
        lf_med_formatted = self.pl_format_meds(lf_med)

        self.stays = self.get_stays()
        self.stay_chunks = self.get_stay_chunks(n_patient_chunk=5_000)

        for chunk_number, stays in enumerate(self.stay_chunks):
            print(f'Chunk {chunk_number}')
            self.ts_chunk = (self.filter_tables(lf_ts,
                                                kept_variables=self.kept_ts,
                                                kept_stays=stays)
                             .collect(streaming=True
                                      ))
            
            self.med_formatted_chunk = (self.filter_tables(lf_med_formatted,
                                            kept_variables=self.kept_med,
                                            kept_stays=stays))

            self.gcs_scores_chunk = (self.filter_tables(self.gcs_scores,
                                                        kept_stays=stays))

            ts_formatted_chunk = (self.pl_format_timeseries(lf_tsver=self.ts_chunk.lazy(),
                                                            lf_tshor=self.gcs_scores_chunk,
                                                            chunk_number=chunk_number)
                                  )

            self.newprocess_tables(ts_formatted_chunk,
                                   med=self.med_formatted_chunk.collect().to_pandas(),
                                   chunk_number=chunk_number)
