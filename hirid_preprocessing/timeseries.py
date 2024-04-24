from database_processing.timeseriesprocessor import TimeseriesProcessor


class hiridTSP(TimeseriesProcessor):
    """
    Our processing of the hirid database handles 
    * 2 long tables: timeseries and pharma
    * 1 medication table that was computed in 1_hirid.py
    """
    def __init__(self, ts_chunks, pharma_chunks):
        super().__init__(dataset='hirid')
        self.lf_ts = self.scan(self.savepath+ts_chunks)
        self.lf_med = self.scan(self.savepath+pharma_chunks)

        self.med_colnames = {'col_id': 'admissionid',
                             'col_var': 'label'}

        self.ts_colnames = {'col_id': 'admissionid',
                            'col_var': 'variable',
                            'col_value': 'value',
                            'col_time': self.col_offset}

        self.loadcols = self.ts_colnames.values()

    def _get_stays(self):
        stays = self.scan(self.labels_savepath).select('admissionid').unique().collect().to_numpy().flatten()
        return stays
    
    def run(self, reset_dir=None):

        self.reset_dir(reset_dir)
        self.stays = self._get_stays()
        self.stay_chunks = self.get_stay_chunks()
        kept_variables = (self.kept_ts+['Body weight', 'Body height measure'])

        self.lf_ts = self.harmonize_columns(self.lf_ts, **self.ts_colnames)
        self.lf_med = self.harmonize_columns(self.lf_med, **self.med_colnames)

        for chunk_number, stay_chunk in enumerate(self.stay_chunks):
            print(f'Chunk {chunk_number}')

            ts = (self.filter_tables(self.lf_ts,
                                    kept_variables=kept_variables,
                                    kept_stays=stay_chunk)
                  .collect(streaming=True)
                  .to_pandas())

            med = (self.filter_tables(self.lf_med,
                                     kept_variables=self.kept_med,
                                     kept_stays=stay_chunk)
                   .collect()
                   .to_pandas())

            self.process_tables(ts, med=med, chunk_number=chunk_number)
