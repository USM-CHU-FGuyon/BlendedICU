from database_processing.timeseriespreprocessing import TimeseriesPreprocessing


class hiridTSP(TimeseriesPreprocessing):
    """
    Our processing of the hirid database handles 
    * 2 long tables: timeseries and pharma
    * 1 medication table that was computed in 1_hirid.py
    """
    def __init__(self, ts_chunks, pharma_chunks):
        super().__init__(dataset='hirid')
        self.ts_files = self.ls(self.parquet_pth+ts_chunks)
        self.pharma_files = self.ls(self.parquet_pth+pharma_chunks)

        self.med_colnames = {'col_id': 'admissionid',
                             'col_var': 'label',
                             'col_value': 'value',
                             'col_time': 'offset'}

        self.ts_colnames = {'col_id': 'admissionid',
                            'col_var': 'variable',
                            'col_value': 'value',
                            'col_time': 'offset'}

        self.loadcols = self.ts_colnames.values()

    def run(self):

        self.reset_dir()

        kept_variables = (self.kept_ts+['Body weight', 'Body height measure'])

        for chunk_number, (ts_pth, pharma_pth) in enumerate(zip(self.ts_files, self.pharma_files)):
            self.timeseries = self.load(ts_pth, columns=self.loadcols)

            self.pharma = self.load(pharma_pth)

            ts = self.filter_tables(self.timeseries,
                                    kept_variables,
                                    **self.ts_colnames)
            self.ts = ts

            med = self.filter_tables(self.pharma,
                                     self.kept_med,
                                     **self.med_colnames)

            self.process_tables(ts, med=med, chunk_number=chunk_number)
