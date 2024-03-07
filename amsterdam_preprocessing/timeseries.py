import pandas as pd

from database_processing.timeseriespreprocessing import TimeseriesPreprocessing


class amsterdamTSP(TimeseriesPreprocessing):
    """
    Our processing of the amsterdam database handles 
    * 2 long tables: numericitems and listitems
    * 1 wide table: the gcs_score tables that was computed in 1_amsterdam.py
    * 1 medication table that was computed in 1_amsterdam.py
    """
    def __init__(self, ts_chunks, listitems_pth, gcs_scores_pth):
        super().__init__(dataset='amsterdam')
        self.ts_chunks = self.ls(self.savepath+ts_chunks)
        self.listitems = self.load(self.savepath+listitems_pth)
        self.medication = self.load(self.med_savepath,
                                    columns=['admissionid',
                                             'label',
                                             'original_drugname',
                                             'start',
                                             'end',
                                             'value'])
        self.gcs_scores = self.load(self.savepath+gcs_scores_pth)
        
        self.colnames_med = {
            'col_id': 'admissionid',
            'col_var': 'label',
            'col_value': 'value'}

        self.colnames_ts = {
            'col_id': 'admissionid',
            'col_var': 'item',
            'col_value': 'value',
            'col_time': 'measuredat'}

        self.colnames_gcs = {
            'col_id': 'admissionid',
            'col_time': 'measuredat'
            }

        self.tscols = self.colnames_ts.values()

        self.gcs_scores = self.filter_tables(self.gcs_scores,
                                             **self.colnames_gcs)

    def _get_chunk(self, table, chunk_idx):
        return table.loc[table.admissionid.isin(chunk_idx)]
    
    def run(self):
        self.reset_dir()
        for chunk_number, data_pth in enumerate(self.ts_chunks):
            numericitems_chunk = self.load(data_pth, columns=self.tscols)

            chunk_ids = numericitems_chunk.admissionid.unique()

            listitems_chunk = self._get_chunk(self.listitems, chunk_ids)
            medication_chunk = self._get_chunk(self.medication, chunk_ids)
            
            timeseries_chunk = pd.concat([numericitems_chunk, listitems_chunk])

            self.ts_table = self.filter_tables(timeseries_chunk,
                                               self.kept_ts,
                                               **self.colnames_ts)

            self.med_table = self.filter_tables(medication_chunk,
                                                self.kept_med,
                                                **self.colnames_med)

            gcs_scores_chunk_idx = self.gcs_scores.patient.isin(self.ts_table.patient)
            self.gcs_scores_chunk = self.gcs_scores.loc[gcs_scores_chunk_idx]

            self.process_tables(ts_ver=self.ts_table,
                                ts_hor=self.gcs_scores_chunk,
                                med=self.med_table,
                                chunk_number=chunk_number)
