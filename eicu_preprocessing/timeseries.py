import polars as pl

from database_processing.timeseriesprocessor import TimeseriesProcessor


class eicuTSP(TimeseriesProcessor):
    """
    Our processing of the eicu database handles 
    * 4 long tables: lab, resp, nurse, inputoutput
    * 2 wide tables: periodic, aperiodic
    * 1 medication table that was computed in 1_eicu.py
    """
    def __init__(self,
                 lab_pth,
                 resp_pth,
                 nurse_pth,
                 aperiodic_pth,
                 periodic_pth,
                 inout_pth):
        super().__init__(dataset='eicu')

        self.lf_med = self.scan(self.med_savepath)
        self.labels = self.scan(self.labels_savepath)
        self.flat = self.scan(self.flat_savepath).collect().to_pandas()
        self.lf_tslab = self.scan(self.savepath+lab_pth)
        self.lf_tsresp = self.scan(self.savepath+resp_pth)
        self.lf_tsnurse = self.scan(self.savepath+nurse_pth)
        self.lf_tsaperiodic = self.scan(self.savepath+aperiodic_pth)
        self.lf_tsperiodic = self.scan(self.savepath+periodic_pth)
        self.lf_tsinout = self.scan(self.savepath+inout_pth)

        self.colnames_lab = {
            'col_id': 'patientunitstayid',
            'col_var': 'labname',
            'col_value': 'labresult',
            'col_time': self.col_offset
        }

        self.colnames_resp = {
            'col_id': 'patientunitstayid',
            'col_var': 'respchartvaluelabel',
            'col_value': 'respchartvalue',
            'col_time': self.col_offset
        }

        self.colnames_inout = {
            'col_id': 'patientunitstayid',
            'col_var': 'celllabel',
            'col_value': 'cellvaluenumeric',
            'col_time': self.col_offset
        }

        self.colnames_nurse = {
            'col_id': 'patientunitstayid',
            'col_var': 'nursingchartcelltypevallabel',
            'col_value': 'nursingchartvalue',
            'col_time': self.col_offset
        }

        self.colnames_med = {
            'col_id': 'patientunitstayid',
            'col_var': 'label',
            'col_value': 'value'
        }

        self.colnames_tshor = {
            'col_id': 'patientunitstayid',
            'col_time': self.col_offset
        }


    def _get_stays(self):
        return self.labels.select('patientunitstayid').unique().collect().to_numpy().flatten()
    
    
    def run_harmonization(self):
        lf_med = self.harmonize_columns(self.lf_med,
                                             **self.colnames_med)

        lf_tsresp = self.harmonize_columns(self.lf_tsresp,
                                                **self.colnames_resp)
        
        lf_tsnurse = self.harmonize_columns(self.lf_tsnurse,
                                                 **self.colnames_nurse)
        
        lf_tsinout = self.harmonize_columns(self.lf_tsinout,
                                            **self.colnames_inout)
        
        lf_tslab = self.harmonize_columns(self.lf_tslab,
                                          **self.colnames_lab)
        
        lf_tsperiodic = self.harmonize_columns(self.lf_tsperiodic,
                                               **self.colnames_tshor)
        
        lf_tsaperiodic = self.harmonize_columns(self.lf_tsaperiodic,
                                                **self.colnames_tshor)

        lf_ts_ver = pl.concat([
            lf_tslab,
            lf_tsresp,
            lf_tsnurse,
            lf_tsinout,
            ])

        print('Collecting tsperiodic and tsaperiodic', end='')
        #not collecting here causes errors in the following methods. polars bug ?
        lf_ts_hor = (pl.concat([lf_tsperiodic, lf_tsaperiodic],
                              how='diagonal',
                              rechunk=True)
                     .collect(streaming=True))
        print('  -> Done')
        
        lf_ts_ver = self.filter_tables(lf_ts_ver,
                                    kept_variables=self.kept_ts)
        
        self.timeseries_to_long(lf_ts_ver, lf_ts_hor.lazy())
        self.medication_to_long(lf_med)
        
    
    def run_preprocessing(self, reset_dir=None):
        raise UserWarning("This function is not maintained. It should be replaced"
                          "by a cleaner/faster alternative in the future.\n"
                          "Contributions welcome.")
        self.reset_dir(reset_dir)

        
        lf_med = self.harmonize_columns(self.lf_med,
                                             **self.colnames_med)

        lf_tsresp = self.harmonize_columns(self.lf_tsresp,
                                                **self.colnames_resp)
        
        lf_tsnurse = self.harmonize_columns(self.lf_tsnurse,
                                                 **self.colnames_nurse)
        
        lf_tsinout = self.harmonize_columns(self.lf_tsinout,
                                            **self.colnames_inout)
        
        lf_tslab = self.harmonize_columns(self.lf_tslab,
                                          **self.colnames_lab)
        
        lf_tsperiodic = self.harmonize_columns(self.lf_tsperiodic,
                                               **self.colnames_tshor)
        
        lf_tsaperiodic = self.harmonize_columns(self.lf_tsaperiodic,
                                                **self.colnames_tshor)

        lf_ts_ver = pl.concat([
            lf_tslab,
            lf_tsresp,
            lf_tsnurse,
            lf_tsinout,
            ])

        #not collecting here causes errors in the following methods.
        lf_ts_hor = (pl.concat([lf_tsperiodic, lf_tsaperiodic],
                              how='diagonal',
                              rechunk=True)
                     .collect(streaming=True))

        lf_ts_ver = self.filter_tables(lf_ts_ver,
                                    kept_variables=self.kept_ts)

        self.timeseries_to_long(lf_ts_ver, lf_ts_hor.lazy())
        self.medication_to_long(lf_med)


        med_formatted = self.pl_format_meds(lf_med)

        self.stays = self._get_stays()
        #decrease n_patient_chunk for lower memory usage.
        self.stay_chunks = self.get_stay_chunks(n_patient_chunk=10_000)

        for i, stay_chunk in enumerate(self.stay_chunks):

            self.ts_hor_chunk = (self.filter_tables(lf_ts_hor,
                                        kept_stays=stay_chunk,
                                        ).collect())
            
            self.ts_ver_chunk = (self.filter_tables(lf_ts_ver,
                                        kept_stays=stay_chunk,
                                        kept_variables=self.kept_ts
                                        ).collect())
            
            self.med_chunk = (self.filter_tables(med_formatted, 
                                     kept_stays=stay_chunk,
                                     kept_variables=self.kept_med
                                     ))

            ts_formatted_chunk = self.pl_format_timeseries(lf_tsver=self.ts_ver_chunk.lazy(),
                                                           lf_tshor=self.ts_hor_chunk.lazy(),
                                                           chunk_number=i)
            
            self.newprocess_tables(ts_formatted_chunk,
                                med=self.med_chunk.collect().to_pandas(),
                                chunk_number=i)
