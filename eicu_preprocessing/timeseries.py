import pandas as pd

from database_processing.timeseriespreprocessing import TimeseriesPreprocessing


class eicuTSP(TimeseriesPreprocessing):
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

        self.medication = self.load(self.med_savepath)

        self.flat = self.load(self.flat_savepath)
        self.tslab_files = self.ls(self.parquet_pth+lab_pth)
        self.tsresp_files = self.ls(self.parquet_pth+resp_pth)
        self.tsnurse_files = self.ls(self.parquet_pth+nurse_pth)
        self.tsaperiodic_files = self.ls(self.parquet_pth+aperiodic_pth)
        self.tsperiodic_files = self.ls(self.parquet_pth+periodic_pth)
        self.tsinout_files = self.ls(self.parquet_pth+inout_pth)

        self.colnames_lab = {
            'col_id': 'patientunitstayid',
            'col_var': 'labname',
            'col_value': 'labresult',
            'col_time': 'labresultoffset'
        }

        self.colnames_resp = {
            'col_id': 'patientunitstayid',
            'col_var': 'respchartvaluelabel',
            'col_value': 'respchartvalue',
            'col_time': 'respchartoffset'
        }

        self.colnames_inout = {
            'col_id': 'patientunitstayid',
            'col_var': 'celllabel',
            'col_value': 'cellvaluenumeric',
            'col_time': 'intakeoutputoffset'
        }

        self.colnames_nurse = {
            'col_id': 'patientunitstayid',
            'col_var': 'nursingchartcelltypevallabel',
            'col_value': 'nursingchartvalue',
            'col_time': 'nursingchartoffset'
        }

        self.colnames_med = {
            'col_id': 'patientunitstayid',
            'col_var': 'label',
            'col_value': 'value',
            'col_time': 'drugoffset'
        }

        self.colnames_tsper = {
            'col_id': 'patientunitstayid',
            'col_time': 'observationoffset'
        }

    def _get_admission_hours(self):
        self.flat['patient'] = (self.flat['patientunitstayid']
                                    .apply(lambda x: f'{self.dataset}-{x}'))
        return self.flat.loc[:, ['patient', 'hour']].set_index('patient')

    def run(self):

        self.reset_dir()

        self.medication = self.filter_tables(self.medication,
                                             kept_variables=self.kept_med,
                                             **self.colnames_med)

        admission_hours = self._get_admission_hours()

        for chunk_number, pths in enumerate(zip(self.tslab_files,
                                                self.tsresp_files,
                                                self.tsnurse_files,
                                                self.tsperiodic_files,
                                                self.tsaperiodic_files,
                                                self.tsinout_files)):

            tslab, tsresp, tsnurse, tsperiodic, tsaperiodic, tsinout = map(self.load, pths)

            tslab = self.filter_tables(tslab,
                                       kept_variables=self.kept_ts,
                                       **self.colnames_lab)

            tsresp = self.filter_tables(tsresp,
                                        kept_variables=self.kept_ts,
                                        **self.colnames_resp)

            tsnurse = self.filter_tables(tsnurse,
                                         kept_variables=self.kept_ts,
                                         **self.colnames_nurse)

            tsinout = self.filter_tables(tsinout,
                                         kept_variables=self.kept_ts,
                                         **self.colnames_inout)

            tsperiodic = self.filter_tables(tsperiodic,
                                            **self.colnames_tsper)

            tsaperiodic = self.filter_tables(tsaperiodic,
                                             **self.colnames_tsper)

            ts_ver = pd.concat([tslab, tsresp, tsnurse, tsinout])

            idx_chunk = self.medication.patient.isin(ts_ver.patient.unique())

            medication_chunk = self.medication.loc[idx_chunk]

            ts_hor = tsperiodic.merge(tsaperiodic,
                                      how='outer',
                                      on=['patient', 'time'])

            self.process_tables(ts_ver,
                                ts_hor,
                                med=medication_chunk,
                                admission_hours=admission_hours,
                                chunk_number=chunk_number)
