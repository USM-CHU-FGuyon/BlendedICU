import pandas as pd

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

        self.medication = self.load(self.savepath+med_pth)

        self.timeseries = self.load(self.savepath+ts_pth)
        self.timeseries_lab = self.load(self.savepath+tslab_pth)
        self.outputevents = self.load(self.savepath+outputevents_pth,
                                      columns=['ICUSTAY_ID',
                                               'offset',
                                               'VALUE',
                                               'LABEL'])

        self.colnames_med = {
            'col_id': 'ICUSTAY_ID',
            'col_var': 'label',
            'col_value': 'value',
            'col_time': 'offset'
        }

        self.colnames_ts = {
            'col_id': 'ICUSTAY_ID',
            'col_var': 'LABEL',
            'col_value': 'VALUENUM',
            'col_time': 'CHARTTIME'
        }

        self.colnames_lab = {
            'col_id': 'ICUSTAY_ID',
            'col_var': 'LABEL',
            'col_value': 'VALUENUM',
            'col_time': 'offset'
        }

        self.colnames_outputevents = {
            'col_id': 'ICUSTAY_ID',
            'col_var': 'LABEL',
            'col_value': 'VALUE',
            'col_time': 'offset'
        }

    def run(self):

        self.outputevents = self.filter_tables(self.outputevents,
                                               kept_variables=self.kept_ts,
                                               **self.colnames_outputevents)

        self.timeseries_lab = self.filter_tables(self.timeseries_lab,
                                                 kept_variables=self.kept_ts,
                                                 **self.colnames_lab)

        self.timeseries = self.filter_tables(self.timeseries,
                                             kept_variables=self.kept_ts,
                                             **self.colnames_ts)

        self.medic = self.filter_tables(self.medication,
                                      kept_variables=self.kept_med,
                                      **self.colnames_med)

        self.timeser = pd.concat([self.timeseries,
                                  self.timeseries_lab,
                                  self.outputevents],
                                 axis=0)

        patientids = self.timeser.patient.drop_duplicates()
        self.chunks = self.generate_patient_chunks(patientids)

        for chunk_number, patient_chunk in enumerate(self.chunks):
            ts_chunk = self.timeser.loc[self.timeser.patient.isin(patient_chunk)]
            med_chunk = self.medic.loc[self.medic.patient.isin(patient_chunk)]
            self.ts_chunk = ts_chunk
            self.med_chunk=med_chunk

            self.process_tables(ts_chunk,
                                med=med_chunk,
                                chunk_number=chunk_number)
