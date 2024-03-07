import pandas as pd

from database_processing.medicationprocessor import MedicationProcessor
from database_processing.datapreparator import DataPreparator


class AmsterdamPreparator(DataPreparator):
    def __init__(self,
                 admission_pth,
                 drugitems_pth,
                 numericitems_pth,
                 listitems_pth):
        super().__init__(dataset='amsterdam', col_stayid='admissionid')
        self.admission_pth = self.source_pth+admission_pth
        self.drugitems_pth = self.source_pth+drugitems_pth
        self.numericitems_pth = self.source_pth+numericitems_pth
        self.listitems_pth = self.source_pth+listitems_pth

        self.ts_savepath = f'{self.savepath}/numericitems_{self.n_patient_chunk}_patient_chunks/'
        self.listitems_savepath = f'{self.savepath}/listitems.parquet'
        self.gcs_savepath = f'{self.savepath}/glasgow_coma_scores.parquet'
        self.col_los = 'lengthofstay'
        self.unit_los = 'hour'

    def gen_labels(self):
        print('Labels...')
        admissions = pd.read_csv(self.admission_pth,
                                 compression='gzip',
                                 encoding='ISO-8859-1')
        admissions['care_site'] = 'Amsterdam University medical center'
        return self.save(admissions, self.labels_savepath)

    def gen_medication(self):
        print('Medications...')
        self.reset_chunk_idx()
        self.get_labels()
        drugitems = pd.read_csv(self.drugitems_pth,
                                compression='gzip',
                                encoding='ISO-8859-1',
                                usecols=['admissionid', 'item', 'start'])

        self.mp = MedicationProcessor('amsterdam',
                                      self.labels,
                                      col_pid='admissionid',
                                      col_med='item',
                                      col_time='start',
                                      col_los='lengthofstay',
                                      unit_offset='milisecond',
                                      unit_los='hour')

        self.med = self.mp.run(drugitems)
        return self.save(self.med, self.med_savepath)

    def gen_listitems_timeseries(self):
        """
        The listitems table should fit in memory in most machines.
        It can be read-processed-saved all at once.
        A separate file is saved for the computed Glasgow coma scores.
        """
        print('Listitems ts...')
        self.get_labels()
        listitems = pd.read_csv(self.listitems_pth,
                                compression='gzip',
                                encoding='ISO-8859-1',
                                usecols=['admissionid',
                                         'item',
                                         'itemid',
                                         'value',
                                         'valueid',
                                         'measuredat'])

        listitems = self.prepare_tstable(listitems,
                                         col_offset='measuredat',
                                         col_variable='item',
                                         unit_offset='milisecond')

        df_gcs = self._compute_gcs(listitems)
        listitems = listitems.drop(columns=['valueid', 'itemid'])
        self.save(df_gcs, self.gcs_savepath)
        self.save(listitems, self.listitems_savepath)

    def _compute_gcs(self, df):
        """
        The Glasgow Coma Scores are not provided in the raw dataset. 
        They can be computed. See the sql file on he Amsterdam GitHub:
        https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/amsterdamumcdb/sql/common/gcs.sql
        """
        df = df.set_index(["admissionid", "measuredat"])[['valueid', 'itemid']]

        df_eye = df.loc[df.itemid.isin([6732, 13077, 14470, 16628, 19635, 19638]), :].copy()
        df_motor = df.loc[df.itemid.isin([6734, 13072, 14476, 16634, 19636, 19639]), :].copy()
        df_verbal = df.loc[df.itemid.isin([6735, 13066, 14482, 16640, 19637, 19640]), :].copy()

        df_eye.loc[df_eye.itemid.isin([6732]), 'valueid'] = 5 - df_eye.loc[df_eye.itemid.isin([6732]), 'valueid']
        df_eye.loc[df_eye.itemid.isin([14470, 16628, 19635]), 'valueid'] -= 4
        df_eye.loc[df_eye.itemid.isin([19638]), 'valueid'] -= 8
        df_eye = df_eye.drop(columns="itemid")
        df_eye = df_eye.rename(columns={"valueid":"eyes_score"})

        df_motor.loc[df_motor.itemid.isin([6734]), 'valueid'] = 5 - df_motor.loc[df_motor.itemid.isin([6734]), 'valueid']
        df_motor.loc[df_motor.itemid.isin([14476, 16634, 19636]), 'valueid'] -= 6
        df_motor.loc[df_motor.itemid.isin([19639]), 'valueid'] -= 12
        df_motor = df_motor.drop(columns="itemid")
        df_motor = df_motor.rename(columns={"valueid":"motor_score"})
           
        df_verbal.loc[df_verbal.itemid.isin([6735]), 'valueid'] = 6 - df_verbal.loc[df_verbal.itemid.isin([6735]), 'valueid']
        df_verbal.loc[df_verbal.itemid.isin([14482, 16640, 19637]), 'valueid'] -= 5
        df_verbal.loc[df_verbal.itemid.isin([19640]), 'valueid'] -= 15
        df_verbal.loc[df_verbal.valueid < 1] = 1
        df_verbal = df_verbal.drop(columns="itemid")
        df_verbal = df_verbal.rename(columns={"valueid":"verbal_score"})

        df_gcs = (df_eye.join(df_motor)
                        .join(df_verbal))
        df_gcs["gcs_score"] = df_gcs.sum(1, min_count=3)
        return df_gcs.reset_index()


    def gen_num_timeseries(self):
        """
        The numericitems table does not fit in memory on most machines. 
        The processing is done by chunks:
             The numericitem csv file is read by chunks of 10_000_000 rows by
             default.
             when the patient chunksize is reached (by default n_patient_chunk
                                                    is 1000)
             a chunk of patient is saved to a parquet file.
        It is possible to process in this way because the patients in the 
        numericitems table are ordered.
        """
        print('numericitems ts...')
        self.get_labels()

        df_chunks = pd.read_csv(self.numericitems_pth,
                                compression='gzip',
                                encoding='ISO-8859-1',
                                chunksize=self.chunksize,
                                usecols=['admissionid',
                                         'item',
                                         'value',
                                         'measuredat'])

        df = pd.DataFrame()
        for chunk in df_chunks:
            chunk = self.prepare_tstable(chunk,
                                         col_offset='measuredat',
                                         col_variable='item',
                                         unit_offset='milisecond')

            df = pd.concat([df, chunk])
            patients = df.admissionid.unique()
            while len(patients) > self.n_patient_chunk:
                patient_chunk = patients[:self.n_patient_chunk]
                save_idx = df.admissionid.isin(patient_chunk)

                self.save_chunk(df.loc[save_idx], self.ts_savepath)
                df = df.loc[~save_idx]
                patients = df.admissionid.unique()
