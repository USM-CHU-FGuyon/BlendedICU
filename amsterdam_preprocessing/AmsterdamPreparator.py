from pathlib import Path

import polars as pl

from database_processing.medicationprocessor import MedicationProcessor
from database_processing.datapreparator import DataPreparator


class AmsterdamPreparator(DataPreparator):
    def __init__(self,
                 admission_pth,
                 drugitems_pth,
                 numericitems_pth,
                 listitems_pth):
        super().__init__(dataset='amsterdam', col_stayid='admissionid')
        self.admission_pth = self.source_pth + admission_pth
        self.drugitems_pth = self.source_pth + drugitems_pth
        self.numericitems_pth = self.source_pth + numericitems_pth
        self.listitems_pth = self.source_pth + listitems_pth

        self.admission_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(admission_pth)
        self.drugitems_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(drugitems_pth)
        self.numericitems_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(numericitems_pth)
        self.listitems_parquet_pth = self.raw_as_parquet_pth + self._get_name_as_parquet(listitems_pth)

        self.ts_savepath = f'{self.savepath}/numericitems.parquet'
        self.listitems_savepath = f'{self.savepath}/listitems.parquet'
        self.gcs_savepath = f'{self.savepath}/glasgow_coma_scores.parquet'
        self.col_los = 'lengthofstay'
        self.unit_los = 'hour'

    def raw_tables_to_parquet(self):
        """
        Writes initial csv.gz files to parquet files. This operations 
        needs only to be done once and allows further methods to be 
        done laziy using polars.
        """
        for i, src_pth in enumerate([
                self.admission_pth,
                self.drugitems_pth,
                self.listitems_pth,
                self.numericitems_pth,
                ]):

            tgt = self.raw_as_parquet_pth + self._get_name_as_parquet(src_pth)
            print(src_pth, tgt)
            if Path(tgt).is_file() and i==0:
                inp = input('Some parquet files already exist, skip conversion to parquet ?[n], y')
                if inp.lower() == 'y':
                    break
            
            self.write_as_parquet(src_pth,
                                  tgt,
                                  astype_dic={},
                                  encoding='ISO-8859-1')
            
    def gen_labels(self):
        print('Labels...')
        admissions = (pl.scan_parquet(self.admission_parquet_pth)
                      .with_columns(care_site=pl.lit('Amsterdam University medical center'))
                      .collect())
        
        return self.save(admissions, self.labels_savepath)

    def gen_medication(self):
        print('Medications...')
        self.reset_chunk_idx()
        self.get_labels()
        
        drugitems = (pl.scan_parquet(self.drugitems_parquet_pth)
                     .select('admissionid', 'item', 'start')
                     .collect()
                     .to_pandas())

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
        print('Listitems ts...')
        self.get_labels(lazy=True)

        df_listitems = (pl.scan_parquet(self.listitems_parquet_pth)
                        .select('admissionid',
                                'item',
                                'itemid',
                                'value',
                                'valueid',
                                'measuredat')
                        .pipe(self.pl_prepare_tstable,
                              col_offset='measuredat',
                              col_variable='item',
                              col_value='value',
                              unit_offset='millisecond',
                              unit_los='hour')
                        .collect())
        
        df_gcs = self._compute_gcs(df_listitems.to_pandas())
        
        df_listitems = df_listitems.drop('valueid', 'itemid')
        
        self.save(df_gcs, self.gcs_savepath)
        self.save(df_listitems, self.listitems_savepath)

    def _compute_gcs(self, df):
        """
        The Glasgow Coma Scores are not provided in the raw dataset. 
        They can be computed. See the sql file on he Amsterdam GitHub:
        https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/amsterdamumcdb/sql/common/gcs.sql
        """
        df = df.set_index(["admissionid", self.col_offset])[['valueid', 'itemid']]

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
        print('numericitems...')
        self.get_labels(lazy=True)

        df = (pl.scan_parquet(self.numericitems_parquet_pth)
              .select('admissionid', 'item', 'value', 'measuredat')
              .pipe(self.pl_prepare_tstable,
                    col_offset='measuredat',
                    col_variable='item',
                    col_value='value',
                    unit_offset='millisecond',
                    unit_los='hour')
              .collect(streaming=True))

        self.save(df, self.ts_savepath)
