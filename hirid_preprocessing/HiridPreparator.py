from pathlib import Path

import pandas as pd
import polars as pl

from database_processing.datapreparator import DataPreparator
from database_processing.newmedicationprocessor import NewMedicationProcessor

class hiridPreparator(DataPreparator):
    def __init__(
            self,
            variable_ref_path,
            ts_path,
            pharma_path,
            admissions_path,
            imputedstage_path):

        super().__init__(dataset='hirid', col_stayid='admissionid')
        self.col_los = 'lengthofstay'
        self.unit_los = 'second'
        self.variable_ref_path = self.source_pth + variable_ref_path
        self.admissions_path = self.source_pth + admissions_path
        self.ts_path = self.source_pth + ts_path
        self.med_path = self.source_pth + pharma_path
        self.imputedstage_path = self.source_pth + imputedstage_path
        
        self.variable_ref_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(self.variable_ref_path)
        self.admissions_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(self.admissions_path)
        self.ts_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(Path(self.ts_path).parent)
        self.med_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(Path(self.med_path).parent)
        self.imputedstage_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(Path(self.imputedstage_path).parent)
        
        self._check_files_untarred()
        
        self.ts_savepth = self.savepath + 'timeseries.parquet'
        
        self.weights = None
        self.heights = None
        

    def raw_tables_to_parquet(self):
        """
        Writes initial csv.gz files to parquet files. This operations 
        needs only to be done once and allows further methods to be 
        done laziy using polars.
        """
        pths_as_parquet = {
                self.variable_ref_path: (False, ';'),
                self.admissions_path: (False, ','),
                self.imputedstage_path: (True, None),
                self.ts_path: (True, None),
                self.med_path: (True, None),
                }
        
        for i, (src_pth, (src_is_multiple_parquet, sep)) in enumerate(pths_as_parquet.items()):
            
            if src_is_multiple_parquet:
                tgt = self.raw_as_parquet_pth + self._get_name_as_parquet(Path(src_pth).parent)
            else:
                tgt = self.raw_as_parquet_pth + self._get_name_as_parquet(src_pth)
            
            if Path(tgt).is_file() and i==0:
                inp = input('Some parquet files already exist, skip conversion to parquet ?[n], y')
                if inp.lower() == 'y':
                    break

            self.write_as_parquet(src_pth,
                                  tgt,
                                  astype_dic={},
                                  encoding='unicode_escape',
                                  sep=sep,
                                  src_is_multiple_parquet=src_is_multiple_parquet)

    def init_gen(self):
        self.id_mapping = self._variablenames_mapping()
        self.lazyadmissions, self.admissions = self._load_admissions()
        
    def _check_files_untarred(self):
        '''Checks that files were properly untarred at step 0.'''
        notfound = False
        files = [self.ts_path,
                 self.med_path,
                 self.imputedstage_path,
                 self.admissions_path]
        for file in files:
            if not Path(file).exists():
                notfound = True
                print(f'\n/!\ {file} was not found, consider running step 0 to'
                      ' untar Hirid source files\n')
        if notfound:
            raise ValueError('Some files are missing, see warnings above.')
            
            
    def _load_pharma_chunks(self):
        df_all_in_one_chunk = pd.read_parquet(self.med_parquet_path,
                                              columns=['patientid',
                                                       'pharmaid',
                                                       'givenat',
                                                       'givendose'])
        return (df_all_in_one_chunk,)
    
    def _load_los(self):
        """
        As is usually done with this database, the length of stay is defined as
        the last timeseries measurement of a patient.
        """
        timeseries = pl.scan_parquet(self.imputedstage_parquet_path)

        los = (timeseries
               .select('patientid', 'reldatetime')
               .drop_nulls()
               .rename({'patientid': 'admissionid',
                        'reldatetime': 'lengthofstay'})
               .group_by('admissionid')
               .max()
               .with_columns(
                   pl.duration(seconds=pl.col('lengthofstay')).alias('lengthofstay')
                   ))
                
        return los
        
    
    def _load_admissions(self):
        try:
            adm = pl.scan_parquet(self.admissions_parquet_path)
        except FileNotFoundError:
            print(self.admissions_parquet_path,
                  'was not found.\n run raw_tables_to_parquet first.' )
            return None, None

        adm = (adm
               .rename({'patientid': 'admissionid'})
               .with_columns(
                   admissiontime=pl.col('admissiontime').str.to_datetime(),
                   admissionid=pl.col('admissionid').cast(pl.Int32)
                   )
               )
        return adm, adm.collect().to_pandas()

    def _variablenames_mapping(self):
        try:        
            lf = pl.scan_parquet(self.variable_ref_parquet_path)
        except FileNotFoundError:
            print(self.variable_ref_parquet_path,
                  'was not found.\n run raw_tables_to_parquet first.' )
            return None
        variable_ref = (lf
                        .select('Source Table',
                                'ID',
                                'Variable Name')
                        .collect()
                        .to_pandas()
                        .dropna()
                        )

        idx_obs = variable_ref['Source Table'] == 'Observation'
        idx_pharma = variable_ref['Source Table'] == 'Pharma'

        obs_id_mapping = variable_ref.loc[idx_obs]
        pharma_id_mapping = variable_ref.loc[idx_pharma]

        obs_id_mapping = dict(zip(obs_id_mapping['ID'],
                                  obs_id_mapping['Variable Name']))

        pharma_id_mapping = dict(zip(pharma_id_mapping['ID'],
                                     pharma_id_mapping['Variable Name']))

        obs_id_mapping[310] = 'Respiratory rate id=310'
        obs_id_mapping[5685] = 'Respiratory rate id=5685'

        return {'observation': obs_id_mapping,
                'pharma': pharma_id_mapping}

    def _load_heights_weights(self):
        print('Fetching heights and weights in timeseries data, this step '
              'takes several minutes.')
        variables = {'weight': 10000400,
                     'height': 10000450}
        
        ts = pl.scan_parquet(self.ts_parquet_path)
        
        df = (ts
             .select('datetime',
                     'patientid',
                     'value',
                     'variableid')
             .rename({'datetime': 'valuedate',
                      'patientid': 'admissionid'})
             .filter((pl.col('variableid')==variables['height']) 
                     | (pl.col('variableid')==variables['weight']))
             .join(self.lazyadmissions.select('admissiontime', 'admissionid'),
                   on='admissionid')
             .with_columns(
                 valuedate = pl.col('valuedate') - pl.col('admissiontime')
                 )
             .filter(pl.col('valuedate')<pl.duration(seconds=self.flat_hr_from_adm.total_seconds()))
             .drop('valuedate', 'admissiontime')
             .group_by('admissionid', 'variableid')
             .mean()
             .collect(streaming=True))
        
        partitions = df.partition_by(['variableid'], as_dict=True)
        
        weights = (partitions[variables['weight'],]
                           .rename({'value': 'weight'})
                           .drop('variableid')
                           .lazy())
        
        heights = (partitions[variables['height'],]
                           .rename({'value': 'height'})
                           .drop('variableid')
                           .lazy())
        
        print('  -> Done')
        return heights, weights

    def gen_labels(self):
        """
        The admission table does not contain the heights and weights. 
        These variables must be fetched from the timeseries table.
        The length of stay (los) is not specified either.
        It is usually derived from the last measurement of a timeseries
        variable. 
        """
        print('o Labels')
        
        lengthsofstay = self._load_los()

        if (self.heights is None) or (self.weights is None):
            self.heights, self.weights = self._load_heights_weights()

        admissions = (self.lazyadmissions
                      .join(self.heights, on='admissionid', how='left')
                      .join(self.weights, on='admissionid', how='left')
                      .join(lengthsofstay, on='admissionid', how='left')
                      .with_columns(
                          care_site=pl.lit('Bern University Hospital')
                          ))

        self.save(admissions, self.savepath+'labels.parquet')

    def gen_timeseries(self):
        self.get_labels(lazy=True)
        ts = pl.scan_parquet(self.ts_parquet_path)
        
        lf = (ts
              .select(['datetime', 'patientid', 'value', 'variableid'])
              .with_columns(pl.col('patientid').alias(self.col_stayid))
              .pipe(self.pl_prepare_tstable, 
                    itemid_label='variableid',
                    col_intime='admissiontime',
                    col_measuretime='datetime',
                    id_mapping=self.id_mapping['observation'],
                    col_value='value',
                    )
              )
        
        self.save(lf, self.ts_savepth)
        return lf
    
    
    def gen_medication(self):
        self.get_labels(lazy=True)
        
        labels = self.labels.select('admissionid',
                                    'lengthofstay',
                                    'admissiontime')
        pharma = (pl.scan_parquet(self.med_parquet_path)
                  .select('patientid',
                          'pharmaid',
                          'givenat',
                          'givendose',
                          'doseunit',
                          'route')
                  .with_columns(
                      pl.col('pharmaid').replace(self.id_mapping['pharma']).alias('pharmaitem')
                      )
                  .drop('pharmaid')
                  .rename({'patientid': 'admissionid'}))
        
        self.nmp = NewMedicationProcessor('hirid',
                                          lf_med=pharma,
                                          lf_labels=labels,
                                          col_pid='admissionid',
                                          col_med='pharmaitem',
                                          col_start='givenat',
                                          col_end=None,
                                          col_los='lengthofstay',
                                          col_dose='givendose',
                                          col_dose_unit='doseunit',
                                          col_route='route',
                                          offset_calc=True,
                                          col_admittime='admissiontime'
                                        )

        self.med = self.nmp.run()
        self.save(self.med, self.med_savepath)
