from pathlib import Path

import pandas as pd
import polars as pl

from database_processing.medicationprocessor import MedicationProcessor
from database_processing.datapreparator import DataPreparator


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
        self.pharma_savepth = self.savepath+'/pharma_1000_patient_chunks/'
        
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
                #self.imputedstage_path: (True, None),
                #self.ts_path: (True, None),
                #self.med_path: (True, None),
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
             .max())
                
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
                   admissionid=pl.col('admissionid').cast(pl.Int32())
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

    def _build_patient_chunk(self,
                             start,
                             chunk_loader=None,
                             rename_dic={},
                             valuedate_label='valuedate',
                             itemid_label='itemid',
                             value_label='value',
                             id_mapping=None):
        '''
        In the timeseries files, the admissionids are not ordered.
        To create patient chunks, we have to go through the whole file and
        select a list of patient ids.
        '''
        numcols = [itemid_label, 'patientid']
        if self.labels is None:
            raise ValueError('Please run labels first.')

        chunk_tables = []
        patient_chunk = self.stays[start:start+self.n_patient_chunk]

        for k, table in enumerate(chunk_loader()):
            table = table.rename(columns=rename_dic)

            table[numcols] = table[numcols].apply(pd.to_numeric, errors='coerce')

            table = (table.dropna(subset=numcols)
                     .astype({itemid_label: int,
                              'patientid': int}))

            table = table.loc[table.patientid.isin(patient_chunk)]

            chunk_tables.append(table)

        df_chunk = pd.concat(chunk_tables)
        df_chunk['admissionid'] = df_chunk['patientid']
        df_chunk['variable'] = df_chunk[itemid_label].map(id_mapping)

        df_chunk[value_label] = pd.to_numeric(df_chunk[value_label],
                                              errors='coerce')
        return df_chunk

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
        """
        Because patients are not ordered in the 
        raw files. the _build_patient_chunk goes through the whole table to
        extract data from a chunk of patients.
        TODO : to polars !
        """
        
        #col_dosage=givendose
        #col_unit=doseunit
        #col_route=route
        self.reset_chunk_idx()
        self.get_labels()
        self.mp = MedicationProcessor('hirid',
                                      self.labels,
                                      col_pid='admissionid',
                                      col_med='variable',
                                      col_time='offset',
                                      col_los='lengthofstay',
                                      unit_los='second',
                                      offset_calc=True,
                                      col_admittime='admissiontime')

        for start in range(0, len(self.stays), self.n_patient_chunk):

            chunk = self._build_patient_chunk(
                                    start,
                                    chunk_loader=self._load_pharma_chunks,
                                    rename_dic={'givenat': 'offset'},
                                    itemid_label='pharmaid',
                                    value_label='givendose',
                                    id_mapping=self.id_mapping['pharma'])

            chunk = (chunk.drop(columns=['givendose', 'pharmaid'])
                          .pipe(self.mp.run))

            self.save_chunk(chunk, self.pharma_savepth)
