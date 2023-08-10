import pandas as pd

from utils.parquet_utils import compute_offset
from database_processing.medicationprocessor import MedicationProcessor
from database_processing.datapreparator import DataPreparator


class hiridPreparator(DataPreparator):
    def __init__(
            self,
            variable_ref_path,
            raw_ts_path,
            raw_pharma_path,
            admissions_path,
            imputedstage_path):

        super().__init__(dataset='hirid', col_stayid='admissionid')
        self.col_los = 'lengthofstay'
        self.unit_los = 'second'
        self.variable_ref_path = self.source_pth+variable_ref_path
        self.raw_ts_path = self.source_pth+raw_ts_path
        self.raw_pharma_path = self.source_pth+raw_pharma_path
        self.admissions_path = self.source_pth+admissions_path
        self.imputedstage_path = self.source_pth+imputedstage_path

        self.ts_savepth = self.parquet_pth+'/timeseries_1000_patient_chunks/'
        self.pharma_savepth = self.parquet_pth+'/pharma_1000_patient_chunks/'
        self.id_mapping = self._variablenames_mapping()

        self.weights = None
        self.heights = None

        self.admissions = self._load_admissions()

    def _load_ts_chunks(self):
        return pd.read_csv(self.raw_ts_path,
                           usecols=['observation_tables/',
                                    'patientid',
                                    'value',
                                    'variableid'],
                           chunksize=self.chunksize)

    def _load_pharma_chunks(self):
        return pd.read_csv(self.raw_pharma_path,
                           usecols=['pharma_records/',
                                    'pharmaid',
                                    'givenat',
                                    'givendose'],
                           chunksize=self.chunksize)

    def _load_los(self):
        """
        As is usually done with this database, the length of stay is defined as
        the last timeseries medurement of a patient.
        """
        timeseries = pd.read_csv(self.imputedstage_path,
                                 usecols=['imputed_stage/', 'reldatetime'])

        timeseries = timeseries.dropna().astype({'imputed_stage/': int})

        timeseries['reldatetime'] = pd.to_numeric(timeseries['reldatetime'],
                                                  errors='coerce')
        los = (timeseries.dropna()
               .rename(columns={'imputed_stage/': 'admissionid',
                                'reldatetime': 'lengthofstay'})
               .groupby('admissionid')
               .lengthofstay
               .max())
        return los

    def _load_admissions(self):
        adm = pd.read_csv(self.admissions_path)
        adm = adm.rename(columns={'general_table.csv': 'admissionid'})

        adm['admissiontime'] = pd.to_datetime(adm['admissiontime'],
                                              errors='coerce')

        adm['admissionid'] = pd.to_numeric(adm['admissionid'],
                                           errors='coerce')

        adm = (adm.dropna(subset=['admissiontime', 'admissionid'])
                  .astype({'admissionid': int,
                           'age': int}))
        return adm

    def _variablenames_mapping(self):
        variable_ref = pd.read_csv(self.variable_ref_path,
                                   sep=';',
                                   encoding='unicode_escape',
                                   usecols=['Source Table',
                                            'ID',
                                            'Variable Name'])

        variable_ref = variable_ref.dropna()

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
        """
        Fetches the admission height and weight to fill the labels table.
        The filled heights and weights are those that were available at 
        self.flat_hr_from_adm hours after admission.
        """
        print('Fecting heights and weights from timesersies tables, this may '
              'take several minutes...')
        variables = {'weight': 10000400,
                     'height': 10000450}

        ts_chunks = self._load_ts_chunks()

        heights, weights = [], []
        for i, chunk in enumerate(ts_chunks):
            print(f'Read {(i+1)*self.chunksize} lines from observation table...')
            chunk = (chunk.rename(columns={'observation_tables/': 'valuedate',
                                           'patientid': 'admissionid'})
                     .merge(self.admissions[['admissiontime', 'admissionid']],
                            on='admissionid')
                     .pipe(compute_offset,
                           col_intime='admissiontime',
                           col_measuretime='valuedate'))

            h_idx = chunk.variableid == variables['height']
            w_idx = chunk.variableid == variables['weight']

            time_idx = chunk.valuedate < self.flat_hr_from_adm*60
            heights.append(chunk.loc[h_idx & time_idx,
                           ['admissionid', 'value']])
            weights.append(chunk.loc[w_idx & time_idx,
                           ['admissionid', 'value']])

        df_heights = (pd.concat(heights)
                        .rename(columns={'value': 'height'})
                        .groupby('admissionid')
                        .mean())
        df_weights = (pd.concat(weights)
                        .rename(columns={'value': 'weight'})
                        .groupby('admissionid')
                        .mean())
        self.weights = df_weights
        self.heights = df_heights

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

        chunks = chunk_loader()

        for k, table in enumerate(chunks):

            table = table.rename(columns=rename_dic)

            table[numcols] = table[numcols].apply(
                pd.to_numeric, errors='coerce')

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
        if (self.heights is None) or (self.weights is None):
            self._load_heights_weights()

        lengthsofstay = self._load_los()

        admissions = (self.admissions.merge(self.heights,
                                            left_on='admissionid',
                                            right_index=True,
                                            how='left')
                                     .merge(self.weights,
                                            left_on='admissionid',
                                            right_index=True,
                                            how='left')
                                     .merge(lengthsofstay,
                                            left_on='admissionid',
                                            right_index=True))
        admissions['care_site'] = 'Bern University Hospital'
        self.save(admissions, self.parquet_pth+'labels.parquet')

    def gen_timeseries(self):
        """
        The timeseries table is too large to be loaded in memory. 
        The icu stays are not ordered in the table. 
        To create a chunk of 1000 patient, we must go through the whole file. 
        Consequently, this processing step is longer than other databases.
        """
        self.reset_chunk_idx()
        self.get_labels()
        for start in range(0, len(self.stays), self.n_patient_chunk):

            chunk = self._build_patient_chunk(
                                    start,
                                    chunk_loader=self._load_ts_chunks,
                                    rename_dic={
                                        'observation_tables/': 'offset'},
                                    itemid_label='variableid',
                                    value_label='value',
                                    id_mapping=self.id_mapping['observation'])

            self.chunk = self.prepare_tstable(chunk,
                                              col_offset='offset',
                                              col_intime='admissiontime')

            self.save_chunk(self.chunk, self.ts_savepth)

    def gen_medication(self):
        """
        Similary to the timeseries table, the patients are not ordered in the 
        raw files. the _build_patient_chunk goes through the whole table to
        extract data from a chunk of patients.
        """
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
                                    rename_dic={'pharma_records/': 'patientid',
                                                'givenat': 'offset'},
                                    itemid_label='pharmaid',
                                    value_label='givendose',
                                    id_mapping=self.id_mapping['pharma'])

            chunk = (chunk.drop(columns=['givendose', 'pharmaid'])
                          .pipe(self.mp.run))

            self.save_chunk(chunk, self.pharma_savepth)