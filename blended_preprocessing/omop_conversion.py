from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pyarrow as pa

from blended_preprocessing.timeseries import blendedicuTSP
from omop_cdm import cdm


class OMOP_converter(blendedicuTSP):
    def __init__(self,
                 initialize_tables=False,
                 parquet_format=True,
                 full_init=True):
        super().__init__()
        self.toparquet = parquet_format
        self.data_pth = self.savepath
        self.ref_date = datetime(year=2023, month=1, day=1)
        self.end_date = datetime(year=2099, month=12, day=31)
        self.adm_measuredat = self.flat_hr_from_adm.total_seconds()
        self.admission_data_datetime = (self.ref_date + pd.Timedelta(self.adm_measuredat, unit='second'))
        self.n_chunks = 100
        if full_init:
            self.labels = self._load_labels()
            self.ts_pths_chunks, self.med_pths_chunks = self._get_pth_chunks()

        pth_concept_table = fr'{self.aux_pth}OMOP_vocabulary/CONCEPT.parquet'
        self.omop_concept = pd.read_parquet(pth_concept_table,
                                            columns=['concept_name',
                                                     'concept_code',
                                                     'domain_id',
                                                     'vocabulary_id',
                                                     'concept_class_id',
                                                     'standard_concept'])

        self.savedir = f'{self.data_pth}/OMOP-CDM/'
        print(self.savedir)
        self.start_index = {
            'person': 1000000,
            'visit_occurrence': 2000000,
            'observation': 3000000,
            'drug_exposure': 4000000,
            'measurement': 5000000000,
            'domain': 6000000,
            'care_site': 7000000,
            'location': 8000000,
        }
        
        self.visit_concept_ids = {
            'emergency': 9203,  # Visit
            'other': 8844,  # CMS Place of Service
            'operating_room': 4021813,  # SNOMED
            'direct_admit': 4139502,  # SNOMED
            'icu': 4148981,  # SNOMED
            'unknown': 0,
            'death': 0,
            'home': 4139502,  # SNOMED
            'hospital': 4318944,  # SNOMED
            'rehab': 38004285,  # NUCC
            'medical_icu': 40481392,  # SNOMED
            'cardiac_icu': 4149943,  # SNOMED
            'surgical_icu': 4305366,  # SNOMED
            'trauma_icu': 763903,  # SNOMED
            'neuro_icu': 4148496,  # SNOMED
            'medical_surgical_icu': 4160026,  # SNOMED
            }
        
        self.unit_mapping = {
            'age': 9448,
            'height': 8582,
            'weight': 9529,
            'heart_rate': 8541,
            'invasive_systolic_blood_pressure': 8876,
            'invasive_diastolic_blood_pressure': 8876,
            'invasive_mean_blood_pressure': 8876,
            'noninvasive_systolic_blood_pressure': 8876,
            'noninvasive_diastolic_blood_pressure': 8876,
            'noninvasive_mean_blood_pressure': 8876,
            'respiratory_rate_setting': 8541,
            'tidal_volume_setting': 9571,
            'plateau_pressure': 44777590,
            'O2_pulseoxymetry_saturation': 8554,
            'O2_arterial_saturation': 8554,
            'lactate': 8861,
            'blood_glucose': 8753,
            'magnesium': 8753,
            'sodium': 8753,
            'creatinine': 8749,
            'calcium': 8749,
            'temperature': 586323,
            'FiO2': 8554,
            'chloride': 8554,
            'phosphate': 8840,
            'bicarbonate': 8753,
            'pH': 8482,
            'paO2': 8482,
            'paCO2': 8482,
            'potassium': 8753,
            'PTT': 8555,
            'bilirubine': 8749,
            'urine_output': 44777613,
            'alanine_aminotransferase': 8645,
            'aspartate_aminotransferase': 8645,
            'alkaline_phosphatase': 8645,
            'respiratory_rate': 8541,
            'albumin': 8636,
            'blood_urea_nitrogen': 8840,
            'expiratory_tidal_volume': 9571,
            'white_blood_cells': 8510,
            'platelets': 8510,
            'hemoglobin': 8713,
            'PEEP': 44777590,
            'glasgow_coma_score': 0,
            'glasgow_coma_score_eye': 0,
            'glasgow_coma_score_motor': 0,
            'glasgow_coma_score_verbal': 0,
            'ventilator_mode': 0,
        }

        self.units_concept_ids = np.unique([*self.unit_mapping.values()])

        self.concept_unit = self.omop_concept.loc[self.units_concept_ids]

        self.concept_table()
        
        if initialize_tables:
            self.source_to_concept_map_table()

            self.location_table()
            self.care_site_table()
            self.person_table()
            self.visit_occurence_table()

            self.death_table()
            self.domain_table()
            self.observation_table()

        self.units = self._get_units()
        
        self.measurement_schema = self._measurement_schema()
        self.observation_schema = self._observation_schema()
        self.drug_exposure_schema = self._drug_exposure_schema()
        
    def _get_pth_chunks(self, shuffleseed=974):
        '''
        rglob lists all files, then sorts them and shuffle them with a seed 
        to make a reproducible unsorted order.
        then paths are split into a list of chunks.
        '''
        ts_pths = self.rglob(self.data_pth+'formatted_timeseries/',
                                         '*.parquet',
                                         verbose=True,
                                         sort=True,
                                         shuffleseed=shuffleseed)
        med_pths = self.rglob(self.data_pth+'formatted_medications/',
                                          '*.parquet',
                                          verbose=True,
                                          sort=True,
                                          shuffleseed=shuffleseed)
        
        ts_pths_chunks = self._get_chunks(ts_pths)
        med_pths_chunks = self._get_chunks(med_pths)
        return ts_pths_chunks, med_pths_chunks

    def _measurement_schema(self):
        schema = pa.schema([('value_as_number', pa.float32()),
                            ('time', pa.float32()),
                            ('visit_occurrence_id', pa.int64()),
                            ('visit_start_date', pa.date32()),
                            ('visit_source_value', pa.string()),
                            ('person_id', pa.int32()),
                            ('measurement_datetime', pa.date64()),
                            ('measurement_date', pa.date32()),
                            ('measurement_time', pa.time32('s')),
                            ('measurement_concept_id', pa.int32()),
                            ('measurement_source_value', pa.float32()),
                            ('unit_source', pa.string()),
                            ('unit_concept_id', pa.int32()),
                            ('measurement_id', pa.float32()),
                            ])
        return schema
    
    def _observation_schema(self):
        schema = pa.schema([('observation_id', pa.int32()),
                            ('person_id', pa.int32()),
                            ('observation_concept_id', pa.int32()),
                            ('observation_date', pa.date32()),
                            ('observation_datetime', pa.time32('s')),
                            ('observation_type_concept_id', pa.int32()),
                            ('value_as_number', pa.float64()),
                            ('value_as_string', pa.string()),
                            ('value_as_concept_id', pa.int32()),
                            ('qualifier_concept_id', pa.float32()),
                            ('unit_concept_id', pa.int32()),
                            ('provider_id', pa.float32()),
                            ('visit_occurrence_id', pa.int64()),
                            ('visit_detail_id', pa.float32()),
                            ('observation_source_value', pa.float32()),
                            ('observation_source_concept_id', pa.int32()),
                            ('unit_source_value', pa.string()),
                            ('qualifier_source_value', pa.string()),
                            ('value_source_value', pa.float64()),
                            ('observation_event_id', pa.float64()),
                            ('obs_event_field_concept_id', pa.float64()),
                            ])
        return schema
    
    def _drug_exposure_schema(self):
        schema = pa.schema([('drug_source_value', pa.string()),
                            ('drug_type_concept_id', pa.int32()),
                            ('visit_occurrence_id', pa.int64()),
                            ('person_id', pa.int32()),
                            ('drug_exposure_start_date', pa.date32()),
                            ('drug_exposure_start_datetime', pa.string()),
                            ('drug_exposure_end_date', pa.date32()),
                            ('drug_exposure_end_datetime', pa.string()),
                            ('drug_concept_id', pa.int32()),
                            ])
        return schema
        
        
        
    def _get_chunks(self, pths):
        return map(list, np.array_split(pths, self.n_chunks))
        
    def source_to_concept_map_table(self):
        ts_mapping = self.cols.concept_id.dropna().astype(int)
        visit_mapping = pd.Series(self.visit_concept_ids)
        
        self.concept_mapping = pd.concat([visit_mapping,
                                          ts_mapping,
                                          self.concept_ids_obs])
        print('Source_to_concept_map table...')
        self.source_to_concept_map = cdm.tables['SOURCE_TO_CONCEPT_MAP']
        self.source_to_concept_map['source_code'] = self.concept_mapping.index
        self.source_to_concept_map['source_concept_id'] = 0
        self.source_to_concept_map['target_concept_id'] = self.concept_mapping.values
        self.source_to_concept_map['valid_start_date'] = self.ref_date
        self.source_to_concept_map['valid_end_date'] = self.end_date

        self.source_to_concept_mapping = self.source_to_concept_map.set_index('source_code')['target_concept_id']
        
        
    def _get_units(self):
        """
        uses the concept_ids listed in the unit_mapping dictionary and returns
        the unit concepts.
        """
        unit_mapping = (pd.DataFrame.from_dict(self.unit_mapping,
                                               orient='index')
                        .rename(columns={0:'unit_concept_id'}))
        
        units = (unit_mapping.merge(self.concept_unit['concept_code'],
                                    left_on='unit_concept_id',
                                    right_index=True)
                             .drop(columns='unit_concept_id')
                             .replace('No matching concept', ''))
        return units

    def _load_med(self):
        #TODO : load med by chunk.
        self.med_pths = self.rglob(self.data_pth+'formatted_medications',
                                   '*.parquet')
        print(f'Loading med for {len(self.med_pths)} patients...')
        df = self.load(self.med_pths, verbose=False)
        print('   -> done')
        return df.reset_index()

    def person_table(self):
        print('Person table...')
        self.person = cdm.tables['PERSON']

        self.person['gender_concept_id'] = self.labels.sex.map({1: 8507,
                                                                0: 8532})
        self.person['year_of_birth'] = self.ref_date.year-self.labels.age
        self.person['birth_datetime'] = (self.person.year_of_birth
                                         .apply(lambda x: datetime(year=int(x),
                                                                   month=1,
                                                                   day=1)))
        self.person['person_source_value'] = self.labels.uniquepid
        self.person['gender_source_value'] = self.labels.sex
        self.person['location_id'] = (self.labels.source_dataset.map(
                                                        {'amsterdam': 'NL',
                                                         'hirid': 'CH',
                                                         'mimic': 'US',
                                                         'mimic3': 'US',
                                                         'eicu': 'US'})
                                        .map(self.locationid_mapper))
        unique_pid = self.person['person_source_value'].unique()
        start_idx = self.start_index['person']
        self.person_id_mapper = pd.Series(range(start_idx,
                                                start_idx+len(unique_pid)),
                                          index=unique_pid)
        self.person['person_id'] = self.person['person_source_value'].map(self.person_id_mapper)

    def visit_occurence_table(self):
        self.visit_occurrence = cdm.tables['VISIT_OCCURRENCE']

        self.visit_occurrence['_source_person_id'] = self.labels.uniquepid
        self.visit_occurrence['_source_visit_id'] = self.labels.patient
        self.visit_occurrence['visit_source_value'] = self.labels.patient
        self.visit_occurrence['visit_start_date'] = self.ref_date.date()
        self.visit_occurrence['visit_start_datetime'] = self.ref_date.time()
        self.visit_occurrence['visit_concept_id'] = 32037
        self.visit_occurrence['visit_type_concept_id'] = 44818518
        self.visit_occurrence['admitted_from_concept_id'] = (self.labels.origin
                                                             .map(self.admission_origins)
                                                             .map(self.concept_mapping))
        self.visit_occurrence['admitted_from_source_value'] = self.labels.origin
        visit_end_datetime = self.ref_date + self.labels.lengthofstay.apply(timedelta)
        self.visit_occurrence['visit_end_date'] = visit_end_datetime.dt.date
        self.visit_occurrence['visit_end_datetime'] = visit_end_datetime.dt.time
        self.visit_occurrence['person_id'] = self.visit_occurrence['_source_person_id'].map(self.person_id_mapper)
        self.visit_occurrence['discharged_to_source_value'] = self.labels.discharge_location
        self.visit_occurrence['discharged_to_concept_id'] = self.labels.discharge_location.map(self.concept_mapping)
        self.visit_occurrence['visit_occurrence_id'] = self.visit_occurrence.index + self.start_index['visit_occurrence']
        self.visit_occurrence = self.visit_occurrence.set_index(
            'visit_occurrence_id', drop=False)
        self.visit_mapper = self.visit_occurrence[['_source_visit_id', 'visit_occurrence_id']].set_index(
            '_source_visit_id')['visit_occurrence_id']

        self.visit_occurrence = self.visit_occurrence.drop(
            columns=['_source_person_id', '_source_visit_id'])

        self.labels.index = self.labels.patient.map(self.visit_mapper)
        self.labels = self.labels.rename_axis('visit_occurrence_id')

    def death_table(self):
        self.death_labels = self.labels.loc[self.labels.mortality == 1]

        self.death = cdm.tables['DEATH']
        self.death['person_id'] = self.death_labels.index
        self.death.index = self.death.person_id
        death_datetimes = self.ref_date + self.death_labels.lengthofstay.apply(timedelta)

        self.death['death_date'] = death_datetimes.dt.date
        self.death['death_datetime'] = death_datetimes.dt.time

    def _add_measurement(self, varname, timeseries=None, patients=[]):
        self.ts = timeseries

        print(f'collecting {varname}')
        if timeseries is None:
            keep_idx = self.labels.patient.isin(patients)
            vals = self.labels.loc[keep_idx, ['patient', varname]]
            vals['time'] = self.adm_measuredat
        else:
            try:
                vals = timeseries.loc[:, ['time', 'patient', varname]].dropna()
            except KeyError:
                print(f'Key {varname} not found')
                return self.measurement

        vals = vals.rename(columns={varname: 'value_as_number'})
        self.vals = vals
        unit = self.concept.loc[self.unit_mapping[varname]]
        if len(unit.shape) == 2:
            # There may be several duplicate matches, in which case we convert
            # the dataframe to series.
            unit = unit.iloc[0]

        vals = vals.merge(self.visit_occurrence[['visit_occurrence_id',
                                                 'visit_start_date',
                                                 'visit_source_value',
                                                 'person_id']],
                          left_on='patient',
                          right_on='visit_source_value')

        vals['measurement_datetime'] = (self.ref_date + \
            vals['time'].apply(pd.Timedelta, unit='second')).astype('datetime64[ns]')
        
        vals['measurement_date'] = vals['measurement_datetime'].dt.date
        vals['measurement_time'] = vals['measurement_datetime'].dt.time

        vals['visit_occurrence_id'] = vals['visit_occurrence_id']
        vals['measurement_concept_id'] = self.concept_mapping[varname]
        vals['measurement_source_value'] = vals['value_as_number']
        vals['unit_source'] = unit['concept_code']
        vals['unit_concept_id'] = unit.name
        vals = vals.drop(columns=['patient'])
        return self.concat([self.measurement, vals])


    def measurement_table(self, start_chunk=0):
        start_index = self.start_index['measurement']
        self.admission_measurements = ['height', 'weight']
        self.ts_measurements = [
            'heart_rate', 'invasive_systolic_blood_pressure',
            'invasive_diastolic_blood_pressure',
            'invasive_mean_blood_pressure',
            'noninvasive_systolic_blood_pressure',
            'noninvasive_diastolic_blood_pressure',
            'noninvasive_mean_blood_pressure',
            'O2_pulseoxymetry_saturation', 'O2_arterial_saturation',
            'lactate', 'blood_glucose', 'magnesium', 'sodium',
            'creatinine', 'calcium', 'temperature', 'FiO2', 'hemoglobin',
            'chloride', 'pH', 'paO2', 'paCO2', 'plateau_pressure',
            'respiratory_rate_setting', 'tidal_volume_setting', 'potassium',
            'PTT', 'bilirubine', 'alanine_aminotransferase',
            'aspartate_aminotransferase', 'respiratory_rate', 'albumin',
            'blood_urea_nitrogen', 'expiratory_tidal_volume',
            'white_blood_cells', 'platelets', 'phosphate', 'bicarbonate',
            'alkaline_phosphatase', 'PEEP', 'urine_output',
            'glasgow_coma_score', 'glasgow_coma_score_eye',
            'glasgow_coma_score_motor', 'glasgow_coma_score_verbal']

        for i, pth_chunk in enumerate(self.ts_pths_chunks):
            if i< start_chunk:
                continue
            print(f'Measurement chunk {i}/{self.n_chunks}')
            
            self.measurement = cdm.tables['MEASUREMENT'].copy()
            chunk = pd.read_parquet(pth_chunk).reset_index()
            self.chunk = chunk

            for varname in self.admission_measurements:
                self.measurement = self._add_measurement(varname,
                                                         patients=chunk.patient.unique())

            for varname in self.ts_measurements:
                self.measurement = self._add_measurement(varname,
                                                         timeseries=chunk)

            self.measurement['measurement_id'] = (self.measurement.index + start_index).astype('float32')
            start_index = start_index+self.start_index['measurement']

            self.measurement['visit_start_date'] = pd.to_datetime(self.measurement['visit_start_date'])
            self.measurement['measurement_date'] = pd.to_datetime(self.measurement['measurement_date'])

            self.export_table(self.measurement,
                              'MEASUREMENT',
                              mode='parquet',
                              chunkindex=i,
                              schema=self.measurement_schema)


    def _add_observation(self, breaks, column, concept_id, unit_concept_id):
        obs = pd.DataFrame(columns=self.observation.columns)
        obs_intervals = pd.IntervalIndex.from_breaks(breaks)
        obs_labels = {inter: (str(inter).replace('(', '')
                                        .replace(']', '')
                                        .replace(', ', '-')) 
                      for inter in obs_intervals}
        obs['value_source_value'] = self.labels[column]
        obs['unit_source_value'] = self.concept.loc[unit_concept_id, 'concept_name']
        obs['value_as_string'] = pd.cut(self.labels[column], obs_intervals).map(obs_labels)
        obs['visit_occurrence_id'] = obs.index
        obs['observation_concept_id'] = concept_id
        obs['unit_concept_id'] = unit_concept_id
        obs['person_id'] = self.visit_occurrence.loc[obs.visit_occurrence_id, 'person_id']
        obs['observation_date'] = self.admission_data_datetime.date()
        obs['observation_datetime'] = self.admission_data_datetime.time()

        self.observation = self.concat([self.observation, obs])

    def observation_table(self):
        print('Observation table...')
        self.observation = cdm.tables['OBSERVATION']
        age_breaks = np.arange(0, 100, 5)
        weight_breaks = [-np.inf]+np.arange(30, 150, 5).tolist()+[np.inf]
        height_breaks = [-np.inf]+np.arange(120, 210, 5).tolist()+[np.inf]

        self._add_observation(age_breaks, column='age',
                              concept_id=44804452, unit_concept_id=9448)
        self._add_observation(weight_breaks, column='weight',
                              concept_id=3711521, unit_concept_id=9529)
        self._add_observation(height_breaks, column='height',
                              concept_id=4203099, unit_concept_id=8582)

        self.observation['observation_type_concept_id'] = 38000280

        start_index = self.start_index['observation']
        self.observation['observation_id'] = np.arange(start_index,
                                                       start_index+len(self.observation))

        self.observation['observation_date'] = pd.to_datetime(self.observation['observation_date'])


    def _add_drugs(self, chunk):
        df = pd.DataFrame()
        df['_patient'] = chunk.patient
        df['drug_source_value'] = chunk['variable']
        df['drug_type_concept_id'] = 43542358 # Physician administered drug (identified from EHR observation)
        df['visit_occurrence_id'] = df['_patient'].map(self.visit_mapper)
        df['person_id'] = df['_patient'].map(self.person_id_mapper)
        
        delta_start = chunk['start'].apply(pd.Timedelta, unit='second')
        delta_end = chunk['end'].apply(pd.Timedelta, unit='second')
        df['_start'] = self.ref_date + delta_start
        df['_end'] = self.ref_date + delta_end
        df['drug_exposure_start_date'] = df._start.dt.date
        df['drug_exposure_start_datetime'] = df._start.dt.time
        df['drug_exposure_end_date'] = df._end.dt.date
        df['drug_exposure_end_datetime'] = df._end.dt.time
        df['drug_concept_id'] = df['drug_source_value'].map(self.source_to_concept_mapping)
        df = (df.drop(columns=['_start', '_end'])
                .dropna(subset='visit_occurrence_id'))
        if len(self.drug_exposure)==0:
            df['drug_exposure_id'] = self.start_index['drug_exposure'] + df.index
        else:
            df['drug_exposure_id'] = self.drug_exposure.index.max() + 1 + df.index

        return self.concat([self.drug_exposure, df]).set_index('drug_exposure_id')


    def drug_exposure_table(self, start_chunk=0):
        for i, pth_chunk in enumerate(self.med_pths_chunks):
            if i < start_chunk:
                continue
            self.drug_exposure = cdm.tables['DRUG_EXPOSURE'].copy()
            print(f'Drug exposure chunk {i}/{self.n_chunks}')
            chunk = pd.read_parquet(pth_chunk).reset_index()
            self.chunk = chunk
            self.drug_exposure = (chunk.pipe(self._add_drugs)
                                       .drop(columns='_patient'))
            
            self.drug_exposure['drug_exposure_start_date'] = pd.to_datetime(self.drug_exposure['drug_exposure_start_date'])
            self.drug_exposure['drug_exposure_end_date'] = pd.to_datetime(self.drug_exposure['drug_exposure_end_date'])
            self.drug_exposure['drug_exposure_start_datetime'] = self.drug_exposure['drug_exposure_start_datetime'].astype(str)
            self.drug_exposure['drug_exposure_end_datetime'] = self.drug_exposure['drug_exposure_end_datetime'].astype(str)
            
            self.export_table(self.drug_exposure,
                              'DRUG_EXPOSURE',
                              mode='parquet',
                              chunkindex=i,
                              schema=self.drug_exposure_schema,
                              )

    def care_site_table(self):
        print('Care_site table...')
        self.care_site = cdm.tables['CARE_SITE']
        self.labels['unit_type'] = self.labels['unit_type'].map(self.unit_types)
        self.care_site[['care_site_name', 'place_of_service_source_value']] = self.labels[['care_site', 'unit_type']].drop_duplicates()
        self.care_site['place_of_service_concept_id'] = self.care_site['place_of_service_source_value']

        self.care_site['care_site_source_value'] = self.care_site['care_site_name']

        self.locationid_mapper = (self.location[['location_source_value',
                                                 'location_id']]
                                  .set_index('location_source_value')
                                  .to_dict()['location_id'])

        self.care_site['location_id'] = self.care_site['care_site_name'].map(self.locationid_mapper).astype(int)

        self.care_site['care_site_id'] = self.start_index['care_site'] + np.arange(len(self.care_site))

    def domain_table(self):
        print('Domain table...')
        self.domain = cdm.tables['DOMAIN']

        domains = {
            'Visit': 8,
            'Type Concept': 58,
            'Observation': 27,
            'Drug': 13,
            'Unit': 16,
            'Measurement': 21
        }

        self.domain['domain_name'] = domains.keys()
        self.domain['domain_concept_id'] = self.domain['domain_name'].map(domains)

        start_index = self.start_index['domain']
        self.domain['domain_id'] = np.arange(start_index, start_index+len(self.domain))

    def concept_table(self):
        print('Concept_table...')
        self.concept = cdm.tables['CONCEPT']

        self.concept_dic_misc = [
            {'concept_id': 8844,
             'concept_name': 'Other Place of Service',
             'domain_id': 'Visit',
             'vocabulary_id': 'CMS Place of Service',
             'concept_class_id': 'Visit',
             'standard_concept': None,
             'concept_code': 99},
            {'concept_id': 38004285,
             'concept_name': 'Rehabilitation Hospital',
             'domain_id': 'Visit',
             'vocabulary_id': 'NUCC',
             'concept_class_id': 'Visit',
             'standard_concept': 'S',
             'concept_code': '283X00000X'},
        ]

        self.concept_ids_misc = [
            32037,
            9203,
            4021813,
            44818518,
            38000280,
            43542358,
            4318944,
            40481392,
            4149943,
            4305366,
            763903,
            4148496,
            4160026,
            4330427,
            4320169,
            4330442,
            ]

        self.concept_ids_flat = [
            44804452,  # age
            37111521,  # weight
            4203099,  # height
            ]

        self.concept_ids_obs = pd.Series({
            'height': 607590,
            'weight': 4099154
            })

        self.concept_ids_units = self.concept_unit.index.to_list()
        
        idx_med = self.omop_concept['concept_name'].isin(self.kept_med)
        self.concept_ids_med = self.omop_concept.loc[idx_med].index.to_list()

        self.concept_ids_ts = self.cols.concept_id.dropna().astype(int).to_list()

        self.concept_ids = (self.concept_ids_misc
                            + self.concept_ids_obs.to_list()
                            + self.concept_ids_flat
                            + self.concept_ids_ts
                            + self.concept_ids_med
                            + self.concept_ids_units
                            )

        self.concept = self.omop_concept.loc[np.unique(self.concept_ids)]

        self.concept_mapper = self.concept.reset_index().set_index('concept_name')['concept_id']

    def location_table(self):
        '''
        eicu locations are given as numeric codes. we created a location entry 
        for each to preserve the data.
        All other databases are monocentric, a single location was created.
        '''
        print('Location table...')
        location_cols = cdm.tables['LOCATION'].columns

        eicu_loc_id = [
            '404.0', '420.0', '252.0', '90.0', '94.0', '385.0', '136.0', 
            '259.0', '301.0', '227.0', '449.0', '345.0', '248.0', '279.0',
            '122.0', '264.0', '436.0', '391.0', '338.0', '63.0', '266.0',
            '336.0', '167.0', '197.0', '188.0', '337.0', '400.0', '202.0',
            '69.0', '307.0', '199.0', '141.0', '243.0', '73.0', '269.0',
            '440.0', '403.0', '424.0', '58.0', '280.0', '382.0', '245.0',
            '283.0', '394.0', '249.0', '300.0', '208.0', '155.0', '176.0',
            '435.0', '443.0', '452.0', '171.0', '165.0', '154.0', '183.0',
            '390.0', '282.0', '142.0', '407.0', '458.0', '195.0', '95.0',
            '389.0', '358.0', '412.0', '277.0', '425.0', '85.0', '226.0',
            '331.0', '220.0', '152.0', '357.0', '388.0', '79.0', '353.0',
            '271.0', '392.0', '181.0', '318.0', '148.0', '157.0', '405.0',
            '205.0', '268.0', '419.0', '423.0', '224.0', '198.0', '207.0',
            '272.0', '416.0', '67.0', '146.0', '417.0', '144.0', '256.0',
            '328.0', '184.0', '60.0', '281.0', '444.0', '459.0', '175.0',
            '253.0', '217.0', '310.0', '421.0', '413.0', '360.0', '393.0',
            '131.0', '140.0', '66.0', '71.0', '194.0', '411.0', '387.0',
            '384.0', '364.0', '397.0', '422.0', '158.0', '110.0', '125.0',
            '196.0', '402.0', '92.0', '201.0', '396.0', '102.0', '398.0',
            '386.0', '244.0', '383.0', '209.0', '206.0', '355.0', '365.0',
            '68.0', '251.0', '204.0', '433.0', '445.0', '254.0', '210.0',
            '59.0', '143.0', '437.0', '215.0', '123.0', '138.0', '174.0',
            '438.0', '250.0', '112.0', '56.0', '434.0', '258.0', '342.0',
            '428.0', '108.0', '200.0', '262.0', '312.0', '164.0', '203.0',
            '180.0', '182.0', '439.0', '399.0', '275.0', '323.0', '133.0',
            '429.0', '350.0', '447.0', '263.0', '120.0', '273.0', '267.0',
            '381.0', '356.0', '61.0', '401.0', '246.0', '352.0', '408.0',
            '414.0', '83.0', '265.0', '303.0', '96.0', '91.0', '93.0',
            '179.0', '135.0', '212.0', '361.0', '115.0', '84.0', '86.0',
            '363.0', '151.0', '409.0', '156.0', '351.0'
            ]

        location_dic_1 = [
            {'country_concept_id': 4330442,
             'country_source_value': 'US',
             'location_source_value': n} for n in eicu_loc_id
        ]

        location_dic_2 = [
            {'city': 'Bern',
             'country_concept_id': 4330427,
             'country_source_value': 'CH',
             'location_source_value': 'Bern University Hospital'},
            {'city': 'Boston',
             'country_concept_id': 4330442,
             'country_source_value': 'US',
             'location_source_value': 'Beth Israel Deaconess Medical Center'},
            {'city': 'Amsterdam',
             'country_concept_id': 4320169,
             'country_source_value': 'NL',
             'location_source_value': 'Amsterdam University Medical Center'},
        ]
        
        location_dic_3 = [
            {'country_concept_id': 4330442,
             'country_source_value': 'US',
             'location_source_value': 'US'},
            {'country_concept_id': 4320169,
             'country_source_value': 'NL',
             'location_source_value': 'NL'},
            {'country_concept_id': 4330427,
             'country_source_value': 'CH',
             'location_source_value': 'CH'},
            ]

        self.location = pd.DataFrame(location_dic_1
                                     +location_dic_2
                                     +location_dic_3)

        self.location['location_id'] = (self.start_index['location'] 
                                        + np.arange(len(self.location)))
        self.location = self.location.reindex(columns=cdm.tables['LOCATION'].columns)

    def _load_labels(self):
        labels_pth = self.data_pth+'preprocessed_labels.parquet'
        df = pd.read_parquet(labels_pth).reset_index()
        return df

    def export_table(self,
                     table,
                     name,
                     mode='w',
                     chunkindex=None,
                     schema=None):
        """
        Exports a table
        * to csv file if mode is "w" or "a" with the corresponding mode.
        * to parquet if mode is "parquet". This mode requires to specify 
        a chunkindex for saving several parquet files in the same directory.
        """
        if mode in ['w', 'a']:
            savepath = f'{self.savedir}/{name}.csv'
            print(f'Saving {savepath}')
            table.to_csv(savepath, sep=';', index=False, mode=mode)
        elif mode == 'parquet':
            savedir = f'{self.savedir}/{name}/'
            Path(savedir).mkdir(exist_ok=True, parents=True)
            savepath = f'{savedir}{name}_{chunkindex}.parquet'
            print(f'Saving {savepath}')
            table.to_parquet(savepath, schema=schema)

    def export_tables(self):
        """
        saves tables to csv, except the measurements and drug exposure tables
        that are much larger and should be handled differently.
        """
        Path(self.savedir).mkdir(exist_ok=True)

        for name, table in cdm.tables.items():
            self.export_table(table, name)

        self.export_table(self.concept.reset_index(), 'CONCEPT')
        self.export_table(self.death, 'DEATH')
        self.export_table(self.care_site, 'CARE_SITE')
        self.export_table(self.observation,
                          'OBSERVATION',
                          schema=self.observation_schema)
        self.export_table(self.person, 'PERSON')
        self.export_table(self.source_to_concept_map, 'SOURCE_TO_CONCEPT_MAP')
        self.export_table(self.visit_occurrence, 'VISIT_OCCURRENCE')
        self.export_table(self.domain, 'DOMAIN')
        self.export_table(self.location, 'LOCATION')
