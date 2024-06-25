from pathlib import Path
import warnings
from datetime import datetime
import hashlib

import pandas as pd
import numpy as np
import polars as pl

from blended_preprocessing.timeseries import blendedicuTSP
from omop_cdm import cdm


class OMOP_converter(blendedicuTSP):
    def __init__(self,
                 initialize_tables=False,
                 recompute_index=True,
                 ts_pths=None,
                 med_pths=None):
        super().__init__()
        self.tables_initialized = False
        self.data_pth = self.savepath
        self.ref_date = datetime(year=2000, month=1, day=1)
        self.pl_ref_date = pl.datetime(year=2023, month=1, day=1)
        
        self.end_date = datetime(year=2099, month=12, day=31)
        self.adm_measuredat = self.flat_hr_from_adm.total_seconds()
        self.admission_data_datetime = (self.ref_date + pd.Timedelta(self.adm_measuredat, unit='second'))
        self.omop_concept = self._get_omop_concept()
        self.savedir = f'{self.data_pth}/OMOP-CDM/'
        Path(self.savedir).mkdir(exist_ok=True)
        
        self.labels = self._load_labels()
        self.diagnoses = self._load_diagnoses()
                
        self.lf_ts = pl.scan_parquet(self.dir_long_timeseries+'*.parquet')
        self.lf_med = pl.scan_parquet(self.dir_long_medication+'*.parquet')
        
        self.start_index = {
            'observation': 3000000,
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
        
        
        self.unit_mapping = self._get_unit_mapping()
        self.units_concept_ids = np.unique([*self.unit_mapping.values()])
        self.concept_unit = self.omop_concept.loc[self.units_concept_ids]
        self.concept, self.concept_table = self._concept_table()
        self.mapping_concepts = self.concept.set_index('concept_name').concept_id.to_dict()
        self.mapping_unit_code_concept = self._mapping_unit_code_concept()
        self.units = self._get_units()
        self.schemas = cdm.schemas
        
        if initialize_tables:
            self._initialize_tables()

    def _mapping_unit_code_concept(self):
        return (self.omop_concept
                .loc[self.omop_concept.vocabulary_id=='UCUM', 'concept_code']
                .reset_index()
                .set_index('concept_code')
                .concept_id
                .to_dict()
                )

        
    def _get_unit_mapping(self):
        """
        Mapping between the variable name in blendedICU and the OMOP concept_id
        of the unit
        """
        
        flat_unit_mapping = {'raw_age': 9448,
                             'raw_height': 8582,
                             'raw_weight': 9529}
        
        ts_unit_mapping = (self.ts_variables[['blended', 'unit_concept_id']]
                           .set_index('blended')
                           .unit_concept_id
                           .to_dict())
        
        unit_mapping = flat_unit_mapping | ts_unit_mapping
        return unit_mapping
        
        
    def _initialize_tables(self):
        print('\nInitializing tables')
        self.source_to_concept_map_table()
        self.location_table()
        self.care_site_table()
        self.person_table()
        self.visit_occurrence_table()
        self.death_table()
        self.domain_table()
        self.observation_table()
        self.condition_occurrence_table()
        print('   -> Done')
        self.export_flat_tables()
        self.tables_initialized = True
        
    def _get_omop_concept(self):
        pth_concept_table = fr'{self.aux_pth}OMOP_vocabulary/CONCEPT.parquet'
        return pd.read_parquet(pth_concept_table,
                               columns=['concept_name',
                                        'concept_code',
                                        'domain_id',
                                        'vocabulary_id',
                                        'concept_class_id',
                                        'standard_concept'])

        
    def condition_occurrence_table(self):
        
        condition_occurrence = cdm.tables['CONDITION_OCCURRENCE']
        
        self.condition_occurrence = condition_occurrence.assign(
            person_id=self.diagnoses.uniquepid.map(self.person_id_mapper),
            visit_occurrence_id=self.diagnoses.patient.map(self.visit_mapper),
            condition_concept_id=self.diagnoses.diagnosis_concept_id,
            condition_start_date=self.diagnoses.diagnosis_start.dt.date,
            condition_start_datetime=self.diagnoses.diagnosis_start,
            condition_type_concept_id=32817,#EHR concept type
            condition_source_value=self.diagnoses.diagnosis_source_value
            )
        
        unique_id_cols = ['person_id',
                          'condition_source_value',
                          'condition_start_datetime']
        
        self.condition_occurrence['condition_occurrence_id'] = self._create_id(self.condition_occurrence,
                                                                               unique_id_cols,
                                                                               prefix=5)

    
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


    def _create_id(self, df, unique_key, prefix):
        return (df[unique_key]
                .astype(str).sum(axis=1)
                .apply(self._hash_values, prefix=prefix)
                .astype(np.int64))

    def _hash_values(self, series, prefix):
        '''
        12 digits should be enough for patients or visits.
        '''
        hashed = hashlib.md5(series.encode('utf-8')).hexdigest()
        int_hash = int(hashed, 16) %10**12
        return f'{prefix}{int_hash}'
        
    def _id_mapper(self, hashed_series, prefix):
        return pd.Series(hashed_series.apply(self._hash_values, prefix=prefix).values,
                         index=hashed_series).astype(np.int64)
    
    def person_table(self):
        print('Person table...')
        person = cdm.tables['PERSON']
        person_labels = self.labels.drop_duplicates(subset='uniquepid').reset_index()
        person['gender_concept_id'] = person_labels.sex.map({1: 8507,
                                                             0: 8532})
        person['year_of_birth'] = self.ref_date.year - person_labels['raw_age']
        person['birth_datetime'] = (person.year_of_birth
                                         .apply(lambda x: x if x!=x else datetime(year=int(x),
                                                                   month=1,
                                                                   day=1,
                                                                   hour=0,
                                                                   minute=0,
                                                                   second=0)))
        person['person_source_value'] = person_labels.uniquepid
        person['gender_source_value'] = person_labels.sex.astype(str)
        person['location_id'] = (person_labels.source_dataset.map(
                                                        {'amsterdam': 'NL',
                                                         'hirid': 'CH',
                                                         'mimic': 'US',
                                                         'mimic3': 'US',
                                                         'eicu': 'US'})
                                        .map(self.locationid_mapper))
        self.person = person
        
        self.person_id_mapper = self._id_mapper(person['person_source_value'],
                                                prefix=1)
        person['person_id'] = person['person_source_value'].map(self.person_id_mapper)
        
        self.person = person
        
    def visit_occurrence_table(self):
        print('Visit Occurrence...')
        visit_occurrence = cdm.tables['VISIT_OCCURRENCE'].copy()

        visit_occurrence['_source_person_id'] = self.labels.uniquepid
        visit_occurrence['_source_visit_id'] = self.labels.patient
        visit_occurrence['visit_source_value'] = self.labels.patient
        visit_occurrence['visit_start_date'] = self.ref_date.date()
        visit_occurrence['visit_start_datetime'] = self.ref_date
        visit_occurrence['visit_concept_id'] = 32037
        visit_occurrence['visit_type_concept_id'] = 44818518
        visit_occurrence['admitted_from_concept_id'] = (
                                                self.labels.origin
                                                .map(self.admission_origins)
                                                .map(self.concept_mapping))
        visit_occurrence['admitted_from_source_value'] = self.labels.origin
        visit_end_datetime = self.ref_date + self.labels.lengthofstay
        visit_occurrence['visit_end_date'] = visit_end_datetime.dt.date
        visit_occurrence['visit_end_datetime'] = visit_end_datetime.dt.round('s')
        visit_occurrence['person_id'] = visit_occurrence['_source_person_id'].map(self.person_id_mapper)
        visit_occurrence['discharged_to_source_value'] = self.labels.discharge_location
        visit_occurrence['discharged_to_concept_id'] = self.labels.discharge_location.map(self.concept_mapping)
        
        self.visit_mapper = self._id_mapper(visit_occurrence['visit_source_value'],
                                            prefix=2)
        visit_occurrence['visit_occurrence_id'] = visit_occurrence['visit_source_value'].map(self.visit_mapper)

        self.visit_occurrence = (visit_occurrence
                                 .drop(columns=['_source_person_id',
                                                '_source_visit_id'])
                                 .set_index('visit_occurrence_id', drop=False))
        
        self.visit_occurrence_ids = self.visit_occurrence.visit_occurrence_id.unique()
        
        self.labels.index = self.labels.patient.map(self.visit_mapper)
        self.labels = self.labels.rename_axis('visit_occurrence_id')

    def death_table(self):
        print('Death table...')
        self.death_labels = self.labels.loc[self.labels.mortality == 1]

        self.death = cdm.tables['DEATH']
        self.death['person_id'] = self.death_labels.index.map(self.visit_occurrence.person_id)
        self.death.index = self.death_labels.index
        death_datetimes = self.ref_date + self.death_labels.lengthofstay

        self.death['death_date'] = death_datetimes.dt.date
        self.death['death_datetime'] = death_datetimes.dt.round('s')
        self.death = (self.death
                      .reset_index(drop=True)
                      .drop_duplicates(subset='person_id'))


    def measurement_table(self):
        if not self.tables_initialized:
            self._initialize_tables()
        
        mapping_person_id =self.person_id_mapper.to_dict()
        mapping_visit_id = self.visit_mapper.to_dict()
        mapping_concept_id =self.concept_mapping.to_dict()
            
        lf = self.lf_ts
        
        lf = (lf
                .with_columns(
                    self.hash('measurement_id'),
                    pl.col('patient').replace(mapping_person_id, default=0).alias('person_id'),
                    pl.col('patient').replace(mapping_visit_id, default=0).alias('visit_occurrence_id'),
                    pl.col('variable').replace(mapping_concept_id, default=0).alias('measurement_concept_id'),
                    (self.pl_ref_date+pl.duration(seconds=pl.col('time'))).alias('measurement_datetime'),
                    pl.col('variable').replace(self.unit_mapping, default=0).alias('unit_concept_id'),
                    )
              .rename({'variable': 'measurement_source_value',
                      'value': 'value_as_number'}))
        
        lf = pl.concat([lf, pl.LazyFrame(schema=self.schemas['measurement'])])
        self.save(lf, self.savedir+'/MEASUREMENT.parquet')


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
        obs['observation_datetime'] = self.admission_data_datetime
        self.observation = self.concat([self.observation, obs])

    def observation_table(self):
        print('Observation table...')
        self.observation = cdm.tables['OBSERVATION']
        age_breaks = np.arange(0, 100, 5)
        weight_breaks = [-np.inf]+np.arange(30, 150, 5).tolist()+[np.inf]
        height_breaks = [-np.inf]+np.arange(120, 210, 5).tolist()+[np.inf]

        self._add_observation(age_breaks,
                              column='raw_age',
                              concept_id=44804452,
                              unit_concept_id=9448)
        self._add_observation(weight_breaks,
                              column='raw_weight',
                              concept_id=3711521,
                              unit_concept_id=9529)
        self._add_observation(height_breaks,
                              column='raw_height',
                              concept_id=607590,
                              unit_concept_id=8582)

        self.observation['observation_type_concept_id'] = 38000280

        start_index = self.start_index['observation']
        self.observation['observation_id'] = np.arange(start_index,
                                                       start_index+len(self.observation))

        self.observation['observation_date'] = pd.to_datetime(self.observation['observation_date'])

    @staticmethod
    def _hash(alias):
        return pl.concat_str(pl.all().replace(None, ''), separator='|').hash().reinterpret(signed=True).abs().alias(alias)

    def drug_exposure_table(self):
        raise UserWarning('dosage is not fully omop-ized, it is advised to check the data carefully before using.')
        if not self.tables_initialized:
            self._initialize_tables()

        mapping_person_id = self.visit_occurrence['person_id'].to_dict()
        mapping_visit_id = self.visit_mapper.to_dict()
        mapping_concept_id =self.concept_mapping.to_dict()

        lf = self.lf_med

        col_expr = [pl.lit(None).cast(dtype).alias(col)
                    for col, dtype in self.schemas['drug_exposure'].items()
                    if col not in lf.columns]

        df = (lf
              .with_columns(
                  col_expr
                  )
              .with_columns(
                  self.hash('drug_exposure_id'),
                  pl.col('patient').replace(mapping_person_id, default=None).cast(pl.Int64).alias('person_id'),
                  pl.col('patient').replace(mapping_visit_id, default=None).alias('visit_concept_id'),
                  pl.col('variable').replace(mapping_concept_id, default=0).cast(pl.Int32).alias('drug_concept_id'),
                  (self.pl_ref_date+ pl.col('start')).alias('drug_exposure_start_datetime'),
                  (self.pl_ref_date+ pl.col('start')).cast(pl.Date).alias('drug_exposure_start_date'),
                  (self.pl_ref_date+ pl.col('end')).alias('drug_exposure_end_datetime'),
                  (self.pl_ref_date+ pl.col('end')).cast(pl.Date).alias('drug_exposure_end_date'),
                  pl.lit(43542358).alias('drug_type_concept_id'),
                  pl.col('original_drugname').alias('drug_source_value'),
                  pl.col('route').alias('route_source_value'),
                  pl.col('dose_unit').alias('dose_unit_source_value')
                  )
              .drop('variable', 'start', 'end', 'patient', 'original_drugname',
                    'route', 'dose_unit')
              .rename({'dose': '_dose'})
              ).collect(streaming=True)

        self.drug_strength_units = (df.group_by('drug_concept_id')
                                    .agg(pl.col('dose_unit_source_value').mode().first().alias('_most_common_unit')))
        
        df = (df
              .join(self.drug_strength_units, on='drug_concept_id')
              .with_columns(
            pl.when(pl.col('dose_unit_source_value')==pl.col('_most_common_unit'))
            .then(pl.col('_dose')).alias('quantity')
            )
            )

        self.save(df, self.savedir+'/DRUG_EXPOSURE.parquet')
        
        self.drug_strength_table()
        

    def care_site_table(self):
        print('Care_site table...')
        care_site = cdm.tables['CARE_SITE']
        self.labels['unit_type'] = self.labels['unit_type'].replace(self.unit_types)
        
        care_sites = (self.labels[['care_site', 'unit_type']]
                      .drop_duplicates()
                      .reset_index(drop=True))
        
        care_site[['care_site_name', 'place_of_service_source_value']] = care_sites
        care_site['care_site_source_value'] = care_site['care_site_name']

        self.locationid_mapper = (self.location[['location_source_value',
                                                 'location_id']]
                                  .set_index('location_source_value')
                                  .to_dict()['location_id'])

        care_site['location_id'] = care_site['care_site_name'].map(self.locationid_mapper).astype(int)
        
        unique_key_cols = ['care_site_name', 'place_of_service_source_value']
        
        care_site['care_site_id'] = self._create_id(care_site,
                                                    unique_key_cols,
                                                    prefix=6)
        
        self.care_site = care_site

    def observation_period_table(self):
        mapping_person_id = self.visit_occurrence[['person_id', 'visit_source_value']].set_index('visit_source_value').person_id.to_dict()

        lf = (self.lf_ts
              .with_columns(
                  pl.col('patient').replace(mapping_person_id).alias('person_id'),
                  (self.pl_ref_date+pl.duration(seconds=pl.col('time'))).cast(pl.Date).alias('time'),
                  
                  )
              .group_by(["person_id"])
              .agg(pl.min('time').alias('observation_period_start_date'),
                   pl.max("time").alias('observation_period_end_date'))
              .with_columns(
                  pl.lit(32817).alias('period_type_concept_id'),
                  self.hash('observation_period_id'),
                  )
              )
        self.save(lf, self.savedir+'/OBSERVATION_PERIOD.parquet')
        

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
        self.domain['domain_id'] = np.arange(start_index, start_index+len(self.domain)).astype(str)

    def _concept_table(self):
        print('Concept_table...')
        concept = cdm.tables['CONCEPT']

        concept_ids_misc = [
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

        concept_ids_flat = [
            4265453,  # age
            ]

        self.concept_ids_obs = pd.Series({
            'raw_height': 607590,
            'raw_weight': 4099154
            })

        concept_ids_units = self.concept_unit.index.to_list()
        
        idx_med = self.omop_concept['concept_name'].isin(self.kept_med)
        concept_ids_med = self.omop_concept.loc[idx_med].index.to_list()

        concept_ids_ts = self.cols.concept_id.dropna().astype(int).to_list()

        concept_ids = (concept_ids_misc
                       + self.concept_ids_obs.to_list()
                       + concept_ids_flat
                       + concept_ids_ts
                       + concept_ids_med
                       + concept_ids_units)

        concept_data = self.omop_concept.loc[np.unique(concept_ids)].reset_index()
        
        concept = pd.concat([concept, concept_data]).set_index('concept_id', drop=False)

        concept_mapper = concept.set_index('concept_name')['concept_id']
        return concept, concept_mapper

    def drug_strength_table(self):
        lf_schema = pl.LazyFrame(schema=self.schemas['drug_strength'])
        
        try:
            lf = self.drug_strength_units
        except AttributeError: 
            lf = (pl.scan_parquet( self.savedir+'/DRUG_EXPOSURE.parquet')
                  .group_by('drug_concept_id').agg(pl.col('_most_common_unit').first()))
        
        lf = (self.drug_strength_units
              .with_columns(
                  pl.col('drug_concept_id').alias('ingredient_concept_id'),
                  pl.col('_most_common_unit').replace(self.mapping_unit_code_concept, default=0).cast(pl.Int32).alias('amount_unit_concept_id')
                  )
              .drop('_most_common_unit'))
        
        lf = pl.concat([lf, lf_schema], how='align')
        
        self.save(lf, self.savedir+'/DRUG_STRENGTH.parquet')
        


    def location_table(self):
        '''
        eicu locations are given as numeric codes. we created a location entry 
        for each to preserve the data.
        All other databases are monocentric, a single location was created.
        '''
        print('Location table...')

        eicu_loc_id = [
               404, 420, 252,  90,  94, 385, 136, 259, 301, 227, 449, 345, 248,
               279, 122, 264, 436, 391, 338,  63, 266, 336, 167, 197, 188, 337,
               400, 202,  69, 307, 199, 141, 243,  73, 269, 440, 403, 424,  58,
               280, 382, 245, 283, 394, 249, 300, 208, 155, 176, 435, 443, 452,
               171, 165, 154, 183, 390, 282, 142, 407, 458, 195,  95, 389, 358,
               412, 277, 425,  85, 226, 331, 220, 152, 357, 388,  79, 353, 271,
               392, 181, 318, 148, 157, 405, 205, 268, 419, 423, 224, 198, 207,
               272, 416,  67, 146, 417, 144, 256, 328, 184,  60, 281, 444, 459,
               175, 253, 217, 310, 421, 413, 360, 393, 131, 140,  66,  71, 194,
               411, 387, 384, 364, 397, 422, 158, 110, 125, 196, 402,  92, 201,
               396, 102, 398, 386, 244, 383, 209, 206, 355, 365,  68, 251, 204,
               433, 445, 254, 210,  59, 143, 437, 215, 123, 138, 174, 438, 250,
               112,  56, 434, 258, 342, 428, 108, 200, 262, 312, 164, 203, 180,
               182, 439, 399, 275, 323, 133, 429, 350, 447, 263, 120, 273, 267,
               381, 356,  61, 401, 246, 352, 408, 414,  83, 265, 303,  96,  91,
                93, 179, 135, 212, 361, 115,  84,  86, 363, 151, 409, 156, 351]

        location_dic_1 = [
            {'country_concept_id': 4330442,
             'country_source_value': 'US',
             'location_source_value': str(n)} for n in eicu_loc_id
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
    
    def _load_diagnoses(self):
        df = pd.read_parquet(self.diag_savepath).reset_index()
        return df

    def _visits_exist(self, table, name):
        '''
        For a table that has a visit_occurrence_id column, ensures that 
        every visit_occurrence_id entry is in the VISIT_OCCURRENCE table.
        '''
        col = 'visit_occurrence_id'
        if cdm.field_required(col, table):
            table_visits = table[col].drop_duplicates()
            found = table_visits.isin(self.visit_occurrence_ids)
            if not found.all():
                self.found = found
                self.table = table
                self.col = col
                notfound = table_visits.loc[~found].to_list()
                warnings.warn(
                    UserWarning(
                    f'{len(notfound)} visit_occurrence_ids from {name} were not'
                    f'found in visit_occurrence table'))
        

    def _sanity_checks(self, table, name):
        '''
        A set of sanity checks that will be made at every table export.
        '''
        self._visits_exist(table, name)
        

    def export_table(self, table, name, chunkindex=None):
        """
        Exports a table to parquet.
        If a chunkindex is specified, the file is saved in a directory that
            will contain other parquet files,
        else it is saved as a single parquet file.
        """
        table = table.reset_index(drop=True)
        self._sanity_checks(table, name)

        schema = self.schemas[name.lower()]

        if chunkindex is None:
            savepath = f'{self.savedir}/{name}.parquet'
        else:
            savepath = f'{self.savedir}/{name}/{name}_{chunkindex}.parquet'
            Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        print(f'Saving {savepath}')
        table.to_parquet(savepath, schema=schema)

    def _create_empty_tables(self):
        '''Creates empty parquet tables, except for measurement
        and drug_exposure which will be saved as multiple parquet.'''
        print('Initializing empty tables...')
        for name, table in cdm.tables.items():
            if name.lower() not in ['measurement',
                                    'drug_exposure',
                                    'drug_strength']:
                self.export_table(table, name)

    def export_flat_tables(self):
        """
        exports all flat tables.
        """
        self._create_empty_tables()
        
        self.export_table(self.concept, 'CONCEPT')
        self.export_table(self.death, 'DEATH')
        self.export_table(self.care_site, 'CARE_SITE')
        self.export_table(self.person, 'PERSON')
        self.export_table(self.source_to_concept_map, 'SOURCE_TO_CONCEPT_MAP')
        self.export_table(self.visit_occurrence, 'VISIT_OCCURRENCE')
        self.export_table(self.domain, 'DOMAIN')
        self.export_table(self.location, 'LOCATION')
        self.export_table(self.observation, 'OBSERVATION')
        self.export_table(self.condition_occurrence, 'CONDITION_OCCURRENCE')
