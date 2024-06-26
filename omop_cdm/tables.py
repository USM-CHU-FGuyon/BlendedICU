from collections import OrderedDict

import polars as pl
import pandas as pd
import pyarrow as pa

class CDMTables:
    def __init__(self):
        pth = 'auxillary_files/OMOP_CDMv5.4_Field_Level.csv'
        self.field_level = pd.read_csv(pth)
        self._dtype_mapping = {'integer': int,
                               'Integer': int,
                               'float': float,
                               'bigint': int,
                               'varchar(MAX)': str,
                               'varchar(2000)': str,
                               'varchar(1000)': str,
                               'varchar(255)': str,
                               'varchar(250)': str,
                               'varchar(80)': str,
                               'varchar(60)': str,
                               'varchar(50)': str,
                               'varchar(25)': str,
                               'varchar(20)': str,
                               'varchar(10)': str,
                               'varchar(9)': str,
                               'varchar(3)': str,
                               'varchar(2)': str,
                               'varchar(1)': str,
                               'datetime': object,
                               'date': object}

        self.tables = self.initialize_tables()
        #TODO : good enough for now but I should automatize this
        self.schemas = {
            'measurement': self._measurement_schema(),
            'observation': self._observation_schema(),
            'visit_occurrence': self._visit_occurrence_schema(),
            'drug_exposure': self._drug_exposure_schema(),
            'death': self._death_schema(),
            'location': self._location_schema(),
            'domain': self._domain_schema(),
            'care_site': self._care_site_schema(),
            'person': self._person_schema(),
            'concept': self._concept_schema(),
            'source_to_concept_map': self._source_to_concept_map_schema(),
            'cdm_source': None,
            'cohort': None,
            'cohort_definition': None,
            'concept_ancestor': None,
            'concept_class': None,
            'concept_relationship': None,
            'concept_synonym': None,
            'condition_era': None,
            'condition_occurrence': self._condition_occurrence_schema(),
            'cost': None,
            'device_exposure': None,
            'dose_era': None,
            'drug_era': None,
            'drug_strength': self._drug_strength_schema(),
            'episode': None,
            'episode_event': None,
            'fact_relationship': None,
            'metadata': None,
            'note': None,
            'note_nlp': None,
            'observation_period': None,
            'payer_plan_period': None,
            'procedure_occurrence': None,
            'provider': None,
            'relationship': None,
            'specimen': None,
            'visit_detail': None,
            'vocabulary': None,
            }

    def _dtype(self, dtypestr):
        return self._dtype_mapping[dtypestr]

    def field_required(self, field, table ):
        idx = ((self.field_level.cdmTableName=='CONDITION_OCCURRENCE')
               & (self.field_level.cdmFieldName=='visit_occurrence_id'))
        
        isrequired = self.field_level.loc[idx, 'isRequired']
        
        if isrequired.empty:
            return False
        return (isrequired=='Yes').all()
    
    def initialize_tables(self):
        cdmtables = {}

        for tablename, fields in self.field_level.groupby('cdmTableName'):

            series = {fieldname: pd.Series([], dtype=self._dtype(fieldtype))
                      for fieldname, fieldtype in zip(fields.cdmFieldName,
                                                      fields.cdmDatatype)}

            cdmtables[tablename] = pd.DataFrame(series)

        return cdmtables
    

    @staticmethod
    def _measurement_schema():
        schema = OrderedDict([('value_as_number', pl.Float32),
                            ('time', pl.Float32),
                            ('visit_occurrence_id', pl.Int64),
                            ('visit_source_value', pl.String),
                            ('person_id', pl.Int64),
                            ('measurement_datetime', pl.Datetime),
                            ('measurement_date', pl.Date),
                            ('measurement_time', pl.Duration),
                            ('measurement_concept_id', pl.Int32),
                            ('measurement_source_value', pl.String),
                            ('unit_source_value', pl.String),
                            ('unit_concept_id', pl.Int32),
                            ('measurement_id', pl.UInt64),
                            ('measurement_type_concept_id', pl.Int32),
                            ('operator_concept_id', pl.Int32),
                            ('value_as_concept_id', pl.Int32),
                            ('range_low', pl.Float32),
                            ('range_high', pl.Float32),
                            ('provider_id', pl.Int32),
                            ('visit_detail_id', pl.Int32),
                            ('measurement_source_concept_id', pl.Int32),
                            ('unit_source_concept_id', pl.Int32),
                            ('value_source_value', pl.String),
                            ('measurement_event_id', pl.Int32),
                            ('meas_event_field_concept_id', pl.Int32)])
        
        return schema
    
    
    @staticmethod
    def _drug_strength_schema():
        schema = OrderedDict([('drug_concept_id', pl.Int32),
                            ('ingredient_concept_id', pl.Int32),
                            ('amount_value', pl.Float32),
                            ('amount_unit_concept_id', pl.Int32),
                            ('numerator_value', pl.Float32),
                            ('numerator_unit_concept_id', pl.Int32),
                            ('denominator_value', pl.Float32),
                            ('denominator_unit_concept_id', pl.Int32),
                            ('box_size', pl.Int32),
                            ('valid_start_date', pl.Date),
                            ('valid_end_date', pl.Date),
                            ('invalid_reason', pl.String),
                            ])
        return schema
    
    @staticmethod
    def _observation_schema():
        schema = pa.schema([('observation_id', pa.int32()),
                            ('person_id', pa.int64()),
                            ('observation_concept_id', pa.int32()),
                            ('observation_date', pa.date32()),
                            ('observation_datetime', pa.timestamp('s')),
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
    
    @staticmethod
    def _drug_exposure_schema():
        schema = OrderedDict([('drug_source_value', pl.String),
                            ('drug_type_concept_id', pl.Int32),
                            ('visit_occurrence_id', pl.Int64),
                            ('person_id', pl.Int64),
                            ('drug_exposure_start_date', pl.Date),
                            ('drug_exposure_start_datetime', pl.Datetime),
                            ('drug_exposure_end_date', pl.Date),
                            ('drug_exposure_end_datetime', pl.Datetime),
                            ('drug_concept_id', pl.Int32),
                            ('drug_exposure_id', pl.UInt64),
                            ('verbatim_end_date', pl.Date),
                            ('stop_reason', pl.String),
                            ('refills', pl.String),
                            ('quantity', pl.String),
                            ('days_supply', pl.Float32),
                            ('sig', pl.String),
                            ('route_concept_id', pl.Int32),
                            ('lot_number', pl.String),
                            ('provider_id', pl.Int32),
                            ('visit_detail_id', pl.Int32),
                            ('drug_source_concept_id', pl.Int32),
                            ('route_source_value', pl.String),
                            ('dose_unit_source_value', pl.String),
                            ])
        return schema
    
    @staticmethod
    def _visit_occurrence_schema():
        schema = pa.schema([('visit_occurrence_id', pa.int64()),
                            ('person_id', pa.int64()),
                            ('visit_concept_id', pa.int32()),
                            ('visit_start_date', pa.date32()),
                            ('visit_start_datetime', pa.timestamp('s')),
                            ('visit_end_date', pa.date32()),
                            ('visit_end_datetime', pa.timestamp('s')),
                            ('visit_type_concept_id', pa.int32()),
                            ('provider_id', pa.int32()),
                            ('care_site_id', pa.int32()),
                            ('visit_source_value', pa.string()),
                            ('visit_source_concept_id', pa.int32()),
                            ('admitted_from_concept_id', pa.int32()),
                            ('admitted_from_source_value', pa.string()),
                            ('discharged_to_concept_id', pa.int32()),
                            ('discharged_to_source_value', pa.string()),
                            ('preceding_visit_occurrence_id', pa.int64()),
                            ])
        return schema
    
    @staticmethod
    def _person_schema():    
        schema = pa.schema([('person_id', pa.int64()),
                            ('gender_concept_id', pa.int32()),
                            ('year_of_birth', pa.int32()),
                            ('month_of_birth', pa.int32()),
                            ('day_of_birth', pa.int32()),
                            ('birth_datetime', pa.timestamp('s')),
                            ('race_concept_id', pa.int32()),
                            ('ethnicity_concept_id', pa.int32()),
                            ('location_id', pa.int32()),
                            ('provider_id', pa.int32()),
                            ('care_site_id', pa.int32()),
                            ('person_source_value', pa.string()),
                            ('gender_source_value', pa.string()),
                            ('gender_source_concept_id', pa.int32()),
                            ('race_source_value', pa.string()),
                            ('race_source_concept_id', pa.int32()),
                            ('ethnicity_source_value', pa.string()),
                            ('ethnicity_source_concept_id', pa.int32())
                            ])
        return schema
    
    
    @staticmethod
    def _death_schema():
        schema = pa.schema([('person_id', pa.int64()),
                            ('death_date', pa.date32()),
                            ('death_datetime', pa.timestamp('s')),
                            ('death_type_concept_id', pa.int32()),
                            ('cause_concept_id', pa.int32()),
                            ('cause_source_value', pa.string()),
                            ('cause_source_concept_id', pa.int32()),
                            ])
        return schema
    
    @staticmethod
    def _location_schema():
        schema = OrderedDict([('location_id', pl.Int64),
                            ('address_1', pl.String),
                            ('address_2', pl.String),
                            ('city', pl.String),
                            ('state', pl.String),
                            ('zip', pl.String),
                            ('county', pl.String),
                            ('location_source_value', pl.String),
                            ('country_concept_id', pl.Int64),
                            ('country_source_value', pl.String),
                            ('latitude', pl.Float32),
                            ('longitude', pl.Float32),
                            ])
        return schema
    
    @staticmethod
    def _domain_schema():
        schema = OrderedDict([('domain_id', pl.String),
                              ('domain_name', pl.String),
                              ('domain_concept_id', pl.Int64)])
        return schema

    @staticmethod
    def _care_site_schema():
        schema = pa.schema([('care_site_id', pa.int64()),
                            ('care_site_name', pa.string()),
                            ('place_of_service_concept_id', pa.int32()),
                            ('location_id', pa.int32()),
                            ('care_site_source_value', pa.string()),
                            ('place_of_service_source_value', pa.string())])
        return schema  

    @staticmethod
    def _concept_schema():
        schema = pa.schema([('concept_id', pa.int32()),
                            ('concept_name', pa.string()),
                            ('domain_id', pa.string()),
                            ('vocabulary_id', pa.string()),
                            ('concept_class_id', pa.string()),
                            ('standard_concept', pa.string()),
                            ('concept_code', pa.string()),
                            ('valid_start_date', pa.date32()),
                            ('valid_end_date', pa.date32()),
                            ('invalid_reason', pa.string())])
        return schema
    
    @staticmethod
    def _source_to_concept_map_schema():
        schema = pa.schema([('source_code', pa.string()),
                            ('source_concept_id', pa.int32()),
                            ('source_vocabulary_id', pa.string()),
                            ('source_code_description', pa.string()),
                            ('target_concept_id', pa.int32()),
                            ('target_vocabulary_id', pa.string()),
                            ('valid_start_date', pa.date32()),
                            ('valid_end_date', pa.date32()),
                            ('invalid_reason', pa.string())])
        return schema

    @staticmethod
    def _condition_occurrence_schema():
        schema = pa.schema([('condition_occurrence_id', pa.int64()),
                            ('person_id', pa.int64()),
                            ('condition_concept_id', pa.int32()),
                            ('condition_start_date', pa.date32()),
                            ('condition_start_datetime', pa.timestamp('s')),
                            ('condition_end_date', pa.date32()),
                            ('condition_end_datetime', pa.timestamp('s')),
                            ('condition_type_concept_id', pa.int32()),
                            ('condition_status_concept_id', pa.int32()),
                            ('stop_reason', pa.string()),
                            ('provider_id', pa.int32()),
                            ('visit_occurrence_id', pa.int64()),
                            ('visit_detail_id', pa.int64()),
                            ('condition_source_value', pa.string()),
                            ('condition_source_concept_id', pa.int32()),
                            ('condition_status_source_value', pa.string())])
        return schema


cdm = CDMTables()
