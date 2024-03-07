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
        self.schemas = {
            'measurement': self._measurement_schema(),
            'observation': self._observation_schema(),
            'drug_exposure': self._drug_exposure_schema(),
            }

    def _dtype(self, dtypestr):
        return self._dtype_mapping[dtypestr]

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
        schema = pa.schema([('value_as_number', pa.float32()),
                            ('time', pa.float32()),
                            ('visit_occurrence_id', pa.string()),
                            ('visit_start_date', pa.date32()),
                            ('visit_source_value', pa.string()),
                            ('person_id', pa.string()),
                            ('measurement_datetime', pa.timestamp('s')),
                            ('measurement_date', pa.date32()),
                            ('measurement_time', pa.time32('s')),
                            ('measurement_concept_id', pa.int32()),
                            ('measurement_source_value', pa.float32()),
                            ('unit_source_value', pa.string()),
                            ('unit_concept_id', pa.int32()),
                            ('measurement_id', pa.float32()),
                            ('measurement_type_concept_id', pa.int32()),
                            ('operator_concept_id', pa.int32()),
                            ('value_as_concept_id', pa.int32()),
                            ('range_low', pa.float32()),
                            ('range_high', pa.float32()),
                            ('provider_id', pa.int32()),
                            ('visit_detail_id', pa.int32()),
                            ('measurement_source_concept_id', pa.int32()),
                            ('unit_source_concept_id', pa.int32()),
                            ('value_source_value', pa.float32()),
                            ('measurement_event_id', pa.int32()),
                            ('meas_event_field_concept_id', pa.int32())
                            ])
        return schema
    
    @staticmethod
    def _observation_schema():
        schema = pa.schema([('observation_id', pa.int32()),
                            ('person_id', pa.string()),
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
                            ('visit_occurrence_id', pa.string()),
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
        schema = pa.schema([('drug_source_value', pa.string()),
                            ('drug_type_concept_id', pa.int32()),
                            ('visit_occurrence_id', pa.string()),
                            ('person_id', pa.string()),
                            ('drug_exposure_start_date', pa.date32()),
                            ('drug_exposure_start_datetime', pa.timestamp('s')),
                            ('drug_exposure_end_date', pa.date32()),
                            ('drug_exposure_end_datetime', pa.timestamp('s')),
                            ('drug_concept_id', pa.int32()),
                            ('drug_exposure_id', pa.int32()),
                            ('verbatim_end_date', pa.date32()),
                            ('stop_reason', pa.string()),
                            ('refills', pa.string()),
                            ('quantity', pa.string()),
                            ('days_supply', pa.float32()),
                            ('sig', pa.string()),
                            ('route_concept_id', pa.int32()),
                            ('lot_number', pa.string()),
                            ('provider_id', pa.int32()),
                            ('visit_detail_id', pa.int32()),
                            ('drug_source_concept_id', pa.int32()),
                            ('route_source_value', pa.string()),
                            ('dose_unit_source_value', pa.string()),
                            ])
        return schema
    


cdm = CDMTables()
