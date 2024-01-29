import pandas as pd


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


cdm = CDMTables()
