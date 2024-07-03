import operator
from functools import reduce

import polars as pl

from database_processing.dataprocessor import DataProcessor

class NewMedicationProcessor(DataProcessor):
    """
    This class allows to process the drug data of each database in a unified 
    manner. It maps the brand names and labels in the source database to 
    omop ingredients using the drug mapping (auxillary_files/medications.json).
    The outputs a dataframe that is saved as medication.parquet file or as 
    several parquet chunks when filesize is inconvenient.
    
    
    """
    def __init__(self,
                 dataset,
                 lf_med,
                 lf_labels,
                 col_pid,
                 col_med,
                 col_start,
                 col_los,
                 col_dose,
                 col_dose_unit,
                 col_route,
                 col_end,
                 dose_unit_conversion_dic=None,
                 unit_offset=None,
                 offset_calc=False,
                 col_admittime=None):
        super().__init__(dataset)

        self.med_mapping = self._load_med_mapping()

        if not offset_calc:
            if unit_offset is None:
                raise ValueError('unit_offset should be specified.')
        
        self.lf_labels = lf_labels
        self.lf_med = lf_med
        self.col_med = col_med
        self.col_pid = col_pid
        self.col_start = col_start
        self.col_los = col_los
        self.col_admittime = col_admittime
        self.offset_calc = offset_calc
        self.unit_offset = unit_offset
        self.col_dose = col_dose
        self.col_dose_unit = col_dose_unit
        self.col_route = col_route
        self.col_end = col_end
        self.dose_unit_conversion_dic =dose_unit_conversion_dic
        self.dose_unit_expressions, self.dose_unit_replacements = self._dose_unit_expressions()
        self.schema = {
            'start': pl.Datetime,
            'end': pl.Datetime,
            'dose': pl.Float32,
            'dose_unit': pl.String,
            'route': pl.String
            }
        
        col_mapping = {
                        self.col_start: 'start',
                        self.col_end : 'end',
                        self.col_dose: 'dose',
                        self.col_dose_unit : 'dose_unit',
                        self.col_route : 'route'
                        }
        self.cols_to_create = [v for k, v in col_mapping.items() if k is None]
        self.rename_dic = {k: v for k, v in col_mapping.items() if k is not None}
        
    
    def _dose_unit_expressions(self):
        exprs = []
        if self.dose_unit_conversion_dic is None:
            return exprs, {}
        
        unit_replacements = {old_label: v['omop_code'] for old_label, v in self.dose_unit_conversion_dic.items()}
        
        for source_unit, dic in self.dose_unit_conversion_dic.items():
            expr = [
                    (pl.when(pl.col('dose_unit').eq(source_unit))
                    .then(pl.col('dose').mul(dic['mul']))
                    .otherwise(pl.col('dose'))),
                    ]
            
            exprs.append(expr)
        
        return exprs, unit_replacements
        
    
    def _compute_offset(self, lf):
        """
        Some databases are already "offset" based: admission is taken as the 
        origin of times for each stay. 
        This functions is used to convert all databases to this format.
        
        """
        
        lf = (lf 
              .with_columns(
                  (pl.col('start') - pl.col(self.col_admittime)).alias('start'),
                  (pl.col('end') - pl.col(self.col_admittime)).alias('end')
                  )
              .drop(self.col_admittime)      
              )

        return lf

    def _convert_offset(self, lf):
        convertdict = {
                'milisecond': 1e-3,
                'second': 1,
                'minute':60,
                'hour': 24*60,
                'day': 24*60*60
            }
        
        lf = lf.with_columns(
            pl.duration(seconds=pl.col('start').mul(convertdict[self.unit_offset])).alias('start'),
            pl.duration(seconds=pl.col('end').mul(convertdict[self.unit_offset])).alias('end'),
            )
        return lf
    
    
    def _get_offset(self, lf):
        """
        This function only calls the compute_offset_pl function if needed
        """
        lf = lf.with_columns(pl.col('end').fill_null(pl.col('start')))
        if self.offset_calc:
            return self._compute_offset(lf)
        else:
            return self._convert_offset(lf)



    def _second_conversion_constant(self, unit):
        d = {
            'milisecond': 0.001,
            'second': 1,
            'minute': 60,
            'hour': 3600,
            'day': 3600*24,
        }
        try:
            return d[unit]
        except KeyError:
            raise ValueError(
                f'unit should be one of {[*d.keys()]}, not {unit}')

    def _filter_times(self, lf):
        """
        Stop tracking drugs after discharge and ignore drug admission prior
        to self.preadm_anteriority. This includes past icu stays of the same 
        patients.
        These parameters are necessary when resampling to hourly data, as we 
        don't want to resample the time in between admission when it is very 
        long.
        """
        lf = (lf
              .filter(pl.col('start') < pl.col(self.col_los),
                      pl.col('start') <= pl.col('end'),
                      pl.col('start') > -pl.duration(days=self.preadm_anteriority))
              .drop(self.col_los)
              )
    
        return lf


    def _load_med_mapping(self):
        mapping = ({v: key for v in val[self.dataset]}
                   for key, val in self.ohdsi_med.items())
        return reduce(operator.ior, mapping)

    def _map_ingredients(self, lf):
        """
        Retrieve ingredients based on the drug mapping.
        """
        
        lf = (lf
              .rename({self.col_med: 'original_drugname'})
              .with_columns(
                  pl.col('original_drugname').replace(self.med_mapping, default=None).alias('label')
                  )
              )
        return lf

    def _add_missing_cols(self, lf):
        lf = lf.with_columns(
                (pl.lit(None).cast(self.schema[col]).alias(col) for col in self.cols_to_create)
            )
        
        return lf

    def _unit_conversion(self, lf):
        
        for expr in self.dose_unit_expressions:
            lf = lf.with_columns(expr)
        lf = lf.with_columns(pl.col('dose_unit').replace(self.dose_unit_replacements))
        
        return lf
            
    def _map_routes(self, lf):

        lf = lf.with_columns(
            pl.col('route').replace(self.mapping_drug_route, default=None).cast(pl.Int32).alias('route_concept_id')
            )
        return lf
    
    def run(self):
        med = (self.lf_med
               .pipe(self._add_missing_cols)
               .join(self.lf_labels, on=self.col_pid)
               .rename(self.rename_dic)
               .pipe(self._get_offset)
               .pipe(self._filter_times)
               .pipe(self._map_ingredients)
               .pipe(self._map_routes)
               .pipe(self._unit_conversion)
               )

        return med

        
