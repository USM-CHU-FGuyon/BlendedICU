import operator
from functools import reduce

import polars as pl
import numpy as np

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
                 unit_los,
                 col_dose,
                 col_dose_unit,
                 col_route,
                 col_end,
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
        self.unit_los = unit_los
        self.unit_offset = unit_offset
        self.col_dose = col_dose
        self.col_dose_unit = col_dose_unit
        self.col_route = col_route
        self.col_end = col_end
        self.unit_list = ['grams', 'mg', 'mcg']
        self.gram_dict = {
            'grams' : 
                {'grams' : 1, 'mg' : 1e-3, 'mcg' : 1e-6}
                ,
            'mg' : 
                {'grams' : 1e3, 'mg' : 1, 'mcg' : 1e-3}
                ,
            'mcg' : 
                {'grams' : 1e6, 'mg' : 1e3, 'mcg' : 1}
            }
        
    
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
                'miliseconds': 1e-3,
                'seconds': 1,
                'minutes':60,
                'hour': 24*60,
                'day': 24*60*60
            }
        
        lf = lf.with_columns(
            pl.duration(seconds=pl.col('start').mul(convertdict[self.unit_offset])),
            pl.duration(seconds=pl.col('end').mul(convertdict[self.unit_offset])),
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


    def _rename(self, lf):
        """
        Rename the columns with more conventional names
        """
        lf = lf.rename({self.col_start: 'start',
                        self.col_end : 'end',
                        self.col_dose: 'dose',
                        self.col_dose_unit : 'dose_unit',
                        self.col_route : 'route'
                        })
        return lf

    def _main_unit(self, lf):
        """
        This function iterates through the labels 
        and say if needed,what is the main unit to use for conversion.
        It returns None if no conversion is needed.
        """
        
        # replace the 'gm' unit name by 'grams'
        lf = lf.with_columns(
            pl.col('dose_unit').str.replace(r'gm', 'grams')
            )
        
        # Create a dictionnary {label : main unit used among grams, mg and mcg}
        label_df = (lf.select('label', 'dose_unit').collect(streaming = True).to_pandas())
        #self.l = label_df
        #1/0
        
        label_pivot = (label_df.pivot_table(index='label',
                                            columns='dose_unit',
                                            aggfunc='size',
                                            fill_value=0)
                       [['grams', 'mg', 'mcg']])
        main_unit_series = label_pivot.apply(lambda row: label_pivot.columns[np.argmax(row)], axis=1)
        main_unit_dict = main_unit_series.to_dict()
        
        # generate 3 temporary columns used to modify the units converted
        lf = lf.with_columns([
            pl.lit(unit).alias(unit) for unit in self.unit_list]
            )
        
        # convert the values into the main unit by iterating through labels and units 
        for label, main_unit in main_unit_dict.items():
            for source_unit in self.unit_list :
                lf = lf.with_columns(
                    pl.when(
                        (pl.col('label') == label)
                        & (pl.col('dose_unit') == source_unit)
                        )
                    .then(pl.col('dose').mul(self.gram_dict[main_unit][source_unit]))
                    .otherwise(pl.col('dose'))
                    )
        # change the dose_unit colmun to match the modification made on the dose column
                lf = lf.with_columns(
                    pl.when((pl.col('label') == label) & (pl.col('dose_unit') == source_unit))
                    .then(pl.col(main_unit))
                    .otherwise(pl.col('dose_unit')).alias('dose_unit')
                    )
        # get rid of the 3 temporary columns
        lf = lf.drop(self.unit_list)

        return lf
            


    def run(self):
        med = (self.lf_med
               .join(self.lf_labels, on=self.col_pid)
               .pipe(self._rename)
               .pipe(self._get_offset)
               .pipe(self._filter_times)
               .pipe(self._map_ingredients)
               #.pipe(self._main_unit) #TODO : non functional yet !
               )
        return med
        
