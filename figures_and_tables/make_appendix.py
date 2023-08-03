from functools import reduce
import operator

import pandas as pd

from blended_preprocessing.flat_and_labels import blended_FLProcessor

flp = blended_FLProcessor(size=None, datasets=['hirid',
                                               'amsterdam',
                                               'mimic',
                                               'eicu'])


def get_appendix_cat(flp=flp):
    unit_types = pd.DataFrame.from_dict(flp.unit_types, orient='index')
    discharge_locations = pd.DataFrame.from_dict(flp.discharge_locations,
                                                 orient='index')
    admission_origins = pd.DataFrame.from_dict(flp.admission_origins,
                                               orient='index')
    return (pd.concat({'unit_types': unit_types,
                       'discharge_locations': discharge_locations,
                       'admission_origins': admission_origins})
            .rename_axis(['variable', 'source_value'])
            .rename(columns={0: 'category'})
            .reset_index()
            .set_index(['variable', 'category'])
            .sort_index())


def get_appendix_med(flp=flp):
    mappings = {d: [{v: key for v in val[d]}
                    for key, val in flp.ohdsi_med.items()]
                for d in ['amsterdam', 'hirid', 'eicu', 'mimic']}

    dic = {d: reduce(operator.ior, ma) for d, ma in mappings.items()}

    D = {d: pd.DataFrame.from_dict(v, orient='index') for d, v in dic.items()}
    return (pd.concat(D)
            .rename_axis(['dataset', 'source_value'])
            .rename(columns={0: 'medication'})
            .reset_index()
            .set_index(['medication', 'dataset'])
            .sort_index())


def latexify_appendix_med(appendix_med):
    s = ''
    for med, gp_med in appendix_med.groupby(level=0):
        s += fr'{med.capitalize()} &&\\\midrule \midrule'
        gped_dataset = gp_med.droplevel(0).groupby(level=0)
        for i, (dataset, gp_dataset) in enumerate(gped_dataset):
            if i != 0:
                s += r'\cmidrule{2-3}'
            s += '\n'+fr'& {dataset}'
            for i, v in enumerate(gp_dataset.source_value.values):
                v = v.strip().replace('#', '\#')
                if i == 0:
                    s += fr' & {v} \\'
                else:
                    s += '\n'+fr' && {v} \\'
            
        s += r'\midrule'+'\n'
    print(s)
    return s


def get_appendix_ts(flp=flp):
    df = pd.read_csv(f'{flp.aux_pth}/timeseries_variables.csv',
                     sep=';',
                     usecols=flp.datasets,
                     index_col='hirid')
    df = (df.rename_axis('OMOP standard vocabulary'))
    return df
