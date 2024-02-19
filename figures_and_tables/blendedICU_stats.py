import random
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from blended_preprocessing.timeseries import blendedicuTSP
from blended_preprocessing.omop_conversion import OMOP_converter


class Blendedicu_stats(blendedicuTSP):
    def __init__(self):
        super().__init__()
        random.seed(0)
        self.plot_savepath = f'{self.savepath}/plots/'
        Path(self.plot_savepath).mkdir(exist_ok=True)
        self.labels_pth = f'{self.savepath}/preprocessed_labels.parquet'

        self.colors = {'amsterdam': 'tab:blue',
                       'eicu': 'tab:orange',
                       'mimic4': 'tab:green',
                       'mimic3': 'tab:purple',
                       'hirid': 'tab:red'}

        self.raw_ts_pths = self.get_ts_pths(self.formatted_ts_dir, sample=1000)
        self.med_pths = self.get_ts_pths(self.formatted_med_dir, sample=1000)

        self.omop = OMOP_converter(ts_pths=self.raw_ts_pths,
                                   med_pths=self.med_pths)


    def flat_stats(self):
        self.labels = pd.read_parquet(self.labels_pth)
        self.dataset_mapper = self.labels.source_dataset
        self.labels = self.labels.replace({'sex': {1: 'Male',
                                                   0: 'Female',
                                                   0.5: 'Unknown/Other'}})

        self.group = self.labels.reset_index().groupby('source_dataset')
        self.mortality = self.group.mortality.mean()
        self.los = self.group.true_lengthofstay.median()
        self.los_std = self.group.true_lengthofstay.std()
        self.n_caresites = self.group.care_site.nunique()
        self.n_stays = self.group.patient.nunique()
        self.n_patients = self.group.original_uniquepid.nunique()
        self.age = self.group.raw_age.median()
        self.age_std = self.group.raw_age.std()
        self.height = self.group.raw_height.median()
        self.height_std = self.group.raw_height.std()
        self.weight = self.group.raw_weight.median()
        self.weight_std = self.group.raw_weight.std()

        self.sex = self.group.sex.value_counts()/self.group.sex.count()

        self.unit_type = (self.group.unit_type.value_counts()
                          / self.group.unit_type.count())
        self.origin = (self.group.origin.value_counts()
                       / self.group.origin.count())
        self.discharge_loc = (self.group.discharge_location.value_counts()
                              / self.group.discharge_location.count())

        self.discharge_groups = self.labels.groupby(['source_dataset',
                                                     'discharge_location'])
        self.sex_groups = self.labels.groupby(['source_dataset',
                                               'sex'])
        self.admission_groups = self.labels.groupby(['source_dataset',
                                                     'origin'])
        self.unittype_groups = self.labels.groupby(['source_dataset',
                                                    'unit_type'])

        self.sex_los = self.sex_groups.true_lengthofstay.median()
        self.sex_los_std = self.sex_groups.true_lengthofstay.std()

        discharge_los = self.discharge_groups.true_lengthofstay
        admission_los = self.admission_groups.true_lengthofstay
        unittype_los = self.unittype_groups.true_lengthofstay
        self.dischargeloc_los = discharge_los.median()
        self.dischargeloc_los_std = discharge_los.std()
        self.admissions_los = admission_los.median()
        self.admissions_los_std = admission_los.std()
        self.unittype_los = unittype_los.median()
        self.unittype_los_std = unittype_los.std()
        self.initial_patients = pd.Series({
            'amsterdam': '20109',
            'eicu': '139367',
            'hirid': '33932',
            'mimic4': '50048',
            'mimic3': '.'})
        self.initial_icu_stays = pd.Series({
            'amsterdam': '23106',
            'eicu': '200859',
            'hirid': r'33932\textsuperscript{1}',
            'mimic4': '69619',
            'mimic3': '57874'})

        self.n_variables, self.n_variables_std = self._n_variables()

        self.numeric_flats = self._numeric_flats()

        self.sex_table = self._flat_cat_table(self.sex)
        self.unit_type_table = self._flat_cat_table(self.unit_type)
        self.origin_table = self._flat_cat_table(self.origin)
        self.discharge_loc_table = self._flat_cat_table(self.discharge_loc)

        self.sex_los_table = (self.sex_los.apply(lambda x: f'{x:.2f}').unstack(level=1).fillna('-')
                              +self.sex_los_std.apply(lambda x: f' ({x:.2f})').unstack(level=1).fillna('')).transpose()
    
        self.unit_type_los_table = (self.unittype_los.apply(lambda x: f'{x:.2f}').unstack(level=1).fillna('-')
                          +self.unittype_los_std.apply(lambda x: f' ({x:.2f})').unstack(level=1).fillna('')).transpose()

        self.categorical_los = pd.concat({
            'sex': self.sex_los,
            'unit_type': self.unittype_los,
            'admission_origin': self.admissions_los,
            'discharge_location': self.dischargeloc_los,
        }).unstack(level=1)

        self.categorical_los_std = pd.concat({
            'sex': self.sex_los_std,
            'unit_type': self.unittype_los_std,
            'admission_origin': self.admissions_los_std,
            'discharge_location': self.dischargeloc_los_std,
        }).unstack(level=1)

    def _flat_cat_table(self, df):
        return (df.unstack(level=1)
                  .transpose()
                  .multiply(100).round(2)
                  .replace('nan', '-'))

    def _format_mean_std(self, dfmean, dfstd, rounding=1):
        mean_str = dfmean.round(rounding).astype(str)
        std_str = dfstd.round(rounding).astype(str)
        return mean_str+' (' + std_str+')'

    def _numeric_flats(self):
        age = self._format_mean_std(self.age, self.age_std)
        height = self._format_mean_std(self.height, self.height_std)
        weight = self._format_mean_std(self.weight, self.weight_std)
        los = self._format_mean_std(self.los, self.los_std)
        compl = self._format_mean_std(self.n_variables, self.n_variables_std)
        mortality = self.mortality.multiply(100).round(1)

        return (pd.concat({
            'Initial patients': self.initial_patients,
            'Initial ICU stays': self.initial_icu_stays,
            'Final patients': self.n_patients.round(0).astype(str),
            'Final ICU stays': self.n_stays.round(0).astype(str),
            'Number of care sites': self.n_caresites.round(0).astype(str),
            'Age (Years)': age,
            'Height (cm)': height,
            'Weight (kg)': weight,
            'Median length of stay (Days)': los,
            r'Mortality (\%)': mortality,
            r'Completeness\textsuperscript{2} (\%)': compl
        }).unstack(level=1))

    def _n_variables(self):
        self.raw_timeseries = (self.load(self.raw_ts_pths.ts_pth.to_list(),
                                         verbose=False)
                               .drop(columns='ventilator_mode'))

        self.n_variables_patients = ((self.raw_timeseries.drop(columns='time')
                                     .groupby(level=0).sum() > 0).sum(axis=1)
                                     .rename('n_variables')
                                     .multiply(100/len(self.kept_ts)))

        self.n_variables_groups = (self.labels[['source_dataset']]
                                       .join(self.n_variables_patients)
                                       .groupby('source_dataset'))

        self.n_variables = self.n_variables_groups.n_variables.mean()
        self.n_variables_std = self.n_variables_groups.n_variables.std()
        return self.n_variables, self.n_variables_std
    
    def medication_inclusion_stats(self):
        
        ohdsi_med_df = (pd.DataFrame(self.ohdsi_med)
                        .drop('blended')  
                        .map(lambda x: len(x)))
        self.n_drugs = ohdsi_med_df.shape[1]
        self.n_labels = ohdsi_med_df.sum(1)
        print('\nDrug exposure inclusion stats:')
        print(f'{self.n_drugs} drugs included')
        print('\nNumber of labels:')
        print(f'{self.n_labels}')

    def ts_frequencies(self):
        print('Computing timeseries frequencies...')
        ts_freq, ts_freq_std = {}, {}
        for dataset, pths in self.raw_ts_pths.groupby('source_dataset'):
            print(f'  -> {dataset}')
            ts = (self.load(pths.ts_pth.to_list(), verbose=False)
                  .drop(columns='ventilator_mode', errors='ignore')
                  .set_index('time', append=True)
                  .groupby(level=[0, 1]).mean().droplevel(1)
                  .groupby(level=0).count()
                  .join(self.labels['true_lengthofstay']))

            freqs = (ts.drop(columns='true_lengthofstay')
                       .div(ts.true_lengthofstay, axis=0))

            ts_freq[dataset] = freqs.mean()
            ts_freq_std[dataset] = freqs.std()

        categories = self.ts_variables.set_index('blended').categories

        self.all_ts_freq = pd.DataFrame(ts_freq).dropna()

        self.ts_freq = (self.all_ts_freq.join(categories)
                                        .dropna(subset='categories')
                                        .groupby('categories').sum())

        self.ts_freq_std = (self.all_ts_freq.join(categories)
                                            .dropna(subset='categories')
                                            .groupby('categories').std())

        self.ts_freq.loc['Total'] = self.ts_freq.sum()
        
        std_sum = self.ts_freq_std.apply(lambda x: np.sqrt((x**2).sum(axis=0)))
        self.ts_freq_std.loc['Total'] = std_sum
        self.printable_ts_freq = (self.ts_freq.round(1).astype(str)
                                  + ' ('
                                  + self.ts_freq_std.round(1).fillna('-').astype(str)
                                  + ')')
        return self.ts_freq

    def med_frequencies(self):
        print('Computing drug exposures...')
        self.med_freq = {}
        for dataset, pths in self.med_pths.groupby('source_dataset'):
            print(f'  -> {dataset}')
            self.meds = (self.load(pths.ts_pth.to_list(),
                         columns=['variable'],
                         verbose=False)
                         .reset_index()
                         .drop_duplicates())

            self.med_freq[dataset] = (self.meds.variable.value_counts()
                                      / self.meds.patient.nunique())

        self.med_freq = ((pd.concat(self.med_freq).unstack(level=0)*100)
                         .round(1)
                         .astype(str)
                         .replace({'nan': '-',
                                   '0.0': '$<$0.01'}))

        med_order = (self.med_freq.apply(pd.to_numeric, errors='coerce')
                                  .fillna(0)
                                  .mean(axis=1)
                                  .sort_values(ascending=False)
                                  .index)
        self.med_freq = self.med_freq.loc[med_order, sorted(self.datasets)]
        self.med_freq = self.med_freq.drop(['Kcl',
                                            'NaCL',
                                            'dextrose',
                                            'gelofusine'])
        return self.med_freq

    def _remove_outliers_from_plot(self, df, var):
        lower_thr = df[var].quantile(0.01)
        upper_thr = df[var].quantile(0.99)
        idx = (df[var] < lower_thr) | (df[var] > upper_thr)
        df.loc[idx, var] = np.nan
        return df

    def ts_kde(self):
        ncols=5
        numeric_ts = [v for v in self.numeric_ts if v not in ['time', 'hour']]
        self.fig, axs = plt.subplots(ncols=ncols,
                                     nrows=1+len(numeric_ts)//ncols, 
                                     figsize=(15,25))
        self.axs = axs.flatten()

        handles = [Patch(color=c, label=label, alpha=0.5) for label, c in self.colors.items()]

        self.axs[0].legend(handles=handles,
                           prop={'size':16},
                           loc='lower left',
                           frameon=False)
        
        self.axs[0].set_axis_off()
        for i, (ax, var) in enumerate(zip(self.axs[1:], numeric_ts)):
            ts_list = []
            print(var)
            unit_name = self.omop.units.loc[var].values[0]
            title = '\n'.join(textwrap.wrap(self.omop_mapping[var], 30))
            for dataset, ts_pths in self.raw_ts_pths.groupby('source_dataset'):
                if self.is_measured.loc[var, dataset]:
                    ts_list.append(self.load(ts_pths.ts_pth.tolist(),
                                             verbose=False,
                                             columns=[var]))
            
            self.ts_list = ts_list
            
            self.raw_timeseries = (pd.concat(ts_list)
                                       .join(self.labels[['source_dataset']], 
                                                           how='inner')
                                       .reset_index()
                                       .pipe(self._remove_outliers_from_plot,
                                             var=var))

            groups = self.raw_timeseries.groupby('source_dataset')

            for dataset, group in groups:

                label = dataset if i == 0 else '__nolabel'
                self.val = group[var]
                ax = sns.kdeplot(group[var],
                                 fill=True,
                                 ax=ax,
                                 label=label,
                                 color=self.colors[dataset])

                ax.set_title(title, fontsize=13)
                ax.set_xlabel(unit_name)

            ax.set_xlim([self.raw_timeseries[var].quantile(0.01),
                         self.raw_timeseries[var].quantile(0.99)])

        idx_axis_off = range(len(numeric_ts)+1, len(self.axs))
        [self.axs[i].set_axis_off() for i in idx_axis_off]
        self.fig.tight_layout()
        self.fig.savefig(self.plot_savepath+'kdeplot.png', dpi=800)
