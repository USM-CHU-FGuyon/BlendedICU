import json

import pandas as pd


class meds(object):
    def __init__(self, pth_dic, name='omop'):
        self.pth_dic = pth_dic
        self.aux_pth = pth_dic['auxillary_files']
        self.user_input_pth = pth_dic['user_input']
        self.med_mapping_pth = pth_dic['medication_mapping_files']
        self.source_name = name + '_source_path'
        if self.source_name in pth_dic:
            self.datadir = pth_dic[self.source_name]
        self.name = name
        self.savepath = f'{self.med_mapping_pth}{self.name}_medications.csv'

    def save(self, meds):
        print(f'   ---> Saving {self.savepath}')
        meds.to_csv(self.savepath, sep=';', index=None)


class Eicu_meds(meds):
    def __init__(self, pth_dic):
        super().__init__(pth_dic, name='eicu')

    def get(self):
        print('Loading eICU medications...')
        patients_pth = self.datadir+'patient.csv.gz'
        eicu_admissiondrug_pth = self.datadir+'admissionDrug.csv.gz'
        eicu_infusiondrug_pth = self.datadir+'infusionDrug.csv.gz'
        eicu_medication_pth = self.datadir+'medication.csv.gz'

        eicu_infusion = pd.read_csv(eicu_infusiondrug_pth)
        eicu_admission = pd.read_csv(eicu_admissiondrug_pth)
        eicu_med = pd.read_csv(eicu_medication_pth)

        patients = pd.read_csv(patients_pth,
                               usecols=['patientunitstayid'])

        df_infusion = (patients.merge(eicu_infusion[['patientunitstayid',
                                                     'drugname']]
                                      .drop_duplicates(),
                                      on='patientunitstayid', how='outer')
                       .drugname.value_counts())

        df_admission = (patients.merge(eicu_admission[['patientunitstayid',
                                                       'drugname']]
                                       .drop_duplicates(),
                                       on='patientunitstayid', how='outer')
                        .drugname.value_counts())

        df_med = (patients.merge(eicu_med[['patientunitstayid',
                                           'drugname']]
                                 .drop_duplicates(),
                                 on='patientunitstayid', how='outer')
                  .drugname.value_counts())

        eicu_meds = (pd.concat([df_med, df_infusion, df_admission])
                     .sort_values(ascending=False)/len(patients))

        eicu_meds = (pd.DataFrame(eicu_meds)
                     .reset_index()
                     .rename(columns={'drugname': 'drugcount',
                                      'index': 'drugname'}))

        self.save(eicu_meds)
        return eicu_meds


class Hirid_meds(meds):
    def __init__(self, pth):
        super().__init__(pth, name='hirid')

    def _load_hirid_patient(self):
        hirid_patient_pth = self.datadir+'reference_data.tar.gz'
        hirid_patient = (pd.read_csv(hirid_patient_pth,
                                     usecols=['general_table.csv'])
                         .rename(columns={'general_table.csv': 'patient'}))
        hirid_patient['patient'] = pd.to_numeric(hirid_patient.patient,
                                                 errors='coerce')
        return (hirid_patient.dropna()
                             .astype({'patient': int})
                             .drop_duplicates())

    def _load_hirid_pharmanames(self):
        hirid_pharmaname_pth = self.datadir+'hirid_variable_reference_v1.csv'
        hirid_pharmaname = pd.read_csv(hirid_pharmaname_pth,
                                       sep=';',
                                       encoding='unicode_escape',
                                       usecols=['Source Table',
                                                'ID',
                                                'Variable Name'])
        hirid_pharmaname = (hirid_pharmaname
                            .astype({'ID': int})
                            .rename(columns={'ID': 'pharmaid',
                                             'Variable Name': 'drugname'
                                             })
                            .loc[hirid_pharmaname['Source Table'] == 'Pharma'])
        return hirid_pharmaname

    def _load_hirid_pharma(self):
        hirid_pharma_pth = self.datadir+'raw_stage/pharma_records_csv.tar.gz'
        hirid_pharma = (pd.read_csv(hirid_pharma_pth,
                                    usecols=['pharma_records/', 'pharmaid'])
                        .rename(columns={'pharma_records/': 'patient'})
                        .dropna()
                        .drop_duplicates()
                        .astype({'pharmaid': int,
                                 'patient': int}))
        return hirid_pharma

    def get(self):
        print('Loading HiRID medications...')
        self.pharmanames = self._load_hirid_pharmanames()
        self.pharma = self._load_hirid_pharma()
        self.hirid_patient = self._load_hirid_patient()
        n_patients = self.hirid_patient.patient.nunique()

        hirid_meds = (self.pharma
                          .merge(self.pharmanames,
                                 on='pharmaid',
                                 how='inner')
                          .drop(columns='Source Table')
                          .drop_duplicates()
                          .merge(self.hirid_patient,
                                 on='patient',
                                 how='outer')
                          .drugname.value_counts()/n_patients)

        hirid_meds = (pd.DataFrame(hirid_meds)
                      .reset_index()
                      .rename(columns={'drugname': 'drugcount',
                                       'index': 'drugname'}))

        self.save(hirid_meds)
        return hirid_meds


class Amsterdam_meds(meds):
    def __init__(self, pth):
        super().__init__(pth, name='amsterdam')

    def get(self):
        print('Loading Amsterdam medications...')
        pth_drugs = self.datadir+'drugitems.csv.gz'
        pth_patients = self.datadir+'admissions.csv.gz'

        admissions = pd.read_csv(pth_patients,
                                 compression='gzip',
                                 encoding='ISO-8859-1',
                                 usecols=['admissionid'])

        n_patients = admissions.admissionid.nunique()
        drugs = pd.read_csv(pth_drugs,
                            compression='gzip',
                            encoding='ISO-8859-1',
                            usecols=['admissionid', 'item'])

        drugs = drugs.drop_duplicates()

        amsterdam_meds = (drugs.merge(admissions,
                                      on='admissionid',
                                      how='outer')
                          .item
                          .value_counts()/n_patients)

        amsterdam_meds = (pd.DataFrame(amsterdam_meds)
                          .reset_index()
                          .rename(columns={'drugname': 'drugcount',
                                           'index': 'drugname'}))

        self.save(amsterdam_meds)
        return amsterdam_meds


class Mimic_meds(meds):
    def __init__(self, pth):
        super().__init__(pth, name='mimic')

    def get(self):
        print('Loading MIMIC medications...')
        pth_meds = self.datadir+'icu/inputevents.csv.gz'
        pth_d_items = self.datadir+'icu/d_items.csv.gz'
        pth_patients = self.datadir+'icu/icustays.csv.gz'

        patients = pd.read_csv(pth_patients, usecols=['stay_id'])
        meds = pd.read_csv(pth_meds, usecols=['stay_id', 'itemid'])
        d_items = pd.read_csv(pth_d_items, usecols=['label',
                                                    'itemid',
                                                    'category'])

        n_patients = patients.stay_id.nunique()

        df = (meds.drop_duplicates()
                  .merge(d_items, on='itemid')
                  .merge(patients, on='stay_id', how='outer'))

        mimic_medications = (df.loc[df.category == 'Medications', 'label']
                               .value_counts()/n_patients)
        mimic_antibio = (df.loc[df.category == 'Antibiotics', 'label']
                           .value_counts()/n_patients)
        mimic_fluids = (df.loc[df.category == 'Fluids/Intake', 'label']
                          .value_counts()/n_patients)
        mimic_col = (df.loc[df.category == 'Blood Products/Colloids', 'label']
                     .value_counts()/n_patients)

        mimic_meds = (pd.concat([mimic_medications,
                                 mimic_antibio,
                                 mimic_fluids,
                                 mimic_col])
                      .sort_values(ascending=False))
        self.save(mimic_meds)
        return (pd.DataFrame(mimic_meds)
                .reset_index()
                .rename(columns={'drugname': 'drugcount',
                                 'index': 'drugname'}))


class MedicationMapping(meds):
    def __init__(self, pth_dic):
        super().__init__(pth_dic)
        pth_ohdsi = self.med_mapping_pth+'ohdsi_icu_medications.csv'
        pth_manual = self.user_input_pth+'manual_icu_meds.csv'
        self.ohdsi = pd.read_csv(pth_ohdsi, sep=';')
        self.manual_addings = pd.read_csv(pth_manual, sep=';')
        self.drug_mapping = pd.concat([self.ohdsi, self.manual_addings])

    def _get_drugnames(self):
        eicu_meds = Eicu_meds(self.pth_dic).get()
        mimic_meds = Mimic_meds(self.pth_dic).get()
        hirid_meds = Hirid_meds(self.pth_dic).get()
        amsterdam_meds = Amsterdam_meds(self.pth_dic).get()

        self.eicu_meds = eicu_meds

        eicu_meds['dataset'] = 'eicu'
        mimic_meds['dataset'] = 'mimic'
        hirid_meds['dataset'] = 'hirid'
        amsterdam_meds['dataset'] = 'amsterdam'

        df = pd.concat([eicu_meds, mimic_meds, hirid_meds, amsterdam_meds])

        df.to_parquet(self.med_mapping_pth+'drugnames.parquet')
        return df

    def run(self, load_drugnames=True, fname='medications.json'):

        self.med_cids = pd.read_parquet(self.med_mapping_pth+'med_concept_ids.parquet')

        self.drugs = (pd.read_parquet(self.med_mapping_pth+'drugnames.parquet')
                      if load_drugnames
                      else self._get_drugnames())

        self.drugs['drugname_lkup'] = (' '
                                       + self.drugs.drugname
                                             .str.replace(r'[/,\-,.,(,),:]',
                                                          ' ',
                                                          regex=True)
                                       + ' ')
        medications_json = {}
        for name, aliases in self.drug_mapping.items():
            print(name)
            print(f'  -> Looking for {len(aliases.dropna())} aliases...')
            df_al = []

            medications_json[name] = {'blended': [],
                                      'amsterdam': [],
                                      'eicu': [],
                                      'hirid': [],
                                      'mimic': []}
            aliases = (pd.concat([pd.Series([name]), aliases])
                         .dropna().drop_duplicates())
            aliases = ' ' + aliases + ' '
            for alias in aliases:
                keep_idx = self.drugs.drugname_lkup.str.contains(alias,
                                                                 case=False,
                                                                 regex=False)
                df_al.append(self.drugs.loc[keep_idx, ['drugname', 'dataset']])

            df_al = pd.concat(df_al).drop_duplicates()

            for d in medications_json[name]:
                medications_json[name][d] = df_al.loc[df_al.dataset == d,
                                                      'drugname'].to_list()
            try:
                self.concept_id = int(self.med_cids.loc[name, 'concept_id'])
            except KeyError:
                self.concept_id = 0
            medications_json[name]['blended'] = self.concept_id
        json.dump(medications_json,
                  open(self.aux_pth+fname, 'w'),
                  indent=4,
                  ensure_ascii=False)

        self.medication_json = medications_json
        return medications_json
