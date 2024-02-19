import json
import tarfile
from pathlib import Path

import pandas as pd


class meds:
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
        print('\n\neICU\n===========\n')
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
                     .reset_index())
        self.save(eicu_meds)
        eicu_meds['dataset'] = self.name
        return eicu_meds


class Hirid_meds(meds):
    def __init__(self, pth):
        super().__init__(pth, name='hirid')
        self._admissions_untar_path = self.datadir + '/reference_data/'
        self.admissions_path = self._admissions_untar_path + 'general_table.csv'
        self.imputedstage_path = self.datadir + 'imputed_stage/parquet/'
        self.ts_path = self.datadir + 'observation_tables/parquet/'
        self.med_path = self.datadir + 'pharma_records/parquet/'
        
        self._untar_files()
        
    def _untar_files(self):
        """
        Uncompresses source files. Only proceeds if uncompressed files are not 
        found.
        """
        ts_tar_path = self.datadir + 'raw_stage/observation_tables_parquet.tar.gz'
        admissions_tar_path = self.datadir + 'reference_data.tar.gz'
        pharma_tar_path = self.datadir + 'raw_stage/pharma_records_parquet.tar.gz'
        imputedstage_tar_path = self.datadir + 'imputed_stage/imputed_stage_parquet.tar.gz'
        
        files = {
            self.admissions_path: (admissions_tar_path, self._admissions_untar_path),
            self.imputedstage_path: (imputedstage_tar_path, self.datadir),
            self.ts_path: (ts_tar_path, self.datadir),
            self.med_path: (pharma_tar_path, self.datadir)
            }

        files_to_untar = {pth: args for pth, args in files.items() if not Path(pth).exists()}
        if files_to_untar:
            if input(f'Untar {list(files_to_untar.keys())} ? y/[n]')=='y': 
                for args in files_to_untar.values():
                    self._untar(*args)
        else:
            print('Hirid files already untarred.')
    
    def _untar(self, src, tgt):
        print(f'Untarring \n   {src} \n   into \n   {tgt}\n   This will only'
              ' be done once')
        tar = tarfile.open(src)
        tar.extractall(path=tgt)
        tar.close()
        print('  -> Done.')

    def _load_hirid_patient(self):
        hirid_patient = pd.read_csv(self.admissions_path, usecols=['patientid'])
        hirid_patient = hirid_patient.rename(columns={'patientid': 'patient'})
        return hirid_patient

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
                                             'Variable Name': 'drugname'})
                            .loc[hirid_pharmaname['Source Table'] == 'Pharma'])
        return hirid_pharmaname

    def _load_hirid_pharma(self):
        
        hirid_pharma = pd.read_parquet(self.med_path,
                                       columns=['patientid', 'pharmaid'])
        
        hirid_pharma = (hirid_pharma
                        .rename(columns={'patientid': 'patient'})
                        .drop_duplicates())
        return hirid_pharma

    def get(self):
        print('\n\nHiRID\n===========\n')
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

        hirid_meds = pd.DataFrame(hirid_meds).reset_index()
        self.save(hirid_meds)
        hirid_meds['dataset'] = self.name
        return hirid_meds


class Amsterdam_meds(meds):
    def __init__(self, pth):
        super().__init__(pth, name='amsterdam')

    def get(self):
        print('\n\nAmsterdam\n===========\n')
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
                          .value_counts()
                          .div(n_patients))

        amsterdam_meds = (pd.DataFrame(amsterdam_meds)
                          .reset_index()
                          .rename(columns={'item': 'drugname'}))

        self.save(amsterdam_meds)
        amsterdam_meds['dataset'] = self.name
        return amsterdam_meds


class Mimic4_meds(meds):
    def __init__(self, pth):
        super().__init__(pth, name='mimic4')

    def _load_inputevents(self):
        print('Loading inputevents')
        pth_meds = self.datadir+'icu/inputevents.csv.gz'
        meds = pd.read_csv(pth_meds, usecols=['stay_id', 'itemid'])
        return meds

    def get(self):
        print('\n\nMIMIC-IV\n===========\n')
        
        pth_d_items = self.datadir+'icu/d_items.csv.gz'
        pth_patients = self.datadir+'icu/icustays.csv.gz'
        
        meds = self._load_inputevents()
        patients = pd.read_csv(pth_patients, usecols=['stay_id'])
        d_items = pd.read_csv(pth_d_items, usecols=['label',
                                                    'itemid',
                                                    'category'])
        n_patients = patients.stay_id.nunique()
        df = (meds.drop_duplicates()
                  .merge(d_items, on='itemid')
                  .merge(patients, on='stay_id', how='outer'))
        
        mimic4_medications = (df.loc[df.category == 'Medications', 'label']
                               .value_counts()/n_patients)
        mimic4_antibio = (df.loc[df.category == 'Antibiotics', 'label']
                           .value_counts()/n_patients)
        mimic4_fluids = (df.loc[df.category == 'Fluids/Intake', 'label']
                          .value_counts()/n_patients)
        mimic4_col = (df.loc[df.category == 'Blood Products/Colloids', 'label']
                     .value_counts()/n_patients)
        
        mimic4_meds = (pd.concat([mimic4_medications,
                                  mimic4_antibio,
                                  mimic4_fluids,
                                  mimic4_col])
                      .sort_values(ascending=False))

        mimic4_meds = (pd.DataFrame(mimic4_meds)
                      .reset_index()
                      .rename(columns={'label': 'drugname'}))
        self.save(mimic4_meds)
        mimic4_meds['dataset'] = self.name
        return mimic4_meds
        

class Mimic3_meds(meds):
    def __init__(self, pth):
        super().__init__(pth, name='mimic3')
        
    def get(self):
        print('\n\nMIMIC-III\n===========\n')
        pth_meds_cv = self.datadir+'INPUTEVENTS_CV.csv.gz'
        pth_meds_mv = self.datadir+'INPUTEVENTS_MV.csv.gz'
        pth_d_items = self.datadir+'D_ITEMS.csv.gz'
        pth_patients = self.datadir+'ICUSTAYS.csv.gz'

        patients = pd.read_csv(pth_patients, usecols=['ICUSTAY_ID'])
        meds_cv = pd.read_csv(pth_meds_cv, usecols=['ICUSTAY_ID', 'ITEMID'])
        meds_mv = pd.read_csv(pth_meds_mv, usecols=['ICUSTAY_ID', 'ITEMID'])
        d_items = pd.read_csv(pth_d_items, usecols=['LABEL',
                                                    'ITEMID',
                                                    'CATEGORY'])
        n_patients = patients.ICUSTAY_ID.nunique()
        
        meds = pd.concat([meds_cv, meds_mv])
        
        df = (meds.drop_duplicates()
                  .merge(d_items, on='ITEMID'))

        mimic3_medications = (df.loc[df.CATEGORY=='Medications', 'LABEL']
                               .value_counts()
                               .div(n_patients))
        mimic3_antibio = (df.loc[df.CATEGORY=='Antibiotics', 'LABEL']
                           .value_counts()
                           .div(n_patients))
        mimic3_fluids = (df.loc[df.CATEGORY=='Fluids/Intake', 'LABEL']
                          .value_counts()
                          .div(n_patients))
        mimic3_col = (df.loc[df.CATEGORY=='Blood Products/Colloids', 'LABEL']
                     .value_counts()
                     .div(n_patients))

        mimic3_meds = (pd.concat([mimic3_medications,
                                  mimic3_antibio,
                                  mimic3_fluids,
                                  mimic3_col])
                      .sort_values(ascending=False))

        mimic3_meds = (pd.DataFrame(mimic3_meds)
                .reset_index()
                .rename(columns={'LABEL': 'drugname'}))
        self.save(mimic3_meds)
        mimic3_meds['dataset'] = self.name
        return mimic3_meds
        

class MedicationMapping(meds):
    def __init__(self,
                 pth_dic,
                 datasets=['amsterdam',
                           'hirid',
                           'eicu',
                           'mimic4',
                           'mimic3']):
        super().__init__(pth_dic)
        pth_ohdsi = self.med_mapping_pth+'ohdsi_icu_medications.csv'
        pth_manual = self.user_input_pth+'manual_icu_meds.csv'
        self.ohdsi = pd.read_csv(pth_ohdsi, sep=';')
        self.manual_addings = pd.read_csv(pth_manual, sep=';')
        self.drug_mapping = pd.concat([self.ohdsi, self.manual_addings])

        self.med_loaders = self._get_med_loaders(datasets)

    def _get_med_loaders(self, datasets):
        medloader_class = {
            'amsterdam': Amsterdam_meds,
            'hirid': Hirid_meds,
            'eicu': Eicu_meds,
            'mimic3': Mimic3_meds,
            'mimic4': Mimic4_meds
            }
        
        med_loaders = {}
        
        for dataset in datasets:
            med_loaders[dataset] = medloader_class[dataset](self.pth_dic)
        
        return med_loaders
        
    def _get_drugnames(self):
        meds = []        
        for loader in self.med_loaders.values():
            meds.append(loader.get())
            
        df = pd.concat(meds)
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
                                      'mimic3': [],
                                      'mimic4': []}
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
                medications_json[name][d] = df_al.loc[df_al.dataset==d,
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
