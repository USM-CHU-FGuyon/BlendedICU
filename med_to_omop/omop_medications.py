from pathlib import Path

import pandas as pd


ingredients = [
    'haloperidol', 'hypromellose', 'xylometazoline', 'simvastatin',
    'enoximone', 'captopril', 'tranexamic acid', 'amlodipine',
    'metoclopramide', 'erythromycin', 'levetiracetam',
    'methylprednisolone', 'clonidine', 'amoxicillin', 'dexamethasone',
    'felodipine', 'triamterene', 'desmopressin', 'spironolactone',
    'clonazepam', 'mirtazapine', 'nimodipine', 'thiopental',
    'vasopressin (USP)', 'ranitidine', 'glycopyrronium', 'neostigmine',
    'warfarin', 'ketorolac', 'lansoprazole', 'octreotide', 'phenytoin',
    'sulfamethoxazole', 'bisacodyl', 'atorvastatin', 'hydrocodone',
    'carvedilol', 'diphenhydramine', 'tramadol', 'aspirin',
    'atracurium', 'dobutamine', 'dopamine', 'fentanyl', 'flumazenil',
    'pancuronium', 'succinylcholine', 'vecuronium', 'rocuronium',
    'cisatracurium', 'sufentanil', 'propofol', 'milrinone',
    'midazolam', 'epinephrine', 'norepinephrine', 'morphine',
    'dexmedetomidine', 'furosemide', 'nitroglycerin', 'nicardipine',
    'hydrazaline', 'labelatol', 'diltiazem', 'esmolol', 'amiodarone',
    'nitroprusside', 'lidocaine', 'bumetanide', 'procainamide',
    'nesiritide', 'hydromorphone', 'lorazepam', 'ketamine',
    'methadone', 'pentobarbital', 'phenylephrine', 'angiotensin II',
    'treprostinil', 'epoprostenol', 'argatroban', 'eptifibatide',
    'bivalirudin', 'heparin', 'alteplase', 'tirofiban', 'lepirudin',
    'abciximab', 'vancomycin', 'cefazolin', 'cefepime', 'piperacillin',
    'metronidazole', 'ceftriaxone', 'ciprofloxacin', 'meropenem',
    'levofloxacin', 'azithromycin', 'ceftazidime', 'acyclovir',
    'ampicillin', 'fluconazole', 'clindamycin', 'linezolid',
    'micafungin', 'gentamicin']


class OMOP_Medications(object):
    def __init__(self, pth_dic, ingredients=ingredients):
        self.pth_to_voc = pth_dic['vocabulary']
        self.savedir = pth_dic['medication_mapping_files']
        self.savepath = self.savedir+'ohdsi_icu_medications.csv'
        print('Loading CONCEPT table...')
        self.concept = pd.read_parquet(self.pth_to_voc+'CONCEPT.parquet',
                                       columns=['concept_id', 'concept_name']
                                       )
        print('Loading CONCEPT_RELATIONSHIP table...')
        relationship_pth = self.pth_to_voc+'CONCEPT_RELATIONSHIP.parquet'
        self.relations = pd.read_parquet(relationship_pth,
                                         columns=['concept_id_1',
                                                  'concept_id_2',
                                                  'relationship_id'])

        self.ingredients = ingredients

        self._check_unique_ingredients()

    def _check_unique_ingredients(self):
        vcounts = pd.Series(self.ingredients).value_counts()
        if vcounts.max() > 1:
            raise ValueError(f'Duplicate names in ingredients : '
                             f'{vcounts.loc[vcounts>1].to_list()}')

    def _ensure_vocabulary_is_downloaded(self):
        if not Path(self.pth_to_voc).is_dir():
            raise ValueError('Please download the RxNorm and RxNorm Extended'
                             'vocabularies from '
                             'https://athena.ohdsi.org/vocabulary/list')

    def run(self):
        med_idx = self.concept.concept_name.isin(self.ingredients)
        med_concept_ids = (self.concept.loc[med_idx]
                               .reset_index()
                               .set_index('concept_name'))
        med_concept_ids.to_parquet(self.savedir+'med_concept_ids.parquet')
        print('Generating OMOP medication file...')
        keep_idx = self.relations.relationship_id == 'Has brand name'
        df = (self.relations.loc[keep_idx]
                            .drop(columns='relationship_id')
                            .reset_index(drop=True)
                            .rename(columns={'concept_id_1': 'ingredient_id',
                                             'concept_id_2': 'drug_id'}))

        df['ingredient'] = (self.concept.loc[df.ingredient_id]
                                        .reset_index(drop=True))
        df['drugname'] = (self.concept.loc[df.drug_id]
                                      .reset_index(drop=True))

        df = df.loc[df.ingredient.isin(ingredients)]

        df = (df.pivot(columns='ingredient', values='drugname')
                .apply(lambda x: pd.Series(x.dropna().values)))

        print(f'Saving {self.savepath}')
        df.to_csv(self.savepath, sep=';', index=None)

        self.ingredient_to_drug = df
        return self.ingredient_to_drug
