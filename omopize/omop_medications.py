from pathlib import Path

import pandas as pd

class OMOP_Medications:
    def __init__(self, pth_dic, ingredients=None):
        '''A list of ingredients may be given as an input. 
        If ingredients is None, the user_input/medication_ingredients.csv
        file will be used as the list of ingredients.'''
        self.pth_to_voc = pth_dic['vocabulary']
        self.pth_user_input = pth_dic['user_input']
        self.savedir = pth_dic['medication_mapping_files']
        self.savepath = self.savedir+'ohdsi_icu_medications.csv'
        
        self.concept = self._load_concept()
        self.relations = self._load_concept_relationship()
        
        self.ingredients = self._get_ingredients(ingredients)

        self._check_unique_ingredients()

    def _load_concept(self):
        print('Loading CONCEPT table...')
        concept_pth = self.pth_to_voc+'CONCEPT.parquet'
        concept = pd.read_parquet(concept_pth,
                                  columns=['concept_id', 'concept_name'])
        return concept
    
    def _load_concept_relationship(self):
        print('Loading CONCEPT_RELATIONSHIP table...')
        relationship_pth = self.pth_to_voc+'CONCEPT_RELATIONSHIP.parquet'
        concept_relationship = pd.read_parquet(relationship_pth,
                                               columns=['concept_id_1',
                                                        'concept_id_2',
                                                        'relationship_id'])
        return concept_relationship
    
    
    def _get_ingredients(self, ingredients):
        """
        If ingredient is None: read ingredient file.
        else: use the ingredient list provided"""
        if ingredients is None: 
            ingredients_pth = self.pth_user_input+'medication_ingredients.csv'    
            print(f'Loading ingredients from {ingredients_pth}')
            df = pd.read_csv(ingredients_pth)
            ingredients = df.ingredient.to_list()
        return ingredients
        
        
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

        df = df.loc[df.ingredient.isin(self.ingredients)]

        df = (df.pivot(columns='ingredient', values='drugname')
                .apply(lambda x: pd.Series(x.dropna().values)))

        print(f'Saving {self.savepath}')
        df.to_csv(self.savepath, sep=';', index=None)
        self.ingredient_to_drug = df
        return self.ingredient_to_drug
