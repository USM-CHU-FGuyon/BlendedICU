'''
This code generates the mapping between ingredients from the OMOP standard
vocabulary and the drugnames in the source databases.

The omop_medication module creates a ohdsi_icu_medications.csv table
which contains brand names for a number of ingredients.
This file can be completed by a manual_icu_meds.csv file that lists additional
ingredients or additional synonyms or brand names for an ingredient in the
ohdsi vocabulary.

The medication_mapping module searches the labels in the source databases
and creates a json file listing all labels associated to an ingredient for each
source database.
'''
import json

from med_to_omop.omop_medications import OMOP_Medications
from med_to_omop.medication_mapping import MedicationMapping
from omop_cdm import omop_parquet 

pth_dic = json.load(open('paths.json', 'r'))

omop_parquet.convert_to_parquet(pth_dic)

om = OMOP_Medications(pth_dic)

ingredient_to_drug = om.run()

mm = MedicationMapping(pth_dic)

medication_json = mm.run(load_drugnames=True, fname='medications.json')
