"""

TODO : Most of the work for diagnoses should be here.


We need to create a json file linking icd_code entries in the source databases
to OMOP concept_ids. 

Some choices should be made, for example I chose to group all Influenza related
codes and link them to the "Influenza" concept id ().

The detailed code label (eg. Influenza due to other identified influenza virus with otitis media)
will be found in the final omop table as condition_occurrence_source_value.


For a systematic approach, a mapping from ICD9/ICD10 to OMOP should be used.



The current minimalistic approach currently OMOPizes Influenza and
 unspecified_essential_hypertension from MIMIC IV.

"""
import json


class DiagnosesMapping:
    '''Class that produces a json file for mapping icd codes to concept ids
    
    mimic 4 contains both ICD 9 and ICD10 codes
        In the codes I chose to prepend "{icd_version}_" to the icd_code.
        
        "9_E8881":Fall resulting in striking against other object
        "10_E8881": Metabolic syndrome
    
    
    json should look like:
        
    {
     'metabolic_syndrome': {
         'blended': 436940,#concept_id
         'mimic4': ['10_E8881'] #icd codes with prepended versiion
         },
     ...
     }
        

    
    
    
    '''
    def __init__(self, pth_dic, datasets):
        self.datasets = datasets
        self.aux_pth = pth_dic['auxillary_files']
    
    
    
    def run(self):
        """
        Need to find an automatized way to produce this file with all possible 
        entries in mimic iv.
        
        Then we will try to adapt the approach to mimic3 & eicu. 
        
        """
        
        raise NotImplementedError
        
        diagnoses_json = {
                        "unspecified_essential_hypertension": {
                            "blended": 320128,
                            "mimic4": [
                                "9_4019"
                            ],
                            "eicu": [
                            ]
                    	},
                        "influenza":{  #https://www.ncbi.nlm.nih.gov/books/NBK550181/table/sb253.tab8/
                    	"blended": 4266367,
                    	"mimic4":["10_J09X1",
                    		 "10_J09X2",
                    		 "10_J09X3",
                    		 "10_J09X9",
                    		 "10_J1000",
                    		 "10_J1001",
                    		 "10_J1008",
                    		 "10_J101",
                    		 "10_J102",
                    		 "10_J1081",
                    		 "10_J1082",
                    		 "10_J1089",
                    		 "10_J1100",
                     		"10_J1108",
                     		"10_J111",
                     		"10_J112",
                     		"10_J1181",
                     		"10_J1189",
                    		 "9_4870",
                     		"9_4871",
                     		"9_4878",
                     		"9_48801",
                     		"9_48802",
                     		"9_48809",
                     		"9_48811",
                     		"9_48881",
                     		"9_48882",
                     		"9_48889"]
                    }

        }
        fname = 'diagnoses.json'
        json.dump(diagnoses_json,
                  open(self.aux_pth+fname, 'w'),
                  indent=4,
                  ensure_ascii=False)
