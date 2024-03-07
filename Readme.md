Welcome to the BlendedICU code repository
===
This is the repository from [Introducing the BlendedICU dataset, the first harmonized, international intensive care dataset](https://www.sciencedirect.com/science/article/abs/pii/S153204642300223X)

[ResearchGate link](https://www.researchgate.net/publication/374201062_Introducing_the_BlendedICU_dataset_the_first_harmonized_international_intensive_care_dataset) 

Abstract
---
**Objective**
This study introduces the **BlendedICU** dataset, a massive dataset of international intensive care data. This dataset aims to facilitate generalizability studies of machine learning models, as well as statistical studies of clinical practices in the intensive care units.

**Methods**
Four publicly available and patient-level intensive care databases were used as source databases. A unique and customizable preprocessing pipeline extracted clinically relevant patient-related variables from each source database. Variables were then harmonized and standardized to the Observational Medical Outcomes Partnership (OMOP) Common Data Format. Finally, a brief comparison was carried out to explore differences in the source databases.

**Results**
The **BlendedICU** dataset features 41 timeseries variables as well as the exposure times to 113 active ingredients extracted from the AmsterdamUMCdb, eICU, HiRID, and MIMIC-IV databases. This resulted in a database of more than 309000 intensive care admissions, spanning over 13 years and three countries. We found that data collection, drug exposure, and patient outcomes varied strongly between source databases.

**Conclusion**
The variability in data collection, drug exposure, and patient outcomes between the source databases indicated some dissimilarity in patient phenotypes and clinical practices between different intensive care units. This demonstrated the need for generalizability studies of machine learning models. This study provides the clinical data research community with essential data to build efficient and generalizable machine learning models, as well as to explore clinical practices in intensive care units around the world.


```
@article{Oliver2023Introducing,
title = {Introducing the BlendedICU dataset, the first harmonized, international intensive care dataset},
journal = {Journal of Biomedical Informatics},
volume = {146},
pages = {104502},
year = {2023},
issn = {1532-0464},
doi = {https://doi.org/10.1016/j.jbi.2023.104502},
url = {https://www.sciencedirect.com/science/article/pii/S153204642300223X},
author = {Matthieu Oliver and Jérôme Allyn and Rémi Carencotte and Nicolas Allou and Cyril Ferdynus},
keywords = {OMOP common data format, Intensive care unit database, Data integration}
}
```
This repository contains the codes and files that allow the creation of the 
BlendedICU dataset from the AmsterdamUMCdb, eICU, HiRID, MIMIC-III and MIMIC-IV databases.

[<img src="plot/kdeplot.png" width="600"/>](plot/kdeplot.png)

Before you begin
---

The codes require some user input to run. 
#### `paths.json` 
Fill the following paths in the `path.json` file. The file should be formatted as such:
```
{
  "data_path": "pth/to/BLENDED_ICU/",
  "eicu_source_path":"pth/to/eicu/installation/",
  "mimic3_source_path": "pth/to/mimic3/installation/",
  "mimic4_source_path": "pth/to/mimic4/installation/",
  "amsterdam_source_path": "pth/to/amsterdam/installation/",
  "hirid_source_path": "pth/to/hirid/installation/",
  "auxillary_files": "auxillary_files/",
  "vocabulary" : "auxillary_files/OMOP_vocabulary/",
  "user_input" : "auxillary_files/user_input/",
  "medication_mapping_files": "auxillary_files/medication_mapping_files/"
}
```
#### `config.json` 
This file contains parameters for customization of the processing pipeline. 
See the content of this file for a description of each parameter.

#### `user_input/`: 
This directory contains the mappings of categorical flat variables (
`admission_origins.json`, `discharge_location.json`, `unit_type_v2.json`) and the mapping of the 
timeseries variables (`timeseries_variables.csv`). Additionaly, `timeseries_variables.csv` contains 
processing options for each timeseries variables, such user-defined minimum and maximum values, as well
as the choice of the aggregation method for downsampling ('last' or 'mean').

##### JSON files
The json files in the `user_input/` directory are structured as follows:
```
{
    "Category_in_blendedicu":[
        "category1_from_some_database",
        "category2",
        "CategOry"
    ],
    "another_category":[
        "this_category_in_mimic",
        "multiple_other_names_for_the_same_category"
]
}
```
A name can only appear in a single category. Category names are case-sensitive.

##### `timeseries_variables.csv`
This file contains the correspondence of variable names from source databases. It is formatted as follows:

```
concept_id;blended;eicu;mimic;amsterdam;hirid;categories;user_min;user_max;is_numeric;agg_method;unit_concept_id
4239408;heart_rate;Heart Rate;Heart Rate;Hartfrequentie;Heart rate;Vitals;0;;1;mean;8541
```
It also contains preprocessing options, such as user-defined minimum and maximum allowable values, and the aggregation method ('last' or 'mean') used when downsampling the original data.


#### Vocabulary
The OMOP vocabulary should be downloaded from https://athena.ohdsi.org/ and include the following vocabularies 'OMOP Extension', 'RxNorm Extension', 'RxNorm', 'LOINC', 'SNOMED'. The directory containing the vocabularies should then be placed in the location specified in paths.json.

**Note:** For better performance, the user should convert the csv files to `parquet` files. 
```
import pandas as pd
pd.read_csv('CONCEPT.csv', sep='\t').to_parquet('CONCEPT.parquet')
```
#### Python versions and packages
We provided the `env.yml` file that lists all dependencies. This code requires Python>=3.9.
This code utilizes the `.parquet` format using the pyarrow library as a backend of pandas. This format facilitates the handling of large dataframes, notably because of faster I/O operations and efficient compression.

Running the pipeline
---

#### Step 0. File preparation

Run `0_prepare_files.py` to create the medication mapping file. The default output is already available in the repository at `auxillary_files/medications_v9.json`. This file contains all the labels corresponding to the included ingredients.

#### Step 1. Database extraction
`1_{dataset}.py` runs the extraction pipeline, it utilises the DataPreparator and MedicationProcessor objects.
1. Flat data is extracted and stored into a `labels.parquet` file.
2. Timeseries data is converted to parquet format as well. Some ununsed variables are dropped at this stage. All timestamps are converted into seconds since admission. In some cases the parquet are saved by chunks of 1000 patients to reduce the memory requirements of the process.
3. Drug exposures are processed using the medication mapping file produced at stage 0. This creates a `medication.parquet` file which contains standardized drug administrations.

#### Step 2. Harmonization
`2_{dataset}.py` runs the harmonization pipeline, it utilises the TimeseriesProcessor and FlatAndLabelsProcessor objects.
1. The timeseries are harmonized to common labels and units between all databases. Some user-defined bounds are applied to the data. The raw harmonized data is saved in the `formatted_timeseries/` and `formatted_medications/` directories. Then, these timeseries are resampled to hourly data and saved to the `partially_processed_timeseries/` directory.
2. Flat categorical variables are mapped to standardized categories. Flat numerical vairables such as length of stay, heights, weights are converted to the same units. Finally an index for each icu stay that is unique in the BlendedICU dataset, it is constituted as follows: {source_database}-{stay_id_in_source_database}

#### Step 3. BlendedICU processing
`3_BlendedICU.py` utilises the TimeseriesProcessor and FlatAndLabelsProcessor objects. It runs a customizable processing pipeline using the partially-processed files from the harmonization step.
1. Flat variables of all soruce databases are concatenated. Some user options are then available for processing: `FLAT_FILL_MEDIAN`, `FLAT_NORMALIZE`, `FLAT_CLIP`. See `config.json` for more detail. The output is saved as `preprocessed_labels.parquet`.
2. Timeseries are processed with four available options: `FORWARD_FILL`, `TS_FILL_MEDIAN`, `TS_CLIP`, and `TS_NORMALIZE`. See `config.json` for more detail. The output is saved in the `preprocessed_timeseries/` directory. A parquet file is created for each icu stay. Note that pandas allows loading a list of parquet files by using the following syntax: ```pd.read_parquet([pth1, pth2])```

#### Step 4. OMOP conversion
`4_write_omop.py` writes the harmonized BlendedICU database to the Observational Medical Outcomes Partnership Common Data Model. The MEASUREMENT and DRUG_EXPOSURE tables are saved as parquet by chunks of 1000 patients. We provide the option to save the data as csv, but the resulting database would exceed 300Go.

#### Steps 5 & 6. Producing the tables and Figures of the article
These files are used to reproduce the tables, figures and appendices of the article in latex-compatible format. They were not as deeply documented as the rest of the code but were included for completeness.

Concluding remarks
---
These codes are meant to be maintained and improved and we welcome any questions, suggestions or bug reports in the Issues section of our GitHub repository.
