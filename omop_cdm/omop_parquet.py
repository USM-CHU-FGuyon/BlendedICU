import pandas as pd


def convert_to_parquet(pth_dic):
    pth_voc = pth_dic['vocabulary']
    print('Concept...')
    df = pd.read_csv(pth_voc+'CONCEPT.csv',
                     sep='\t',
                     index_col='concept_id')
    df.astype({'concept_code': str}).to_parquet(pth_voc+'CONCEPT.parquet')
    df = pd.read_csv(pth_voc+'CONCEPT_RELATIONSHIP.csv',
                     sep='\t')
    df.to_parquet(pth_voc+'CONCEPT_RELATIONSHIP.parquet')
