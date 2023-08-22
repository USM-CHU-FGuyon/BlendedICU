import pandas as pd

from database_processing.flatandlabelsprocessor import FlatAndLabelsProcessor


def _take_sample(df, size, groupcol='source_dataset'):
    if size is None:
        return df
    return (df.groupby(groupcol)
              .apply(lambda x: x.sample(size))
              .reset_index(level=0, drop=True))


class blended_FLProcessor(FlatAndLabelsProcessor):
    def __init__(self, datasets, size=None):
        super().__init__(dataset='blended')
        self.size = size
        self.datasets = datasets
        self.labels = self._load_labels()

    def _load_labels(self):
        """
        Loads the label.parquet file from each source database.
        """
        labels_pths = {d: f'{self.pth_dic[d]}/preprocessed_labels.parquet'
                       for d in self.datasets}
        return {d: self.load(p).reset_index() for d, p in labels_pths.items()}

    def _fill_flat(self, df, **kwargs):
        """
        Missing values imputation of flat variables. This is optionnal using
        the FLAT_FILL_MEDIAN variable in config.json.
        """
        if self.FLAT_FILL_MEDIAN:
            return self.medianfill(df, **kwargs)
        return df

    def preprocess_labels(self):
        """
        Generates the flat variables from the labels.parquet files of each
        source database.
        Uses the parameters set in config.json for customizable processing 
        pipeline.
        """
        labels = self.labels

        for dataset, label in labels.items():
            label['source_dataset'] = dataset
            label['original_uniquepid'] = label['uniquepid']

        keepcols = ['source_dataset',
                    self.idx_col,
                    'original_uniquepid',
                    'care_site',
                    'unit_type',
                    'discharge_location',
                    'origin',
                    'age',
                    'sex',
                    'raw_height',
                    'raw_weight',
                    'true_lengthofstay',
                    'lengthofstay',
                    'mortality'
                    ]

        labels_blended = pd.concat([*labels.values()])[keepcols]

        labels_blended['discharge_location'] = (labels_blended.discharge_location
                                                .fillna('unknown')
                                                .map(self.discharge_locations))

        labels_blended['unit_type'] = (labels_blended.unit_type
                                                     .fillna('unknown')
                                                     .map(self.unit_types))

        labels_blended['origin'] = (labels_blended.origin
                                                  .fillna('unknown')
                                                  .map(self.admission_origins))

        labels_blended['from_US'] = (labels_blended.source_dataset
                                                   .isin(['mimic', 'eicu'])
                                                   .astype(int))

        labels_blended = labels_blended.astype({'original_uniquepid': str})

        labels_blended['uniquepid'] = labels_blended.apply(
            lambda x: f'{x.source_dataset}-{x.original_uniquepid}', axis=1)

        labels_blended['weight'] = labels_blended['raw_weight']
        labels_blended['height'] = labels_blended['raw_height']

        labels_blended = (labels_blended.set_index(self.idx_col)
                                        .pipe(_take_sample,
                                              size=self.size)
                                        .pipe(self.clip_and_norm,
                                              cols=['height', 'weight'],
                                              clip=self.FLAT_CLIP,
                                              normalize=self.FLAT_NORMALIZE)
                                        .pipe(self._fill_flat,
                                              cols=['height', 'weight'])
                                        .astype({'age': int,
                                                 'raw_weight': float,
                                                 'raw_height': float,
                                                 'uniquepid': str,
                                                 'source_dataset': str,
                                                 'original_uniquepid': str,
                                                 'mortality': int,
                                                 'lengthofstay': float,
                                                 'origin': str,
                                                 'care_site': str}))

        flat = labels_blended.loc[:, ['age',
                                      'sex',
                                      'raw_height',
                                      'raw_weight',
                                      'from_US',
                                      'source_dataset']]

        flat = (flat.groupby('source_dataset')
                    .apply(self._fill_flat,
                           cols=['raw_height', 'raw_weight'])
                    .pipe(self.clip_and_norm,
                          cols=['age', 'raw_height', 'raw_weight'],
                          clip=self.FLAT_CLIP,
                          normalize=self.FLAT_NORMALIZE,
                          recompute_quantiles=True
                          ))

        return labels_blended, flat

    def preprocess_flat(self):
        pass
