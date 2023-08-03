from database_processing.datapreparator import DataPreparator


class blendedicuPreparator(DataPreparator):
    def __init__(self):
        super().__init__(dataset='blended', col_stayid='patient')
