import pandas as pd

import paths_constants

class DataframeReader:
    def __init__(self, dataset_file, execution_type):
        valid = {'strongly', 'weakly'}
        if execution_type not in valid:
            raise ValueError("DataframeReader: execution_type must be one of %r." % valid)

        datafame_file = paths_constants.metrics_dataframe_subgraph_file(dataset_file.stem, execution_type)

        self.dataset_file = dataset_file
        self.execution_type = execution_type
        self.dataframe = self.__read_dataframe(datafame_file)

    def __read_dataframe(self, dataframe_file):
        print('Reading dataframe file... Depending on the size this may take a while.')
        dataframe = pd.read_csv(dataframe_file)
        dataframe = dataframe.set_index('username')
        return dataframe
