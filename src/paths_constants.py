from pathlib import Path

data = Path('../data/')
data_raw = data / 'raw'
data_processed = data / 'processed'

results = Path('../results/')

bots_file = data_processed / 'all_bots'

def data_processed_dataset(dataset):
    return data_processed / dataset

def dataset_edgelist(dataset):
    return data_processed_dataset(dataset) / 'edgelist'

def subgraph_edgelist(dataset, execution_type):
    return data_processed_dataset(dataset) / ('subgraph_' + execution_type)

def results_dataset(dataset):
    return results / dataset

def results_subgraph(dataset, execution_type):
    return results_dataset(dataset) / ('subgraph_' + execution_type)

def metrics_subgraph(dataset, execution_type):
    return results_subgraph(dataset, execution_type) / 'metrics'

def charts_subgraph(dataset, execution_type):
    return results_subgraph(dataset, execution_type) / 'charts'

def neural_network(dataset, execution_type):
    return results_subgraph(dataset, execution_type) / 'neural_network'
