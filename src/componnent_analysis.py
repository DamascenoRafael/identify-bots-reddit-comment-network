import numpy as np
import pandas as pd
import networkx as nx

import paths_constants
from network import Network

class ComponnentAnalysis(Network):
    def __init__(self, dataset_file, execution_type):
        valid = {'strongly', 'weakly'}
        if execution_type not in valid:
            raise ValueError("ComponnentAnalysis: execution_type must be one of %r." % valid)

        edgelist_file = paths_constants.subgraph_edgelist(dataset_file.stem, execution_type)
        bots_file = paths_constants.bots_file

        self.dataset_file = dataset_file
        self.execution_type = execution_type
        self.DG = self._read_edgelist(edgelist_file)
        self.all_bots = self._read_bots(bots_file)
        self.bots_on_network = self._calculate_bots_on_network()

        self.dataframe = pd.DataFrame()
        self.dataframe_metrics_dict = dict()
        self.dataframe_column_names = []

    def calculate_all_metrics(self):
        self._calculate_degree_and_weight_metrics()
        self._calculate_clustering_metric()
        self._calculate_reciprocity_metrics()

    def _calculate_degree_and_weight_metrics(self):
        print('Calculating metrics related to degree and weight')
        in_degree = self.DG.in_degree()
        out_degree = self.DG.out_degree()
        self.__save_tuple_file(in_degree, 'in_degree')
        self.__save_tuple_file(out_degree, 'out_degree')

        sum_in_weight = self.DG.in_degree(weight='weight')
        sum_out_weight = self.DG.out_degree(weight='weight')
        self.__save_tuple_file(sum_in_weight, 'sum_in_weight')
        self.__save_tuple_file(sum_out_weight, 'sum_out_weight')

        diff_degree = []
        diff_weight = []
        sum_degree  = []
        sum_weight  = []
        frac_degree = []
        frac_weight = []
        for node in self.DG.nodes():
            diff_degree.append((node, in_degree[node] - out_degree[node]))
            diff_weight.append((node, sum_in_weight[node] - sum_out_weight[node]))
            sum_degree.append((node, in_degree[node] + out_degree[node]))
            sum_weight.append((node, sum_in_weight[node] + sum_out_weight[node]))

            num_frac_degree = in_degree[node]
            den_frac_degree = out_degree[node]
            num_frac_weight = sum_in_weight[node]
            den_frac_weight = sum_out_weight[node]
            if self.execution_type == 'weakly':
                num_frac_degree += 1
                den_frac_degree += 1
                num_frac_weight += 1
                den_frac_weight += 1
            frac_degree.append((node, num_frac_degree/den_frac_degree))
            frac_weight.append((node, num_frac_weight/den_frac_weight))
        self.__save_tuple_file(diff_degree, 'diff_degree')
        self.__save_tuple_file(diff_weight, 'diff_weight')
        self.__save_tuple_file(sum_degree, 'sum_degree')
        self.__save_tuple_file(sum_weight, 'sum_weight')
        self.__save_tuple_file(frac_degree, 'frac_degree')
        self.__save_tuple_file(frac_weight, 'frac_weight')

    def _calculate_clustering_metric(self):
        print('Calculating clustering')
        clustering = [(node, val) for node, val in nx.clustering(self.DG).items()]
        self.__save_tuple_file(clustering, 'clustering')

    def _calculate_reciprocity_metrics(self):
        print('Calculating metrics related to reciprocity')
        network_reciprocity_numerador = 0
        network_reciprocity_denominador = 0
        frac_reciprocity = []
        frac_reciprocity_bots = []
        frac_reciprocity_users = []
        for node in self.DG.nodes():
            successors = [n for n in self.DG.successors(node)]
            predecessors = [n for n in self.DG.predecessors(node)]
            neighbors = set(successors + predecessors)
            reciprocals = 0
            total_neighbors = len(neighbors)
            for neighbor in neighbors:
                if self.DG.has_edge(node,neighbor) and self.DG.has_edge(neighbor,node):
                    reciprocals += 1
    
            network_reciprocity_numerador += reciprocals
            network_reciprocity_denominador += total_neighbors
            node_reciprocity = reciprocals/total_neighbors
            frac_reciprocity.append((node, node_reciprocity))
            if node in self.all_bots:
                frac_reciprocity_bots.append((node, node_reciprocity))
            else:
                frac_reciprocity_users.append((node, node_reciprocity))
        self.__save_tuple_file(frac_reciprocity, 'frac_reciprocity')
        self.__save_tuple_file(frac_reciprocity_bots, 'frac_reciprocity_bots', only_summary=True)
        self.__save_tuple_file(frac_reciprocity_users, 'frac_reciprocity_users', only_summary=True)

        network_reciprocity_author = network_reciprocity_numerador/network_reciprocity_denominador
        self.__save_single_value_file(network_reciprocity_author, 'network_reciprocity_author_metric')

        network_reciprocity_nx = nx.overall_reciprocity(self.DG)
        self.__save_single_value_file(network_reciprocity_nx, 'network_reciprocity_nx_metric')

    def __save_tuple_file(self, node_info, filename, only_summary=False):
        subgraph_metrics_path = paths_constants.metrics_subgraph(self.dataset_file.stem, self.execution_type)
        subgraph_metrics_path.mkdir(parents=True, exist_ok=True)

        values = []
        if only_summary:
            for _, val in node_info:
                values.append(val)
        else:
            self.dataframe_column_names.append(filename)
            for key, val in node_info:
                values.append(val)
                self.dataframe_metrics_dict.setdefault(key, []).append(float(val))

        with open(subgraph_metrics_path / (filename + '_summary'), 'w') as file:
            file.write(filename + '\n')
            file.write('Max: ' + str(max(values)) + '\n')
            file.write('Min: ' + str(min(values)) + '\n')
            file.write('Mean: ' + str(np.mean(values)) + '\n')
            file.write('Median: ' + str(np.median(values)) + '\n')

    def __save_single_value_file(self, value, filename):
        subgraph_metrics_path = paths_constants.metrics_subgraph(self.dataset_file.stem, self.execution_type)
        subgraph_metrics_path.mkdir(parents=True, exist_ok=True)
        with open(subgraph_metrics_path / filename, 'w') as file:
            file.write(filename + '\n')
            file.write(str(value) + '\n')
    
    def save_dataframe(self):
        if self.dataframe_column_names == []:
            print('Run calculate_all_metrics() first. Current metrics are empty.')
            return
        
        self.dataframe_column_names.append('is_bot')
        for key in self.dataframe_metrics_dict:
            self.dataframe_metrics_dict[key].append(int(key in self.all_bots))

        self.dataframe = pd.DataFrame.from_dict(
            self.dataframe_metrics_dict,
            orient='index',
            columns=self.dataframe_column_names).astype(
                {'diff_degree': int,
                'diff_weight': int,
                'sum_degree': int,
                'sum_weight': int,
                'in_degree': int,
                'sum_in_weight': int,
                'out_degree': int,
                'sum_out_weight': int})

        self.dataframe.index.name = 'username'
        datafame_file = paths_constants.metrics_dataframe_subgraph_file(self.dataset_file.stem, self.execution_type)
        self.dataframe.to_csv(datafame_file)


class WeaklyComponnentAnalysis(ComponnentAnalysis):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'weakly')


class StronglyComponnentAnalysis(ComponnentAnalysis):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'strongly')
