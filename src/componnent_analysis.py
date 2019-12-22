import numpy as np
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

    def calculate_all_metrics(self):
        self._calculate_degree_and_weight_metrics()
        self._calculate_clustering_metric()
        self._calculate_reciprocity_metrics()

    def _calculate_degree_and_weight_metrics(self):
        print('Calculating metrics related to degree and weight')
        dg_in_degree = self.DG.in_degree()
        dg_out_degree = self.DG.out_degree()
        self.__save_tuple_file(dg_in_degree, 'in_degree')
        self.__save_tuple_file(dg_out_degree, 'out_degree')

        dg_in_degree_weighted = self.DG.in_degree(weight='weight')
        dg_out_degree_weighted = self.DG.out_degree(weight='weight')
        self.__save_tuple_file(dg_in_degree_weighted, 'in_degree_weighted')
        self.__save_tuple_file(dg_out_degree_weighted, 'out_degree_weighted')

        diff_degree = []
        sum_degree  = []
        frac_degree = []
        frac_degree_weighted = []
        for node in self.DG.nodes():
            diff_degree.append((node, dg_in_degree[node] - dg_out_degree[node]))
            sum_degree.append((node, dg_in_degree[node] + dg_out_degree[node]))

            num_frac_degree = dg_in_degree[node]
            den_frac_degree = dg_out_degree[node]
            num_frac_degree_weighted = dg_in_degree_weighted[node]
            den_frac_degree_weighted = dg_out_degree_weighted[node]
            if self.execution_type == 'weakly':
                num_frac_degree += 1
                den_frac_degree += 1
                num_frac_degree_weighted += 1
                den_frac_degree_weighted += 1
            frac_degree.append((node, num_frac_degree/den_frac_degree))
            frac_degree_weighted.append((node, num_frac_degree_weighted/den_frac_degree_weighted))
        self.__save_tuple_file(diff_degree, 'diff_degree')
        self.__save_tuple_file(sum_degree, 'sum_degree')
        self.__save_tuple_file(frac_degree, 'frac_degree')
        self.__save_tuple_file(frac_degree_weighted, 'frac_degree_weighted')

    def _calculate_clustering_metric(self):
        print('Calculating clustering')
        clustering = [(node, val) for node, val in nx.clustering(self.DG).items()]
        self.__save_tuple_file(clustering, 'clustering')

    def _calculate_reciprocity_metrics(self):
        print('Calculating metrics related to reciprocity')
        frac_reciprocals_network_numerador = 0
        frac_reciprocals_network_denominador = 0
        frac_neighbors_reciprocals = []
        frac_neighbors_reciprocals_bots = []
        frac_neighbors_reciprocals_users = []
        for node in self.DG.nodes():
            successors = [n for n in self.DG.successors(node)]
            predecessors = [n for n in self.DG.predecessors(node)]
            neighbors = set(successors + predecessors)
            reciprocals = 0
            total_neighbors = len(neighbors)
            for neighbor in neighbors:
                if self.DG.has_edge(node,neighbor) and self.DG.has_edge(neighbor,node):
                    reciprocals += 1
    
            frac_reciprocals_network_numerador += reciprocals
            frac_reciprocals_network_denominador += total_neighbors
            node_reciprocity = reciprocals/total_neighbors
            frac_neighbors_reciprocals.append((node, node_reciprocity))
            if node in self.all_bots:
                frac_neighbors_reciprocals_bots.append((node, node_reciprocity))
            else:
                frac_neighbors_reciprocals_users.append((node, node_reciprocity))
        self.__save_tuple_file(frac_neighbors_reciprocals, 'frac_neighbors_reciprocals')
        self.__save_tuple_file(frac_neighbors_reciprocals_bots, 'frac_neighbors_reciprocals_bots', only_summary=True)
        self.__save_tuple_file(frac_neighbors_reciprocals_users, 'frac_neighbors_reciprocals_users', only_summary=True)

        network_reciprocity_user = frac_reciprocals_network_numerador/frac_reciprocals_network_denominador
        self.__save_single_value_file(network_reciprocity_user, 'network_reciprocity_from_user')

        network_reciprocity_nx = nx.overall_reciprocity(self.DG)
        self.__save_single_value_file(network_reciprocity_nx, 'network_reciprocity_from_nx')

    def __save_tuple_file(self, node_info, filename, only_summary=False):
        subgraph_metrics_path = paths_constants.metrics_subgraph(self.dataset_file.stem, self.execution_type)
        subgraph_metrics_path.mkdir(parents=True, exist_ok=True)

        values = []
        if only_summary:
            for _, val in node_info:
                values.append(val)
        else:
            with open(subgraph_metrics_path / filename, 'w') as file:
                for key, val in node_info:
                    values.append(val)
                    if not only_summary:
                        file.write(key + ' ' + str(val) + '\n')

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
            file.write(str(value) + '\n')


class WeaklyComponnentAnalysis(ComponnentAnalysis):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'weakly')


class StronglyComponnentAnalysis(ComponnentAnalysis):
    def __init__(self, dataset_file):
        super().__init__(dataset_file, 'strongly')
