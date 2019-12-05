import numpy as np
import networkx as nx
from prettytable import PrettyTable

import paths_constants
from network import Network

class ComponnentsInfo(Network):
    def __components_info(self, execution_type, save):
        valid = {'strongly', 'weakly'}
        if execution_type not in valid:
            raise ValueError("execution_type must be one of %r." % valid)

        components = []
        if execution_type == 'weakly':
            components = list(nx.weakly_connected_components(self.DG))
        else:
            components = list(nx.strongly_connected_components(self.DG))
        
        components_size = [len(x) for x in components]
        large_component = max(components, key=len)

        bots_in_components = []
        for component in components:
            bots_count = 0
            for node in component:
                if node in self.all_bots:
                    bots_count += 1
            bots_in_components.append(bots_count)
        
        components_table = self.__components_table(components_size, bots_in_components)

        print(execution_type + ' connected components')
        print(components_table)
        print('# total bots in network:', len(self.bots_on_network))
        print('# total components:', len(components))
        print('large component size:', len(large_component))
        if save:
            self.__save_subgraph(self.DG.subgraph(large_component), execution_type)

    def __components_table(self, components_size, bots_in_components):
        components_table = PrettyTable()
        components_table.align = 'r'
        components_table.field_names = ['# nodes', '# components', '# bots']

        unique_values, indices, counts = np.unique(components_size, return_counts = True, return_inverse = True)
        
        bots_on_components_size = {idx_size: 0 for idx_size in range(len(unique_values))}

        for idx_component, idx_size in enumerate(indices):
            bots_on_components_size[idx_size] += bots_in_components[idx_component]
 
        for i in range(len(unique_values)):
            components_table.add_row([unique_values[i], counts[i], bots_on_components_size[i]])
        
        return components_table

    def __save_subgraph(self, subgraph, execution_type):
        print('Saving subgraph of the largest component...')
        subgraph_file = paths_constants.subgraph_edgelist(self.dataset_file.stem, execution_type)
        nx.write_weighted_edgelist(subgraph, subgraph_file)

    def weakly_components_info(self, save=True):
        self.__components_info('weakly', save)

    def strongly_components_info(self, save=True):
        self.__components_info('strongly', save)
