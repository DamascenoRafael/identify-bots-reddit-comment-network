import networkx as nx

import paths_constants

class Network:
    def __init__(self, dataset_file):
        edgelist_file = paths_constants.dataset_edgelist(dataset_file.stem)
        bots_file = paths_constants.bots_file

        self.dataset_file = dataset_file
        self.DG = self._read_edgelist(edgelist_file)
        self.all_bots = self._read_bots(bots_file)
        self.bots_on_network = self._calculate_bots_on_network()

    def get_summary(self):
        print(nx.info(self.DG))
        print('Graph Density:  ', nx.density(self.DG))
        print('Total of Comments:  ', int(self.DG.size(weight='weight')))
        print('Bots on Network:  ', len(self.bots_on_network))

    def _read_edgelist(self, edgelist_file):
        DG = nx.DiGraph()
        DG.name = self.dataset_file.stem
        with open(edgelist_file) as infile:
            print('Reading edgelist file... Depending on the size this may take a while.')
            for edge in infile:
                u, v, w = edge.split()
                DG.add_edge(u.strip().lower(), v.strip().lower(), weight=int(w))
        return DG
    
    def _read_bots(self, bots_file):
        all_bots = set()
        with open(bots_file) as infile:
            print('Reading bots file...')
            for bot in infile:
                all_bots.add(bot.strip().lower())
        return all_bots
    
    def _calculate_bots_on_network(self):
        bots_on_network = []
        for bot in self.all_bots:
            if bot in self.DG:
                bots_on_network.append(bot)
        return bots_on_network
