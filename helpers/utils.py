#!/usr/bin/env python
# coding: utf-8


import networkx as nx
import numpy as np

def reoder_graph_labels(networkx_graphs):
    
    for i in range(len(networkx_graphs)-1):
    
        start = sorted(networkx_graphs[i])[-1]

        keys = sorted(networkx_graphs[i+1])

        values = list(range(start+1, start + keys[-1] + 2))
        mapping = dict(zip(keys, values))
        networkx_graphs[i+1] = nx.relabel_nodes(networkx_graphs[i+1], mapping)
  
    return networkx_graphs


def get_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params