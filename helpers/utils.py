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

def class_wise_accuracy(confusion_matrix, target_names):
    """
    confusion matrix of multi-class classification
    
    target_names: name of a particular class 
    
    """
    confusion_matrix = np.float64(confusion_matrix)
    classwise_accuracy = {}
    for i in range(confusion_matrix.shape[0]):
        class_id = i
        TP, FN, FP, TN, accuracy = 0, 0, 0, 0, 0
        
        TP = confusion_matrix[class_id,class_id]
        FN = np.sum(confusion_matrix[class_id]) - TP
        FP = np.sum(confusion_matrix[:,class_id]) - TP
        TN = np.sum(confusion_matrix) - TP - FN - FP

        #print(target_names[i], (TP,TN), (FP,FN))
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        classwise_accuracy[target_names[i]] = accuracy
        
    return classwise_accuracy