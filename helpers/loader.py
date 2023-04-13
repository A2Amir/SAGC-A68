#!/usr/bin/env python
# coding: utf-8

import torch
import dgl
import networkx as nx
import json
import numpy as np
import random

from dgl.data import DGLDataset
from node2vec import Node2Vec
from operator import itemgetter
from helpers.graph import add_edge_attribute, normalize_coordinate
from helpers.graph import add_node_attribute, random_walk_augmentation, dtype_transformation




def load_graph(label_path, graph_type = 'space_access_relation'):

    assert graph_type in ['space_access_relation', 'space_adjacency_relation']
    
    with open(label_path, 'r') as f:
        data = json.load(f)
        
    x_min = dtype_transformation(data["graph_bounding_box"] ["min_x"])
    y_min = dtype_transformation(data["graph_bounding_box"] ["min_y"])
    z_min = dtype_transformation(data["graph_bounding_box"] ["min_z"])
    x_max = dtype_transformation(data["graph_bounding_box"] ["max_x"])
    y_max = dtype_transformation(data["graph_bounding_box"] ["max_y"])
    z_max = dtype_transformation(data["graph_bounding_box"] ["max_z"])

    graph_name = label_path.split('.')[1].split('\\')[-1]
    G = nx.Graph(name= graph_name, graph_bounding_box = [x_min, y_min, z_min, x_max, y_max, z_max] )
    
    space_element_ids = list(map(itemgetter('id'), data['space_element_set']))
    space_ids = list(map(itemgetter('id'), data['whole_space_set']))
    
    unique_edge_ids = []
    for edge in data[graph_type]:
        
        
        source_node_id = str(edge['source'])
        source_node_type = ['space_element_set' if source_node_id in space_element_ids else 'whole_space_set'][0]
        source_node_data = list(filter(lambda node: node['id'] == source_node_id,  data[source_node_type]))[0]
        
        source_node_x = dtype_transformation(source_node_data['x']) 
        source_node_y = dtype_transformation(source_node_data['y'])
        source_node_z = dtype_transformation(source_node_data['z'])   
        source_node_width = dtype_transformation(source_node_data['width'])
        source_node_height = dtype_transformation(source_node_data['height'])
        source_node_depth = dtype_transformation(source_node_data['depth'])
        source_node_area = dtype_transformation(source_node_data['area'])
        source_node_volume = dtype_transformation(source_node_data['volume'])
        source_node_internal = dtype_transformation(source_node_data['is_internal'])
        source_node_door_opening_quantity = dtype_transformation(source_node_data['door_opening_quantity'])
        source_node_window_quantity = dtype_transformation(source_node_data['window_quantity'])
        source_node_max_door_width = dtype_transformation(source_node_data['max_door_width'])
        source_node_encloses_ws = dtype_transformation(source_node_data['encloses_ws'])
        source_node_contained_in_ws = dtype_transformation(source_node_data['is_contained_in_ws'])
        source_node_class = source_node_data['class']

        G.add_nodes_from([(source_node_id, {'label': source_node_class,
                                            'pos':   [source_node_x, source_node_y, source_node_z], 
                                            'feats': [source_node_width, source_node_height, source_node_depth,
                                                      source_node_area, source_node_volume, source_node_internal,
                                                      source_node_door_opening_quantity, source_node_window_quantity,
                                                      source_node_max_door_width, source_node_encloses_ws,
                                                      source_node_contained_in_ws
                                                     ]
                                           })])
    
    
        target_node_id = str(edge['target'])
        target_node_type = ['space_element_set' if target_node_id in space_element_ids else 'whole_space_set'][0]
        target_node_data = list(filter(lambda node: node['id'] == target_node_id,  data[target_node_type]))[0]
       
        target_node_x = dtype_transformation(target_node_data['x'])
        target_node_y = dtype_transformation(target_node_data['y']) 
        target_node_z = dtype_transformation(target_node_data['z'])
        target_node_width = dtype_transformation(target_node_data['width'])
        target_node_height = dtype_transformation(target_node_data['height'])
        target_node_depth = dtype_transformation(target_node_data['depth']) 
        target_node_area = dtype_transformation(target_node_data['area'])
        target_node_volume = dtype_transformation(target_node_data['volume'])
        target_node_internal = dtype_transformation(target_node_data['is_internal'])
        target_node_door_opening_quantity = dtype_transformation(target_node_data['door_opening_quantity'])
        target_node_window_quantity = dtype_transformation(target_node_data['window_quantity'])
        target_node_max_door_width = dtype_transformation(target_node_data['max_door_width'])
        target_node_encloses_ws = dtype_transformation(target_node_data['encloses_ws'])
        target_node_contained_in_ws = dtype_transformation(target_node_data['is_contained_in_ws'])
        target_node_class = target_node_data['class']
        
        G.add_nodes_from([(target_node_id, {'label': target_node_class,
                                            'pos':   [target_node_x, target_node_y, target_node_z],
                                            'feats': [target_node_width, target_node_height, target_node_depth,
                                                      target_node_area, target_node_volume,  target_node_internal,
                                                      target_node_door_opening_quantity, target_node_window_quantity,
                                                      target_node_max_door_width, target_node_encloses_ws,
                                                      target_node_contained_in_ws
                                                     ]
                                           
                                           
                                           })])
        
        z_angle = dtype_transformation(edge['z_angle']) 
        delta_z = dtype_transformation(edge['delta_z']) 
        length = dtype_transformation(edge['length']) 
        G.add_edge(source_node_id, target_node_id, key=edge['id'], feats= [z_angle, delta_z, length])
                
        unique_edge_ids.extend([source_node_id, target_node_id])
    
    # adding disconnected sapce elements
    for se_id in space_element_ids:
        if se_id not in set(unique_edge_ids):
            disc_se_node_data = list(filter(lambda node: node['id'] == se_id,  data['space_element_set']))[0]
            
            disc_se_node_id = str(disc_se_node_data['id'])
            disc_se_node_x = dtype_transformation(disc_se_node_data['x']) 
            disc_se_node_y = dtype_transformation(disc_se_node_data['y'])
            disc_se_node_z = dtype_transformation(disc_se_node_data['z'])   
            disc_se_node_width = dtype_transformation(disc_se_node_data['width'])
            disc_se_node_height = dtype_transformation(disc_se_node_data['height'])
            disc_se_node_depth = dtype_transformation(disc_se_node_data['depth'])
            disc_se_node_area = dtype_transformation(disc_se_node_data['area'])
            disc_se_node_volume = dtype_transformation(disc_se_node_data['volume'])
            isc_se_node_internal = dtype_transformation(disc_se_node_data['is_internal'])
            disc_se_node_door_opening_quantity = dtype_transformation(disc_se_node_data['door_opening_quantity'])
            disc_se_node_window_quantity = dtype_transformation(disc_se_node_data['window_quantity'])
            disc_se_node_max_door_width = dtype_transformation(disc_se_node_data['max_door_width'])
            disc_se_node_encloses_ws = dtype_transformation(disc_se_node_data['encloses_ws'])
            disc_se_node_contained_in_ws = dtype_transformation(disc_se_node_data['is_contained_in_ws'])
            disc_se_node_class = disc_se_node_data['class']

            G.add_nodes_from([(disc_se_node_id, {'label': disc_se_node_class,
                                                 'pos':   [ disc_se_node_x,  disc_se_node_y,  disc_se_node_z], 
                                                 'feats': [ disc_se_node_width,  disc_se_node_height,  disc_se_node_depth,
                                                            disc_se_node_area,  disc_se_node_volume,  disc_se_node_internal,
                                                            disc_se_node_door_opening_quantity,  disc_se_node_window_quantity,
                                                            disc_se_node_max_door_width,  disc_se_node_encloses_ws,
                                                            disc_se_node_contained_in_ws
                                                          ]
                                                 })
                             ])

    # adding disconnected sapces
    for s_id in space_ids:
        if s_id not in set(unique_edge_ids):
            disc_s_node_data = list(filter(lambda node: node['id'] == s_id,  data['whole_space_set']))[0]
            
            disc_s_node_id = str(disc_s_node_data['id'])
            disc_s_node_x = dtype_transformation(disc_s_node_data['x']) 
            disc_s_node_y = dtype_transformation(disc_s_node_data['y'])
            disc_s_node_z = dtype_transformation(disc_s_node_data['z'])   
            disc_s_node_width = dtype_transformation(disc_s_node_data['width'])
            disc_s_node_height = dtype_transformation(disc_s_node_data['height'])
            disc_s_node_depth = dtype_transformation(disc_s_node_data['depth'])
            disc_s_node_area = dtype_transformation(disc_s_node_data['area'])
            disc_s_node_volume = dtype_transformation(disc_s_node_data['volume'])
            disc_s_node_internal = dtype_transformation(disc_s_node_data['is_internal'])
            disc_s_node_door_opening_quantity = dtype_transformation(disc_s_node_data['door_opening_quantity'])
            disc_s_node_window_quantity = dtype_transformation(disc_s_node_data['window_quantity'])
            disc_s_node_max_door_width = dtype_transformation(disc_s_node_data['max_door_width'])
            disc_s_node_encloses_ws = dtype_transformation(disc_s_node_data['encloses_ws'])
            disc_s_node_contained_in_ws = dtype_transformation(disc_s_node_data['is_contained_in_ws'])
            disc_s_node_class = disc_s_node_data['class']

            G.add_nodes_from([(disc_s_node_id, {'label': disc_s_node_class,
                                                 'pos':   [ disc_s_node_x,  disc_s_node_y,  disc_s_node_z], 
                                                 'feats': [ disc_s_node_width,  disc_s_node_height,  disc_s_node_depth,
                                                            disc_s_node_area,  disc_s_node_volume,  disc_s_node_internal,
                                                            disc_s_node_door_opening_quantity,  disc_s_node_window_quantity,
                                                            disc_s_node_max_door_width,  disc_s_node_encloses_ws,
                                                            disc_s_node_contained_in_ws
                                                          ]
                                                 })
                             ])       
            

    return G

def preprocessing(graph_paths, graph_type = 'space_access_relation'):
    
    networkx_graphs = []
    adjacency_matrices = []
    

    for graph_path in graph_paths:

        graph = load_graph(graph_path, graph_type)
        adjacency_matrix = nx.to_numpy_array(graph) # weight='distance'

        # p stay on the surrounding of the same node and q is how much we want to move away
        node2vec = Node2Vec(graph, dimensions=128, walk_length=18, num_walks=12, workers=4, p= 0.7,  q= 0.3)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)

        node2vec_features = {}
        for node in graph.nodes:
            node2vec_features[node] =  model.wv.get_vector(node)
        nx.set_node_attributes(graph, node2vec_features, "node2vec_feats") 
        
        
        networkx_graphs.append(graph)
        adjacency_matrices.append(adjacency_matrix)
        
        
    return networkx_graphs, adjacency_matrices



class create_dataset(DGLDataset):
    """Subset of a dataset at specified indices
    Code adapted from PyTorch.

    Parameters
    ----------
    dataset
        dataset[i] should return the ith datapoint
    indices : list
        List of datapoint indices to construct the subset
    """
    def __init__(self, networkx_graphs, LabEnc, selected_node_feats, augmentation=False):
        
        self.networkx_graphs = networkx_graphs
        self.LabEnc = LabEnc
        self.selected_node_feats = selected_node_feats
        self.augmentation = augmentation
        
    
    def __getitem__(self, item):
        """Get the datapoint indexed by item

        Returns
        -------
        tuple
            datapoint
        """
        nx_graph = self.networkx_graphs[item]
        di_nx_graph = nx_graph.to_directed()
        di_nx_graph = add_edge_attribute(di_nx_graph) # add edge information to the edge features 
        di_nx_graph = add_node_attribute(di_nx_graph) # add node information to the node features 
        di_nx_graph = normalize_coordinate(di_nx_graph) # normalizing node position

        if self.augmentation:
            if random.uniform(0,1) < 0.25:
                di_nx_graph = random_walk_augmentation (di_nx_graph, num_steps=12)
                if di_nx_graph.number_of_nodes() <= 5:
                    di_nx_graph = random_walk_augmentation (di_nx_graph, num_steps=12)
                    
        g = dgl.from_networkx(di_nx_graph,       
                              node_attrs=None,
                              edge_attrs=None)
        
        node_labels = []
        for k, v in di_nx_graph.nodes(data=True):
            node_labels.append(v['label'])
        g.ndata['label'] = torch.from_numpy(self.LabEnc.transform(node_labels)).to(torch.long)
        
        #To merge selected node attributes into a new attribute 'merged_feats'
        merged_node_feats = []
        for key, feats in di_nx_graph.nodes(data=True):
            merged_node_feat = []
            for feat in feats:
                if feat in self.selected_node_feats and (feat != 'label'):
                    merged_node_feat.extend(feats[feat])
            merged_node_feats.append(merged_node_feat)
        merged_node_feats = np.array(merged_node_feats)
        merged_node_feats = torch.from_numpy(merged_node_feats).to(torch.float32)   
        merged_node_feats = (merged_node_feats - merged_node_feats.mean(axis=0)) / (merged_node_feats.std(axis=0) + 1e-8)
        g.ndata['merged_feats'] = merged_node_feats

        #To merge selected edge attributes into a new attribute 'feats'
        edge_feats = []
        for key in di_nx_graph.edges(data=True):
            edge_feats.append(key[2]['feats'])
        edge_feats = np.array(edge_feats)
        edge_feats = torch.from_numpy(edge_feats).to(torch.float32)  
        edge_feats = (edge_feats - edge_feats.mean(axis=0)) / (edge_feats.std(axis=0) + 1e-8)
        g.edata['feats'] = edge_feats
        g = dgl.add_self_loop(g)
              
        return g
    
    def __len__(self):
        """Get subset size

        Returns
        -------
        int
            Number of datapoints in the subset
        """
        return len(self.networkx_graphs)
    
    
    def __num_classes__(self):
        """Get subset size

        Returns
        -------
        int
            Number of datapoints in the subset
        """
        return len(self.LabEnc.classes_)

    

 
 