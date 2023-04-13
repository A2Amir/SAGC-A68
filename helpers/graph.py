#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random


from scipy.spatial.distance import euclidean


    
def visulaize_graph(nx_graph, save_path, labels):
    
    positions = nx.get_node_attributes(nx_graph,'pos')
    pos = {}
    for k, v in positions.items():
        pos[k] = v[:2]
    nx.draw_networkx_nodes(nx_graph, pos, node_size=40, node_color="blue")


    # edges
    edges = [(u, v) for (u, v, d) in nx_graph.edges(data=True)]
    nx.draw_networkx_edges(nx_graph, pos, edgelist=edges, width=1, edge_color="Cyan")

    
    font = {'fontname':'Arial',
            'size':'10',
            'color':'black',
            'weight':'normal'
           }
    
    if labels:
        angle = 45
        for n in nx_graph.nodes():
            t = plt.text(nx_graph.nodes[n]['pos'][0]+.5, nx_graph.nodes[n]['pos'][1]+.5, nx_graph.nodes[n]['label'],
                         rotation=angle, ha='center', va='center', fontdict=font)
            nx_graph.nodes[n]['text'] = t
     
    #if edges:
    #    edge_labels = nx.get_edge_attributes(nx_graph, "key")
    #    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels, font_size=eval(font['size']), font_family="sans-serif")
        

    ax = plt.gca()
    plt.axis("off")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.json', '.svg'))
        
    plt.show()
    
def dtype_transformation(value):
    if value == 'nan':
         return float(-1) 
    elif value =='True':
        return float(1)
    elif value =='False':
        return float(0) 
    else:
        return float(value)

def add_edge_attribute(di_nx_graph):
    
    edge_centrality = nx.edge_betweenness_centrality(di_nx_graph)
    for u, v in di_nx_graph.edges:
        di_nx_graph.edges[(u, v)]['feats'].append(edge_centrality[(u, v)])
    
    for u, v in di_nx_graph.edges:

        source_node_pos = di_nx_graph.nodes[u]['pos']
        target_node_pos = di_nx_graph.nodes[v]['pos']
        vec = np.array(target_node_pos) - np.array(source_node_pos)

        # Compute the angle between the two vectors
        angle = np.arccos(np.dot(vec, np.array([1, 0, 0])) / np.linalg.norm(vec)) * 180 / np.pi
        di_nx_graph.edges[(u, v)]['feats'].append(angle)
    
    return di_nx_graph


def add_node_attribute(di_nx_graph):
    
    for n in di_nx_graph.nodes:
        di_nx_graph.nodes[n]['feats'].append(nx.clustering(di_nx_graph, n))
        di_nx_graph.nodes[n]['feats'].append(di_nx_graph.degree[n])     # add node degree as a feature

    bc = nx.betweenness_centrality(di_nx_graph)
    for node, value in bc.items():
        di_nx_graph.nodes[node]['feats'].append(value)

    pr = nx.pagerank(di_nx_graph, alpha=0.9, max_iter=500)
    for node, value in pr.items():
        di_nx_graph.nodes[node]['feats'].append(value)

    close_cen = nx.closeness_centrality(di_nx_graph)
    for node, value in close_cen.items():
        di_nx_graph.nodes[node]['feats'].append(value)
    
    degree_cen = nx.degree_centrality(di_nx_graph)
    for node, value in degree_cen.items():
        di_nx_graph.nodes[node]['feats'].append(value)
    return di_nx_graph





def normalize_coordinate(di_nx_graph):

    x_min, y_min, z_min, x_max, y_max, z_max = di_nx_graph.graph['graph_bounding_box']
    for node in di_nx_graph.nodes(data=True):
        node_id = node[0]
        nodel_feat = node[1]
        x = nodel_feat['pos'][0]
        y = nodel_feat['pos'][1]
        z = nodel_feat['pos'][2]
        
        relative_x = (x - x_min) / (x_max - x_min)
        relative_y = (y - y_min) / (y_max - y_min)
        relative_z = (z - z_min) / (z_max - z_min)
        
        di_nx_graph.nodes[node_id]['pos']  = [relative_x, relative_y, relative_z]
        
    return di_nx_graph



def random_walk_augmentation (di_nx_g, num_steps=10):
    g = di_nx_g
    start_node = random.choice(list(g.nodes()))

    # number of steps to perform in a random walk
    num_steps = num_steps

    # Perform the random walk and create a subgraph
    subgraph = nx.DiGraph()
    for i in range(num_steps):
        neighbors = list(g.neighbors(start_node))
        if len(neighbors) == 0:
            break
        next_node = random.choice(neighbors)
        subgraph.add_node(start_node, **g.nodes[start_node])
        subgraph.add_node(next_node, **g.nodes[next_node])
        subgraph.add_edge(start_node, next_node, **g[start_node][next_node])
        subgraph.add_edge(next_node, start_node, **g[next_node][start_node])
        start_node = next_node
    
    #plt.figure(1)
    #pos=nx.get_node_attributes(g,'pos')
    #nx.draw(g, with_labels = True, font_size=6, pos =pos, font_family="sans-serif", node_size=6) 
    
    #plt.figure(2)
    #subgraph_pos=nx.get_node_attributes(subgraph,'pos')
    #nx.draw(subgraph, with_labels = True, pos=subgraph_pos, font_size=6,  node_size=5, font_family="sans-serif")
    
    #plt.figure(3)
    #di_subgraph =subgraph.to_directed()
    #di_subgraph_pos=nx.get_node_attributes(di_subgraph,'pos')
    #nx.draw(di_subgraph, with_labels = True, pos=di_subgraph_pos, font_size=6,  node_size=5, font_family="sans-serif")
    
    return subgraph


def check_graph_connection(nx_graph, source_node = '12986647'):
    
    print('Graph name is {}'.format(nx_graph.name))
    source_node_label = nx_graph.nodes[source_node]['label'].split('_')[-1]
    target_nodes = [v for u,v in nx_graph.edges if u == source_node]
    for t_node in target_nodes:
        target_node_label = nx_graph.nodes[t_node]['label'].split('_')[-1]
        print('Node ID {} with {} label, has a connection to Node ID {} with {} label.'.format(source_node, source_node_label,
                                                                                               t_node, target_node_label))
        
        
















