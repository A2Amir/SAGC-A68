#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dgl
import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATv2Conv

class GATLayer(nn.Module):
    def __init__(self, n_in_feats, n_edge_feats, n_out_feats, n_heads, merge):
        super(GATLayer, self).__init__()
        
        self.n_in_feats = n_in_feats
        self.n_edge_feats = n_edge_feats
        self.n_out_feats = n_out_feats
        self.n_heads = n_heads
        self.merge = merge

        self.attn = GATv2Conv(self.n_in_feats, self.n_heads * self.n_out_feats,
                              num_heads=self.n_heads, feat_drop=0.1, attn_drop=0.1,
                              residual= True, activation=F.relu, share_weights=True)
        self.edge_fc = nn.Linear(self.n_edge_feats, self.n_heads * self.n_out_feats)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
                               
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['m'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'h_new': h}
    
    
    def forward(self, g, node_feats, edge_feats):
        
        # Apply attention mechanism
        h = self.attn(g, node_feats)
        
        # Compute edge weights using a linear layer
        e = self.edge_fc(edge_feats)

        
        # Compute weighted sum of neighbors' features using the attention mechanism
        g.ndata['h'] = h
        g.edata['e'] = e
        #print(h.shape, e.shape)
        #print(fn.u_add_e('h', 'e', 'm').shape)
        g.update_all(fn.u_add_e('h', 'e', 'm'), self.reduce_func )
        h_new = g.ndata['h_new']
        
        
        if self.merge == 'concat':
            # merge using average
            h_new =  h_new.view(-1,  self.n_heads * self.n_heads * self.n_out_feats)
            
        elif self.merge == 'sum':
            # merge using average
            h_new =  torch.sum(h_new, dim=1) 
            
        elif self.merge == 'mean':
            # merge using average
            h_new =  torch.mean(h_new, dim=1) 
            
        elif self.merge == 'maxpool':
            # merge using MaxPool
            h_new =  F.adaptive_max_pool2d(h_new, (1, self.n_heads * self.n_out_feats))
            h_new = h_new.view(-1, self.n_heads * self.n_out_feats)
            
        return h_new
    
    

class GAT(nn.Module):
    def __init__(self, n_in_feats, n_edge_feats, n_out_feats, n_heads, merge):
        super(GAT, self).__init__()
        
        
        # Be aware that the output dimension is n_out_feats*num_heads 
        # In case of concat merging is n_out_feats*num_head*num_head since multiple head outputs are concatenated together.         
        n_head_pool = [n_head * n_head if merge == 'concat' else n_head for n_head in n_heads]

        self.layer1 = GATLayer(n_in_feats, n_edge_feats, n_out_feats , n_heads[0], merge)
        self.BatchNorm1 = torch.nn.BatchNorm1d(n_head_pool[0] * n_out_feats)
        
        self.layer2 = GATLayer(n_head_pool[0] * n_out_feats  , n_edge_feats, n_out_feats*2, n_heads[1], merge)
        self.BatchNorm2 = torch.nn.BatchNorm1d(n_head_pool[1] * n_out_feats*2)
        
        self.layer3 = GATLayer(n_head_pool[1] * n_out_feats*2  , n_edge_feats, n_out_feats, n_heads[2], merge)

        
    def forward(self, g, nfeats, efeats):
        
        h1 = self.layer1(g, nfeats, efeats)
        h1 = self.BatchNorm1(h1)
        h1 = F.leaky_relu(h1)
        
        h2 = self.layer2(g, h1, efeats)
        h2 = self.BatchNorm2(h2)
        h2 = F.leaky_relu(h2)
                
        h3 = self.layer3(g, h2, efeats)
         
        return h3, h2
    
    
    
    
