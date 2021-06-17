
import numpy as np
import random
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class MeanAggregator(nn.Module):

    # aggregating neighbors

    def __init__(self,input_dim,output_dim,use_bias=False):

        super(MeanAggregator,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(t.FloatTensor(self.input_dim,self.output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(t.FloatTensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self,features):

        features_hidden = t.matmul(features,self.weight)

        if self.use_bias:
            features_hidden += self.bias

        return features_hidden

class SageGCN(nn.Module):

    # concating center nodes and their neighbors

    def __init__(self,src_dim,nei_dim,output_dim,drop_out=0.5,use_bias=False):

        super(SageGCN,self).__init__()

        self.src_dim = src_dim
        self.nei_dim = nei_dim
        self.output_dim = output_dim // 2
        self.drop_out = drop_out
        self.aggregator = MeanAggregator(self.nei_dim, self.output_dim)
        self.weight = nn.Parameter(t.FloatTensor(self.src_dim, self.output_dim))
        self.use_bias = use_bias

        if use_bias:
            self.bias = nn.Parameter(t.FloatTensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):

        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self,src_features,nei_features):

        nei_features_hidden = self.aggregator(nei_features)
        self_hidden = t.matmul(src_features,self.weight)
        if self.use_bias:
            features_hidden += self.bias


        hidden = t.cat([self_hidden,nei_features_hidden], dim = -1)
        hidden = F.dropout(hidden, self.drop_out, training = self.training)
        hidden = F.relu(hidden)
        return hidden

class GraphSage(nn.Module):

    # two convolution layers, subpart 2 in FIG.4

    def __init__(self, src_dim, dst_dim, edge_dim, src_hidden_dim1, dst_hidden_dim1, hidden_dim2):

        super(GraphSage, self).__init__()  
        self.src_dim = src_dim
        self.dst_dim = dst_dim
        self.edge_dim = edge_dim
        self.src_hidden_dim1 = src_hidden_dim1
        self.dst_hidden_dim1 = dst_hidden_dim1
        self.hidden_dim2 = hidden_dim2


        self.src_agg_1 = SageGCN(self.src_dim, self.dst_dim+self.edge_dim, self.src_hidden_dim1) 
        self.src_agg_2 = SageGCN(self.src_hidden_dim1, self.dst_hidden_dim1, self.hidden_dim2)
        self.dst_agg_1 = SageGCN(self.dst_dim, self.src_dim+self.edge_dim, self.dst_hidden_dim1)
        self.dst_agg_2 = SageGCN(self.dst_hidden_dim1, self.src_hidden_dim1, self.hidden_dim2)

    def forward(self, src_node_feats, src_l1_feats, src_l2_feats, dst_node_feats, dst_l1_feats, dst_l2_feats):

        src_l1_feats_mean = src_l1_feats.mean(dim = 1) #batchsize,hot1,feature_dim
        dst_l1_feats_mean = dst_l1_feats.mean(dim = 1) #batchsize,hot1,feature_dim

        src_feats_agg_1 = self.src_agg_1(src_node_feats,src_l1_feats_mean) #(batch_size,src_dim) (batch_size,hot1,dst+edge_dim)
        dst_feats_agg_1 = self.dst_agg_1(dst_node_feats,dst_l1_feats_mean) #(batch_size,dst_dim) (batch_size,hot1,src+edge_edge)


        src_l2_feats_mean = src_l2_feats.mean(dim = 2) #batchsize,hot1,hot2,feature_dim
        dst_l2_feats_mean = dst_l2_feats.mean(dim = 2) #batchsize,hot1,hot2,feature_dim

        src_l1_feats_agg_1 = self.dst_agg_1(src_l1_feats[:,:,0:4], src_l2_feats_mean)
        dst_l1_feats_agg_1 = self.src_agg_1(dst_l1_feats[:,:,0:27], dst_l2_feats_mean)

        src_l1_feats_agg_1_mean = src_l1_feats_agg_1.mean(dim = 1)
        dst_l1_feats_agg_1_mean = dst_l1_feats_agg_1.mean(dim = 1)


        src_feats_agg_2 = self.src_agg_2(src_feats_agg_1, src_l1_feats_agg_1_mean)
        dst_feats_agg_2 = self.dst_agg_2(dst_feats_agg_1, dst_l1_feats_agg_1_mean)

        return src_feats_agg_2, dst_feats_agg_2

class EdgeFeatsEncoder(nn.Module):

    # subpart 1 in FIG.4

    def __init__(self, input_dim, dim1, output_dim, dropout):

        super(EdgeFeatsEncoder,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim1 = dim1
        self.dropout = dropout

        self.layer1 = nn.Linear(self.input_dim, self.dim1)
        self.layer2 = nn.Linear(self.dim1, self.output_dim)

    def forward(self, edge_feats):

        embs = F.relu(self.layer1(edge_feats))
        embs = self.layer2(embs)

        return embs


class Inlier_Decoder(nn.Module):

    def __init__(self, input_dim, dim1, dim2, output_dim, dropout):

        super(Inlier_Decoder,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dropout = dropout
        self.layer1 = nn.Linear(self.input_dim, self.dim1)
        self.layer2 = nn.Linear(self.dim1, self.dim2)
        self.layer3 = nn.Linear(self.dim2, self.output_dim)

    def forward(self,embeddings):

        feats = F.relu(self.layer1(embeddings))
        feats = F.relu(self.layer2(feats))
        feats = self.layer3(feats)
        return feats

class Outlier_Decoder(nn.Module):

    def __init__(self, input_dim, dim1, dim2, output_dim, dropout):

        super(Outlier_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dropout = dropout
        self.layer1 = nn.Linear(self.input_dim, self.dim1)
        self.layer2 = nn.Linear(self.dim1, self.dim2)
        self.layer3 = nn.Linear(self.dim2, self.output_dim)

    def forward(self, embeddings):

        feats=F.relu(self.layer1(embeddings))
        feats=F.relu(self.layer2(feats))
        feats=self.layer3(feats)
        return feats

class Encoder(nn.Module):

    # concateing embeddings of edges, users, and items


    def __init__(self, src_dim, dst_dim, edge_dim, src_hidden_dim1, dst_hidden_dim1, hidden_dim2, dropout):
        
        super(Encoder, self).__init__()

        self.src_dim = src_dim
        self.dst_dim = dst_dim
        self.edge_dim = edge_dim
        self.src_hidden_dim1 = src_hidden_dim1
        self.dst_hidden_dim1 = dst_hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout

        self.GNN_Encoder = GraphSage(self.src_dim, self.dst_dim, self.edge_dim, self.src_hidden_dim1, self.dst_hidden_dim1, self.hidden_dim2)
        self.Feats_Encoder = EdgeFeatsEncoder(self.edge_dim, 6, 8, self.dropout)

    def forward(self, edge_feats, src_node_feats, src_l1_feats, src_l2_feats, dst_node_feats, dst_l1_feats, dst_l2_feats):
        
        user_embedding, item_embedding=self.GNN_Encoder(src_node_feats, src_l1_feats, src_l2_feats, dst_node_feats, dst_l1_feats, dst_l2_feats)
        edge_feats_embedding = self.Feats_Encoder(edge_feats)
        embs=t.cat([user_embedding,item_embedding,edge_feats_embedding],dim = 1)

        return embs