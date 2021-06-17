import torch
import torch.nn as nn
import random

class MeanAggregator(nn.Module):

	# Code partially from https://github.com/williamleif/graphsage-simple/

	def __init__(self, features, cuda = False, gcn = False):

		super(MeanAggregator, self).__init__()

		self.features = features
		self.gcn = gcn

	def forward(self, nodes, to_neighs, num_sample = 10):

		'''
		nodes -- list of nodes in a batch
		to_neighs -- list of sets, each set is the set of neighbors for node in batch
		num sumple --number of neighbors to sample
		'''

		_set = set
		if not num_sample is None:
			_sample = random.sample 
			samp_neighs = [_set(_sample(to_neigh,num_sample,)) if len(to_neigh)>=num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs


		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes={n:i for i,n in enumerate(unique_nodes_list)}

		mask = torch.zeros(len(samp_neighs),len(unique_nodes))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		
		mask[row_indices,column_indices] = 1

		if self.cuda:
			mask = mask.cuda()

		num_neigh = mask.sum(1,keepdim=True)
		mask = mask.div(num_neigh)

		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats
