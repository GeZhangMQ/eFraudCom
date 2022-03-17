import os
import pickle
import json
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim
from torch.nn import init
from torch.autograd import Variable
from aggregators import MeanAggregator
from encoders import Encoder
from sklearn.metrics import roc_auc_score
import pandas as pd

"""
    Paper: eFraudCom: An E-commerce Fraud Detection System via Competitive Graph Neural Networks

"""

class InlierGraphDecoding(nn.Module):

	def __init__(self):
		super(InlierGraphDecoding,self).__init__()

		self.fc1 = nn.Linear(16,32)
		self.fc2 = nn.Linear(32,64)
		self.fc3 = nn.Linear(64,128)
		self.fc4 = nn.Linear(128,148)

	def forward(self,x):

		h = torch.relu(self.fc1(x))
		h = torch.relu(self.fc2(h))
		h = torch.relu(self.fc3(h))
		return self.fc4(h)

class OutlierGraphDecoding(nn.Module):

	def __init__(self):
		super(OutlierGraphDecoding,self).__init__()

		self.fc1 = nn.Linear(16,32)
		self.fc2 = nn.Linear(32,64)
		self.fc3 = nn.Linear(64,128)
		self.fc4 = nn.Linear(128,148)

	def forward(self,x):

		h = torch.relu(self.fc1(x))
		h = torch.relu(self.fc2(h))
		h = torch.relu(self.fc3(h))
		return self.fc4(h)

class GraphEncoding(nn.Module):

	def __init__(self,enc):
		super(GraphEncoding,self).__init__()
		self.enc = enc
		self.weight = nn.Parameter(torch.FloatTensor(16, self.enc.embed_dim))
		init.xavier_uniform(self.weight)

	def forward(self, nodes):
		embeds = self.enc(nodes)
		scores = self.weight.mm(embeds)
		return scores.t()


def load_data():

	config_dir = './configs'
	file_paths = 'file_paths.json'
	dataSet = 'bitcoinalpha'
	file_paths = json.load(open(f'{config_dir}/{file_paths}'))

	ds = dataSet
	graph_u2u_file = file_paths[ds]['graph_u2u']
	labels_file = file_paths[ds]['labels']
	features_file = file_paths[ds]['features']

	graph_u2u = pickle.load(open(graph_u2u_file,'rb'))
	labels = pickle.load(open(labels_file,'rb'))
	feat_data = pickle.load(open(features_file,'rb'))

	adj_lists = {}
	for i in range(np.shape(graph_u2u)[0]):
		adj_lists[i] = set(graph_u2u[i,:].nonzero()[1])
	best_adj_lists = {}
	for i in range(np.shape(graph_u2u)[0]):
		if len(adj_lists[i]) <= 15:
			best_adj_lists[i] = adj_lists[i]
		else:
			adjs=graph_u2u[i].toarray()[0]
			best_adj_lists[i] = set(np.argpartition(adjs,-15)[-15:])
	return feat_data,labels,best_adj_lists

criterion = torch.nn.MSELoss(reduction = 'mean')
criterion_row = torch.nn.MSELoss(reduction = 'none')

def run_():
	
	epoches = 100
	np.random.seed(1)
	random.seed(1)
	torch.manual_seed(1)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	feat_data,labels,adj_lists = load_data()
	labels_ = labels.reshape(3275,1)
	feat_data_ = np.concatenate((feat_data,labels_), axis=1) 
	features = nn.Embedding(3275,148)
	features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

	agg1 = MeanAggregator(features, cuda = False)
	enc1 = Encoder(features, 148, 64, adj_lists, agg1, gcn = True, cuda = False)

	agg2 = MeanAggregator(lambda nodes:enc1(nodes).t(), cuda = False)
	enc2 = Encoder(lambda nodes:enc1(nodes).t(), enc1.embed_dim, 32, adj_lists, agg2, base_model=enc1, gcn=True, cuda=False)
	

	encoder = GraphEncoding(enc2)
	Inlier_decoder = InlierGraphDecoding()
	Outlier_decoder = OutlierGraphDecoding()

	optimizer1 = torch.optim.SGD(encoder.parameters(), lr = 0.001)
	optimizer2 = torch.optim.SGD(Inlier_decoder.parameters(), lr = 0.001)
	optimizer3 = torch.optim.SGD(Outlier_decoder.parameters(), lr = 0.001)

	train=[i for i in range(3275)]

	for epoch in range(epoches):

		batch_nodes = train

		feat_data_0 = feat_data_[batch_nodes]

		normal_location = np.where(feat_data_0[:,-1] == 0)[0].tolist()
		other_location=np.where(feat_data_0[:,-1] != 0)[0].tolist()

		embs = encoder(batch_nodes)

		recon_normal_feats = Inlier_decoder(embs[normal_location])
		recon_other_feats_in = Inlier_decoder(embs[other_location])
		recon_other_feats_out = Outlier_decoder(embs[other_location])

		normal_feats = torch.FloatTensor(feat_data_0[normal_location,0:148])
		loss_normal = criterion(recon_normal_feats,normal_feats)
		
		other_feats = torch.FloatTensor(feat_data_0[other_location,0:148])
		inlier_loss_other = criterion_row(recon_other_feats_in,other_feats).mean(axis = 1)
		outlier_loss_other = criterion_row(recon_other_feats_out,other_feats).mean(axis = 1)

		pred_labels = np.where(inlier_loss_other < outlier_loss_other,0,1)

		loss_B = torch.mul(torch.FloatTensor(1 - pred_labels),inlier_loss_other).mean()
		loss_C = torch.mul(torch.FloatTensor(pred_labels),outlier_loss_other).mean()

		loss = 10 * loss_normal + loss_B + loss_C

		print('epoch:{} batch:{} --> loss={}'.format(epoch, 0, loss))

		optimizer3.zero_grad()
		optimizer2.zero_grad()
		optimizer1.zero_grad()

		loss.backward()

		optimizer3.step()
		optimizer2.step()
		optimizer1.step()
		
		random.shuffle(train)


	#testing
	encoder.eval()
	Inlier_decoder.eval()
	Outlier_decoder.eval()

	index = np.where(feat_data_[:,-1] != -1)[0].tolist()

	feats = torch.FloatTensor(feat_data_[index,0:148])

	nodes = [i for i in range(214)]

	embs = encoder(index)

	recon_feats_in = Inlier_decoder(embs)
	recon_feats_out = Outlier_decoder(embs)

	inlier_loss = criterion_row(recon_feats_in,feats).mean(axis=1)
	outlier_loss = criterion_row(recon_feats_out,feats).mean(axis=1)

	pred_labels = np.where(inlier_loss < outlier_loss, 0.,1.).squeeze()

	labels = feat_data_[index,-1].squeeze()

	fraud_score = (inlier_loss / (inlier_loss + outlier_loss)).detach().numpy()

	auc = roc_auc_score(labels,fraud_score)
	print('ROC AUC score:{:0.4f}'.format(auc))

	# FOR OUTPUT
	#df = pd.DataFrame({'AD-GCA':fraud_score})
	#df.to_csv('output/{}-scores.csv'.format("bitcoin"), index=False, sep=',')
	#df2 = pd.DataFrame({'gnd':labels})
	#df2.to_csv('output/{}-gnd.csv'.format("bitcoin"), index=False, sep=',')

if __name__ == "__main__":
	run_()
