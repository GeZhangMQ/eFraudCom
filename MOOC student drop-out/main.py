import numpy as np
from scipy.io import loadmat
import scipy.io as io
from collections import defaultdict
import random
import pandas as pd
from preprocess import load_data
import argparse
import torch
from model import *
from sklearn.metrics import roc_auc_score
import pandas as pd
from dgi import *

"""
    Paper: eFraudCom: An E-commerce Fraud Detection System via Competitive Graph Neural Networks

"""

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type = int, default = 40, help = "number of epoches")
parser.add_argument("--lr", type = float, default = 0.001, help = "learning rate")
parser.add_argument("--batch_size", type = int, default = 200, help = "batch_size")
parser.add_argument("--hot1", type = int, default = 4, help = "1-hop neighbors")
parser.add_argument("--hot2", type = int, default = 4, help = "2-hop neighbors")
args=parser.parse_args()

#setting seed
seed = 7
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

#loading data
mooc = loadmat('data/new_mooc.mat')
relation = mooc['bipartite_action'].astype(np.int)
edge_features = mooc['action_features'] 
item_features = mooc['item_features_matrix'] 
user_features = mooc['user_features_matrix'] 
edge_user_features = mooc['edge_user_features_matrix'] 
edge_item_features = mooc['edge_item_features_matrix'] 

def get_dict():

	# user_dict
	# { key: user_id value:[edge_id] }
	
	sorted_by_user_data = relation[np.argsort(relation[:,1])]

	user_dict = defaultdict(list)

	for i in range(0,sorted_by_user_data.shape[0]):

		user_dict[sorted_by_user_data[i,1]].append(sorted_by_user_data[i,0])
	
	#item_dict
	#{ key: item_id value:[edge_id] }
	
	sorted_by_item_data = relation[np.argsort(relation[:,2])]

	item_dict = defaultdict(list)

	for i in range(0,sorted_by_user_data.shape[0]):

		item_dict[sorted_by_item_data[i,2]].append(sorted_by_item_data[i,0])

	return user_dict, item_dict

def get_neighbors(edge_idx, batch_size, user_dict, item_dict, hot1, hot2):

	#[src] <-- [dst] <-- [src] <-- [[edge]] --> [dst] --> [src] --> [dst]
	
	#relation:  action_idx  src_idx dst_idx timestampe labels
	src_idx = relation[edge_idx,1].tolist()
	dst_idx = relation[edge_idx,2].tolist()

	src_feats = user_features[src_idx] # features of src node
	dst_feats = item_features[dst_idx] # featyres of dst node


	# Sampling the 1-hop neighbors for src nodes
	src_l1_idx_list = []
	for i,idx in enumerate(src_idx):
		
		flag=False

		if len(user_dict[idx]) < hot1:
			flag=True

		tmp=np.random.choice(user_dict[idx], args.hot1, replace = flag).tolist()
		src_l1_idx_list.extend(tmp)

	src_l1_feats = edge_item_features[src_l1_idx_list]     
	src_l1_feats = src_l1_feats.reshape(batch_size,hot1,-1)
	src_l1_node_idx_list = relation[src_l1_idx_list,2].tolist()

	#Sampling the 2-hop neighbors for scr nodes

	src_l2_idx_list = []
	for i,idx in enumerate(src_l1_node_idx_list):

		flag = False

		if len(item_dict[idx]) < hot2:
			flag = True

		tmp = np.random.choice(item_dict[idx], args.hot2, replace = flag).tolist()
		src_l2_idx_list.extend(tmp)

	src_l2_feats = edge_user_features[src_l2_idx_list]
	src_l2_feats = src_l2_feats.reshape(batch_size, hot1, hot2, -1)


	#Sampling the 1-hop neighbors for dst nodes
	dst_l1_idx_list = []

	for i,idx in enumerate(dst_idx):

		flag = False

		if len(item_dict[idx]) < hot1:

			flag = True

		tmp = np.random.choice(item_dict[idx], args.hot1, replace = flag).tolist()
		dst_l1_idx_list.extend(tmp)

	dst_l1_feats = edge_user_features[dst_l1_idx_list]
	dst_l1_feats = dst_l1_feats.reshape(batch_size,hot1,-1)
	dst_l1_node_idx_list = relation[dst_l1_idx_list,1].tolist()

	#Sampling the 2-hop neighbors for dst nodes
	dst_l2_idx_list = []

	for i,idx in enumerate(dst_l1_node_idx_list):

		flag = False

		if len(user_dict[idx]) < hot2:
			flag = True

		tmp = np.random.choice(user_dict[idx], args.hot2, replace = flag).tolist()
		dst_l2_idx_list.extend(tmp)

	dst_l2_feats = edge_item_features[dst_l2_idx_list]
	dst_l2_feats = dst_l2_feats.reshape(batch_size,hot1,hot2,-1)

	src_feats = torch.FloatTensor(src_feats)
	dst_feats = torch.FloatTensor(dst_feats)

	src_l1_feats = torch.FloatTensor(src_l1_feats)
	src_l2_feats = torch.FloatTensor(src_l2_feats)

	dst_l1_feats = torch.FloatTensor(dst_l1_feats)
	dst_l2_feats = torch.FloatTensor(dst_l2_feats)


	return src_feats, dst_feats, src_l1_feats, src_l2_feats, dst_l1_feats, dst_l2_feats



encoder_part = Encoder(src_dim = 27, 
    dst_dim = 4,
    edge_dim = 4,
    src_hidden_dim1 = 16,
    dst_hidden_dim1 = 4,
    hidden_dim2 = 8,
    dropout = 0.0
    )

regu = DGI_(100,200)

#decoder
Inlier_decoder = Inlier_Decoder(input_dim = 8 * 3, dim1 = 16, dim2 = 8, output_dim = 4, dropout = 0.0)
Outlier_decoder = Outlier_Decoder(input_dim = 8 * 3, dim1 = 16, dim2 = 8, output_dim = 4, dropout = 0.0)


criterion = t.nn.MSELoss(reduction = 'mean')    
criterion_row = t.nn.MSELoss(reduction = 'none')

optimizer1 = torch.optim.SGD(encoder_part.parameters(),lr = args.lr)
optimizer2 = torch.optim.SGD(Inlier_decoder.parameters(),lr = args.lr)
optimizer3 = torch.optim.SGD(Outlier_decoder.parameters(),lr = args.lr)


def run_(train_loader, test_loader, user_dict, item_dict):

	regu1 = DGI_(4,24)
	b_xent = torch.nn.BCEWithLogitsLoss()
	optimizer4 = torch.optim.Adam(regu1.parameters(),lr = args.lr)

	encoder_part.train()
	Inlier_decoder.train()
	Outlier_decoder.train()
	regu1.train()

	hot1 = args.hot1
	hot2 = args.hot2

	train_idx = train_loader['idx']
	feats_data_ = train_loader['feats']

	for epoch in range(args.num_epochs):


		#shuffle
		random.shuffle(train_idx)
		num_batches = int(len(train_idx)/args.batch_size)


		for batch in range(num_batches):

			i_start = batch * args.batch_size
			i_end = min((batch + 1) * args.batch_size, len(train_idx))
			batch_nodes = train_idx[i_start:i_end]
			batch_size = len(batch_nodes)

			feat_data_0 = feats_data_[batch_nodes]

			normal_location = np.where(feat_data_0[:,-1] == 0)[0].tolist()
			other_location = np.where(feat_data_0[:,-1] != 0)[0].tolist()

			feat_data_0 = torch.FloatTensor(feat_data_0)


			src_feats, dst_feats, src_l1_feats, src_l2_feats, dst_l1_feats, dst_l2_feats = get_neighbors(batch_nodes, batch_size, user_dict, item_dict, hot1, hot2)

			embs = encoder_part(feat_data_0[:,0:4], src_feats, src_l1_feats, src_l2_feats, dst_feats, dst_l1_feats,dst_l2_feats)

			recon_normal_feats = Inlier_decoder(embs[normal_location])
			recon_other_feats_in = Inlier_decoder(embs[other_location])
			recon_other_feats_out = Outlier_decoder(embs[other_location])

			normal_feats = torch.FloatTensor(feat_data_0[normal_location, 0:4])
			loss_normal = criterion(recon_normal_feats, normal_feats)

			other_feats = torch.FloatTensor(feat_data_0[other_location, 0:4])
			inlier_loss_other = criterion_row(recon_other_feats_in, other_feats).mean(axis=1)
			outlier_loss_other = criterion_row(recon_other_feats_out, other_feats).mean(axis=1)


			pred_labels = np.where(inlier_loss_other < outlier_loss_other, 0, 1)
			loss_B = torch.mul(torch.FloatTensor(1 - pred_labels), inlier_loss_other).mean()
			loss_C = torch.mul(torch.FloatTensor(pred_labels), outlier_loss_other).mean()

			other_index_inlier = np.where(inlier_loss_other < outlier_loss_other)[0].tolist()
			other_index_outlier = np.where(inlier_loss_other >= outlier_loss_other)[0].tolist()
			h = embs[other_index_outlier,:]
			seq1 = torch.FloatTensor(feat_data_0[other_index_outlier, 0:4])
			seq2_1 = torch.FloatTensor(feat_data_0[other_index_inlier, 0:4])
			seq2 = torch.cat((normal_feats, seq2_1), 0)
			lbl_1 = torch.ones(1, seq1.shape[0])
			lbl_2 = torch.zeros(1, seq2.shape[0])
			lbl = torch.cat((lbl_1,lbl_2),1)
			logits = regu1(h, seq1, seq2, None, None, None)
			loss_D = b_xent(logits, lbl)
			loss = loss_normal + loss_B + loss_C + 0.1 * loss_D
			print('epoch:{} batch:{} --> loss={}'.format(epoch,0,loss))
			optimizer4.zero_grad()
			optimizer3.zero_grad()
			optimizer2.zero_grad()
			optimizer1.zero_grad()
			loss.backward()
			optimizer3.step()
			optimizer2.step()
			optimizer4.step()
			optimizer1.step()
		
def test_(test_loader, user_dict, item_dict):


	hot1 = args.hot1
	hot2 = args.hot2


	encoder_part.eval()
	Inlier_decoder.eval()
	Outlier_decoder.eval()
	test_idx = test_loader['idx']
	feats = test_loader['feats'][:,0:4]
	labels = test_loader['labels']
	batch_size = len(labels)
	src_feats, dst_feats, src_l1_feats, src_l2_feats, dst_l1_feats, dst_l2_feats = get_neighbors(test_idx, batch_size, user_dict, item_dict, hot1, hot2)
	embs = encoder_part(feats[:,0:4], src_feats, src_l1_feats, src_l2_feats, dst_feats, dst_l1_feats, dst_l2_feats)
	recon_feats_in = Inlier_decoder(embs)
	recon_feats_out = Outlier_decoder(embs)
	inlier_loss = criterion_row(recon_feats_in, feats).mean(axis=1)
	outlier_loss = criterion_row(recon_feats_out, feats).mean(axis=1)

	pred_labels = np.where(inlier_loss < outlier_loss, 0., 1.).squeeze()
	fraud_score = (inlier_loss / (inlier_loss + outlier_loss)).detach().numpy()

	auc = roc_auc_score(labels,fraud_score)
	print('ROC AUC score:{:0.4f}'.format(auc))


	#FOR OUTPUT
	#df = pd.DataFrame({'AD-GCA':fraud_score})
	#df.to_csv('output/{}-scores.csv'.format("mooc"), index=False, sep=',')
	#df2 = pd.DataFrame({'gnd':labels})
	#df2.to_csv('output/{}-gnd.csv'.format("mooc"), index=False, sep=',')

	


if __name__=='__main__':

	train_loader,test_loader = load_data()
	user_dict, item_dict = get_dict()

	run_(train_loader, test_loader, user_dict, item_dict)
	test_(test_loader, user_dict, item_dict)

