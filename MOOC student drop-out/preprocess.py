import torch
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def load_data():

    mooc = loadmat('data/new_mooc.mat')
    data = mooc['action_features']
    features = data 
    labels = data[:,-1] 
    index = list(range(len(labels)))
    # spliting train and test data
    idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify = labels, test_size = 0.60, random_state = 2, shuffle = True)

    train = {}
    test = {}

    features=torch.Tensor(features)

    train['feats'] = features
    train['labels'] = labels
    train['idx'] = idx_train

    test['idx'] = idx_test
    test['feats'] = features[idx_test]
    test['labels'] = labels[idx_test]

    return train, test