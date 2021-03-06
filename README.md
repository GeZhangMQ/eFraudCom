# eFraudCom: An E-commerce Fraud Detection System via Competitive Graph Neural Networks
**Overview** 

This repository is PyTorch implementation of Competitive Graph Neural Network (CGNN) proposed in 

"eFraudCom: An E-commerce Fraud Detection System via Competitive Graph Neural Networks".

## 1. Requirements

* ```numpy``` == 1.19.5

* ```torch``` == 1.6.0

* ```scikit-learn``` == 0.23.2

* ```scipy``` == 1.4.1

## 2. MOOC student drop-out

**2.1 Data**

In the dataset MOOC student drop-out, we regard students as users and actions as items;

* ```action_features.mat``` : the attributes of edges, the last column is the labels of edges;

* ```bipartitie_action.mat``` : the bipartitie graph of students and actions;

* ```item_features_matrix.mat``` : the attrbutes of actions;

* ```user_features_matrix.mat``` : the attrbutes of students;

* ```edge_item_features_matrix.mat``` : the concatenation of attributes of edges and corresponding actions;

* ```edge_item_features_matrix.mat``` : the concatenation of attributes of edges and corresponding students.

**2.2 Structure**

* ```MOOC student dropout/data/new_mooc.mat```: the dataset which contains ```action_features.mat``` ```bipartitie_action.mat``` ```item_features_matrix.mat``` ```user_features_matrix.mat``` ```edge_item_features_matrix.mat``` ```edge_item_features_matrix.mat```

* ```MOOC student dropout/main.py```: training the model and training options; 

* ```MOOC student dropout/model.py```:  CGNN implementaions;

* ```MOOC student dropout/preprocess.py```: utils;

* ```MOOC student dropout/dgi.py```: CGNN implementaions;


**2.3 Run**

To train the model, run ```MOOC student dropout/main.py```

## 3. Bitcoin-Alpha



**3.1 Data**

```Bitcoin-Alpha/data/alpha/alpha_graph_u2u.pickle```: the pickled sparse adjacency matrix about users;

```Bitcoin-Alpha/data/alpha/alpha_graph_u2p.pickle```: the pickled sparse adjacency matrix about users and items;

```Bitcoin-Alpha/data/alpha/alpha_labels.pickle```: the pickled user labels.

**3.2 Structure**

* ```Bitcoin-Alpha/aggregators.py```: the convolution layers in GraphSAGE implementations;

* ```Bitcoin-Alpha/encoders.py```: CGNN implementations;

* ```Bitcoin-Alpha/model.py```: CGNN implementations and training the model.


**3.3 Run**

To train the model, run ```Bitcion-Alpha/model.py```
