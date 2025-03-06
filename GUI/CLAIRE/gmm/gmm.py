import torch
import sys
sys.path.append('/root/CLAIRE')
from dev.utils.utils import *
from sklearn import mixture
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(512, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.ln4 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.ln5 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln4(self.fc4(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln5(self.fc5(x)))
        x = torch.relu(x)

        x = self.fc3(x)
        return x

def get_cluster_center(train_data, train_labels):
    cluster_center_model = {}
    # id_counter = 0
    with torch.no_grad():
        for ec in list(set(train_labels)):
            emb_cluster = []
            for ind, item in enumerate(train_labels):
                if item == ec:
                    emb_cluster.append(train_data[ind])
            emb_cluster = torch.cat(emb_cluster)
            cluster_center = emb_cluster.mean(dim=0)
            cluster_center_model[ec] = cluster_center.detach().cpu()
    return cluster_center_model

def mine_negative_random(anchor_ec, esm_emb):
    result_ec = random.choices(list(esm_emb.keys()))[0]

    while result_ec == anchor_ec:
        result_ec = random.choices(list(esm_emb.keys()))[0]
    
    return result_ec

def get_dist(ec, train_data,train_labels, esm_emb, pretrained_model = None, neg_target = 100):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    model = LayerNormNet(1280, 128, device, dtype)   

    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint)
    model.eval()

    emb_train = model(torch.tensor(train_data).to(device=device, dtype=dtype))

    esm_emb_test = esm_emb[ec]
    emb_test = [i.unsqueeze(0)  for i in esm_emb_test]

    emb_test = model(torch.cat(emb_test, dim=0).to(device=device, dtype=dtype))

    ec_centers = get_cluster_center(emb_train, train_labels)

    neg_emb_list = []
    for i in range(neg_target):
        neg_ec = mine_negative_random(ec, esm_emb)
        neg = random.choice(esm_emb[neg_ec]).unsqueeze(0)
        neg_emb_list.append(neg)

    neg_emb_test = model(torch.cat(neg_emb_list, dim=0).to(device=device, dtype=dtype))

    distances = []
    for i in range(len(emb_test)):
        dist = (emb_test[i] - ec_centers[ec].to(device)).norm(dim = 0, p = 2).detach().cpu().numpy().item()
        distances.append(dist)
        
    neg_distances = []
    for i in range(len(neg_emb_test)):
        dist = (neg_emb_test[i] - ec_centers[ec].to(device)).norm(dim = 0, p = 2).detach().cpu().numpy().item()
        neg_distances.append(dist)
        
    return distances, neg_distances


esm_emb = pickle.load(open('../data/pred_rxn_EC123/esm_emb_dict_ec3.pkl', 'rb'))

train_file = '../data/model_lookup_train.pkl'
with open (train_file, 'rb') as file:
    train_data = pickle.load(file)

labels_file = '../data/pred_rxn_EC123/labels_train_ec3.pkl'
with open (labels_file, 'rb') as file:
    train_labels = pickle.load(file)

pretrained_model = '../results/model/pred_rxn_EC123/layer5_node1280_triplet2000_final.pth'

main_GMM_list = []
# counter = 0
for i in range(40):
    all_distance = []
    for ec in random.choices(list(esm_emb.keys()), k = 500):

        distances, neg_distances = get_dist(ec, train_data,train_labels, esm_emb, 
                 pretrained_model=pretrained_model, neg_target = 100)

        all_distance.extend(neg_distances)
        all_distance.extend(distances)

    dist = np.reshape(all_distance, (len(all_distance), 1))
    main_GMM = mixture.GaussianMixture(n_components=2, covariance_type='full',max_iter=1000,n_init=30,tol=1e-4)
    main_GMM.fit(dist)
    main_GMM_list.append(main_GMM)

    plt.hist(all_distance, bins = 500, alpha = 0.5)
    # plt.savefig('GMM_100_500_' + str(i) + '.png')
pickle.dump(main_GMM_list, open('GMM.pkl', 'wb'))
