import numpy as np
import pickle
import csv
import torch
from tqdm import tqdm 

### model_lookup_train
with open('./rxnfp/model_lookup_train.pkl', 'rb') as file:
    rxnfp_rxn_train = pickle.load(file)
with open('./drfp/model_lookup_train.pkl', 'rb') as file:
    drfp_rxn_train = pickle.load(file)

rxn_embeddings = []
for ind, item in enumerate(rxnfp_rxn_train):
    rxn_emb = np.concatenate((np.reshape(item, (1,256)), np.reshape(drfp_rxn_train[ind], (1,256))), axis=1)
    rxn_embeddings.append(rxn_emb)

rxn_embeddings = np.concatenate(rxn_embeddings,axis=0)
print(rxn_embeddings.shape)

with open('../model_lookup_train.pkl', 'wb') as file:
    pickle.dump(rxn_embeddings,file)

### model_lookup_test
with open('./rxnfp/model_lookup_test.pkl', 'rb') as file:
    rxnfp_rxn_test = pickle.load(file)
with open('./drfp/model_lookup_test.pkl', 'rb') as file:
    drfp_rxn_test = pickle.load(file)

rxn_embeddings = []
for ind, item in enumerate(rxnfp_rxn_test):
    rxn_emb = np.concatenate((np.reshape(item, (1,256)), np.reshape(drfp_rxn_test[ind], (1,256))), axis=1)
    rxn_embeddings.append(rxn_emb)

rxn_embeddings = np.concatenate(rxn_embeddings,axis=0)
print(rxn_embeddings.shape)

with open('../model_lookup_test.pkl', 'wb') as file:
    pickle.dump(rxn_embeddings,file)


### esm_emb
rxn_file = '../model_lookup_train.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = '../train_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    samples = list(samples)

model_lookup_train = {}
for ind, item in tqdm(enumerate(samples),total=len(samples)):    
    # 检索 rxn 的 Emb 
    rxn = torch.Tensor(rxns[ind])

    ec1 = item[1].split('.')[0]

    if ec1 in model_lookup_train.keys():
        model_lookup_train[ec1].append(rxn)
    else:
        temp_list = []
        temp_list.append(rxn)
        
        model_lookup_train[ec1] = temp_list

print(len(model_lookup_train))  # unique_ec: 176

output_file = '../pred_rxn_EC1/esm_emb_dict_ec1.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(model_lookup_train, file)


## labels_train
with open('../pred_rxn_EC123/labels_train_ec3.pkl', 'rb') as file:
    labels_ec3 = pickle.load(file)

labels_ec2 = []
labels_ec1 = []
for item in labels_ec3:
    ec2 = '.'.join(item.split('.')[:2])
    ec1 = item.split('.')[0]
    labels_ec2.append(ec2)
    labels_ec1.append(ec1)
print(labels_ec2[:5])
print(labels_ec1[:5])
with open('../pred_rxn_EC12/labels_train_ec2.pkl', 'wb') as file:
    pickle.dump(labels_ec2, file)
with open('../pred_rxn_EC1/labels_train_ec1.pkl', 'wb') as file:
    pickle.dump(labels_ec1, file)


### labels_test
with open('../pred_rxn_EC123/labels_test_ec3.pkl', 'rb') as file:
    labels_ec3 = pickle.load(file)

labels_ec2 = []
labels_ec1 = []
for item in labels_ec3:
    ec2 = '.'.join(item.split('.')[:2])
    ec1 = item.split('.')[0]
    labels_ec2.append(ec2)
    labels_ec1.append(ec1)
print(labels_ec2[:5])
print(labels_ec1[:5])
with open('../pred_rxn_EC12/labels_test_ec2.pkl', 'wb') as file:
    pickle.dump(labels_ec2, file)
with open('../pred_rxn_EC1/labels_test_ec1.pkl', 'wb') as file:
    pickle.dump(labels_ec1, file)

