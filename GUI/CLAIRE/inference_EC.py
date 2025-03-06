import torch
import os
import sys
if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # sys.path.append('/root/CLAIRE')
# To avoid errors caused by directories having the same name.
from CLAIRE.utils.utils import *

import pandas as pd
import torch.nn as nn
from sklearn.metrics import  accuracy_score, f1_score

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

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

def inference(train_data, test_data, train_tags, test_tags, test_labels, pretrained_model,
              evaluation = False, out_filename = "results/test", topk=3, gmm = None):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        dtype = torch.float32
        model = LayerNormNet(1280, 128, device, dtype)

        try:
            checkpoint = torch.load(pretrained_model, map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')

        model.load_state_dict(checkpoint)
        model.eval()

        emb_train = model(torch.tensor(train_data).to(device=device, dtype=dtype))
        emb_test = model(torch.tensor(test_data).to(device=device, dtype=dtype))
        dist = dist_map(test_tags, emb_test, train_tags, emb_train)

        seed()
        df = pd.DataFrame.from_dict(dist)

        print('=' * 20)
        #if out_filename
        top1_ec_lst, df_all_ec = get_topk_pred(df, out_filename, topk=topk, gmm=gmm)
        if evaluation:
            pred_label = get_pred(out_filename, pred_type='_prediction')
            true_label = test_labels
            f1 = f1_score(true_label, pred_label, average='weighted')
            acc = accuracy_score(true_label, pred_label)
            print(f'f1:{f1:.3f}   |   acc:{acc:.3f}')
    except:
        device = torch.device("cpu")
        dtype = torch.float32
        model = LayerNormNet(1280, 128, device, dtype)

        try:
            checkpoint = torch.load(pretrained_model, map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')

        model.load_state_dict(checkpoint)
        model.eval()

        emb_train = model(torch.tensor(train_data).to(device=device, dtype=dtype))
        emb_test = model(torch.tensor(test_data).to(device=device, dtype=dtype))
        dist = dist_map(test_tags, emb_test, train_tags, emb_train)

        seed()
        df = pd.DataFrame.from_dict(dist)

        print('=' * 20)
        # if out_filename
        top1_ec_lst, df_all_ec = get_topk_pred(df, out_filename, topk=topk, gmm=gmm)
        if evaluation:
            pred_label = get_pred(out_filename, pred_type='_prediction')
            true_label = test_labels
            f1 = f1_score(true_label, pred_label, average='weighted')
            acc = accuracy_score(true_label, pred_label)
            print(f'f1:{f1:.3f}   |   acc:{acc:.3f}')

    return top1_ec_lst, df_all_ec
        

if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    ### train data
    train_data = pickle.load(open (f'{dir_name}/data/model_lookup_train.pkl', 'rb'))
    train_labels = pickle.load(open (f'{dir_name}/data/pred_rxn_EC123/labels_train_ec3.pkl', 'rb'))

    ### test data
    test_data = pickle.load(open (f'{dir_name}/data/model_lookup_test.pkl', 'rb'))
    test_labels = pickle.load(open (f'{dir_name}/data/pred_rxn_EC123/labels_test_ec3.pkl', 'rb'))

    test_tags = ['rxn_' + str(i) for i in range(len(test_data))]

    pretrained_model = f'{dir_name}/results/model/pred_rxn_EC123/layer5_node1280_triplet2000_final.pth'
    inference(train_data, test_data, train_labels, test_tags,test_labels, pretrained_model, evaluation=True, topk=3, gmm = f'{dir_name}/gmm/gmm_ensumble.pkl')
