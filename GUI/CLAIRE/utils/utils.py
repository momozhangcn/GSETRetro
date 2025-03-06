import csv
import random
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def dist_map(tag1, emb1, tag2, emb2):
    dist = {}
    for i, key1 in tqdm(enumerate(tag1), total=len(tag1)):
        flag = emb1[i].unsqueeze(0)
        dist_norm = (flag - emb2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(tag2):
            dist[key1][key2] = dist_norm[j]
    return dist

def get_pred(out_filename, pred_type="_prediction"):
    result = open(out_filename + pred_type + '.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        ec_i = row[1].split(":")[1].split("/")[0]
        pred_label.append(ec_i)
    return pred_label

def GMM(distance, gmm_lst):
    confidence = []
    for j in range(len(gmm_lst)):
        main_GMM = gmm_lst[j]
        a, b = main_GMM.means_
        true_model_index = 0 if a[0] < b[0] else 1
        certainty = main_GMM.predict_proba([[distance]])[0][true_model_index]
        confidence.append(certainty)
    return np.mean(confidence)
    
def get_topk_pred(df, csv_name=None,topk=None, gmm = None):
    if csv_name != None:
        out_file = open(csv_name + '_prediction.csv', 'w', newline='')
        csvwriter = csv.writer(out_file, delimiter=',')
        all_test_EC = set()
        for col in df.columns:
            ec = []
            smallest_10_dist_df = df[col].nsmallest(10)
            dist_lst = list(smallest_10_dist_df)

            for i in range(topk+1):
                EC_i = smallest_10_dist_df.index[i]
                dist_i = dist_lst[i]
                if gmm != None:
                    gmm_lst = pickle.load(open(gmm, 'rb'))
                    dist_i = GMM(dist_i, gmm_lst)
                dist_str = "{:.4f}".format(dist_i)
                all_test_EC.add(EC_i)
                ec.append('EC:' + str(EC_i) + '/' + dist_str)
            ec.insert(0, col)
            csvwriter.writerow(ec)
        return
    else:
        all_test_EC = set()
        ec_lst = []
        top1_ec_lst = []
        for col in df.columns:
            ec = []
            smallest_10_dist_df = df[col].nsmallest(10)
            dist_lst = list(smallest_10_dist_df)

            for i in range(topk):
                EC_i = smallest_10_dist_df.index[i]
                dist_i = dist_lst[i]
                if gmm != None:
                    gmm_lst = pickle.load(open(gmm, 'rb'))
                    dist_i = GMM(dist_i, gmm_lst)
                dist_str = "{:.4f}".format(dist_i)
                all_test_EC.add(EC_i)
                ec.append('EC:' + str(EC_i) + '/' + dist_str)
            ec.insert(0, col)
            ec_lst.append(ec)
            top1_ec_lst.append(ec[1])
        columns = [f'top_{i+1}' for i in range(topk)]
        columns.insert(0, 'rxn_id')
        return top1_ec_lst, pd.DataFrame(ec_lst, columns=columns)