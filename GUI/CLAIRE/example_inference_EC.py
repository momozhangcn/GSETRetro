import os
import sys
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))  # sys.path.append('/root/CLAIRE')

from inference_EC import inference
import pickle
import numpy as np
import pandas as pd
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
from drfp.DrfpEncoder import DrfpEncoder #further modified, skip to pip install drfp#.py

def rxns_EC_prediction(rxns_lst=None):
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    if rxns_lst==None: # for test
        rxns_lst = ["CC(=O)O>>CC(=O)N/C=C(/CC(=O)O)C(=O)O",
                    "CSCCC(=O)C(=O)O>>CSCCC(O)(CC(=O)O)C(=O)O",
                    'C(=O)C/C=C/CCCCCCCCCC>>C(=O)/C=C/CCCCCCCCCCC']

    drfp_fps = DrfpEncoder.encode(rxns_lst, n_folded_length=256)
    # pickle.dump(drfp_fps, open('example_my_rxn_fps.pkl', 'wb'))
    rxnfp = rxnfp_generator.convert_batch(rxns_lst)
    # pickle.dump(rxnfp, open('example_rxnfp_emb.pkl', 'wb'))

    # drfp_fps = pickle.load(open('example_my_rxn_fps.pkl', 'rb'))
    # rxnfp = pickle.load(open('example_rxnfp_emb.pkl', 'rb'))
    test_data = []
    for ind, item in enumerate(rxnfp):
        rxn_emb = np.concatenate((np.reshape(item, (1, 256)), np.reshape(drfp_fps[ind], (1, 256))), axis=1)
        test_data.append(rxn_emb)

    test_data = np.concatenate(test_data, axis=0)
    ###
    current_dir = os.path.dirname(__file__)
    train_data = pickle.load(open(f'{current_dir}/data/model_lookup_train.pkl', 'rb'))
    train_labels = pickle.load(open(f'{current_dir}/data/pred_rxn_EC123/labels_train_ec3.pkl', 'rb'))
    # if you want 1-level EC or 2-level EC, change it to pred_rxn_EC1/labels_trained_ec1.pkl or pred_rxn_EC12/labels_trained_ec2.pkl, resepetively.
    # input your test_labels
    test_labels = None
    test_tags = ['rxn_' + str(i) for i in range(len(test_data))]

    # EC calling results using maximum separation
    pretrained_model = f'{current_dir}/results/model/pred_rxn_EC123/layer5_node1280_triplet2000_final.pth'
    top1_ec_lst, df_all_ec = inference(train_data, test_data, train_labels, test_tags, test_labels, pretrained_model,
              evaluation=False, out_filename=None, topk=3, gmm=f'{current_dir}/gmm/gmm_ensumble.pkl')
    return top1_ec_lst, df_all_ec


if __name__ == '__main__':
    top1_ec_lst, df_all_ec = rxns_EC_prediction()
    print(top1_ec_lst, df_all_ec)