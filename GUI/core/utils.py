# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from GUI.multi_step_plan_4GUI import run_Muiti_Step_Plan
def predict_compound_derivative_GSETransformer(smiles_list, model_type, beam_size, exp_topk, iterations,route_topk,device, model_path):
    result = run_Muiti_Step_Plan(smiles_list, model_type, beam_size, exp_topk, iterations,route_topk, device, model_path)
    return result

from admet_ai import ADMETModel
def predict_compound_ADMET_property(smiles_list):
    results = {}
    try:
        model = ADMETModel()
        smiles_list = np.unique(smiles_list)
        results = model.predict(smiles=smiles_list)
    except:
        model = ADMETModel(num_workers=0)
        # Setting num_workers=0 will slow down the process.
        # If not set, it might raise the following error:
        # self.multiprocessing_context = multiprocessing_context
        # raise ValueError( ValueError: multiprocessing_context option should
        # specify a valid start method in ['spawn'], but got multiprocessing_context='forkserver'
        smiles_list = np.unique(smiles_list)
        results = model.predict(smiles=smiles_list)
    return results.reset_index()

def refine_compound_ADMET_property(ADMET_list, smiles, property_class = 'Physicochemical'):
    properties_mapping = {
        'Physicochemical': [
            'molecular_weight',
            'logP',
            'hydrogen_bond_acceptors',
            'hydrogen_bond_donors',
            'Lipinski',
            'stereo_centers',
            'tpsa',
            'molecular_weight_drugbank_approved_percentile',
            'logP_drugbank_approved_percentile',
            'hydrogen_bond_acceptors_drugbank_approved_percentile',
            'hydrogen_bond_donors_drugbank_approved_percentile',
            'Lipinski_drugbank_approved_percentile',
            'stereo_centers_drugbank_approved_percentile',
            'tpsa_drugbank_approved_percentile',
            'HydrationFreeEnergy_FreeSolv',
            'HydrationFreeEnergy_FreeSolv_drugbank_approved_percentile',
            'Lipophilicity_AstraZeneca',
            'Lipophilicity_AstraZeneca_drugbank_approved_percentile',
            'Solubility_AqSolDB',
            'Solubility_AqSolDB_drugbank_approved_percentile'
        ],
        'Absorption': [
            'HIA_Hou',
            'PAMPA_NCATS',
            'Pgp_Broccatelli',
            'Caco2_Wang',
            'HIA_Hou_drugbank_approved_percentile',
            'PAMPA_NCATS_drugbank_approved_percentile',
            'Pgp_Broccatelli_drugbank_approved_percentile',
            'Caco2_Wang_drugbank_approved_percentile'
        ],
        'Distribution': [
            'BBB_Martins',
            'BBB_Martins_drugbank_approved_percentile',
            'VDss_Lombardo',
            'VDss_Lombardo_drugbank_approved_percentile'
        ],
        'Metabolism': [
            'CYP1A2_Veith',
            'CYP2C19_Veith',
            'CYP2C9_Substrate_CarbonMangels',
            'CYP2C9_Veith',
            'CYP2D6_Substrate_CarbonMangels',
            'CYP2D6_Veith',
            'CYP3A4_Substrate_CarbonMangels',
            'CYP3A4_Veith',
            'CYP1A2_Veith_drugbank_approved_percentile',
            'CYP2C19_Veith_drugbank_approved_percentile',
            'CYP2C9_Substrate_CarbonMangels_drugbank_approved_percentile',
            'CYP2C9_Veith_drugbank_approved_percentile',
            'CYP2D6_Substrate_CarbonMangels_drugbank_approved_percentile',
            'CYP2D6_Veith_drugbank_approved_percentile',
            'CYP3A4_Substrate_CarbonMangels_drugbank_approved_percentile',
            'CYP3A4_Veith_drugbank_approved_percentile'
        ],
        'Excretion': [
            'Half_Life_Obach',
            'Half_Life_Obach_drugbank_approved_percentile',
            'Clearance_Hepatocyte_AZ',
            'Clearance_Hepatocyte_AZ_drugbank_approved_percentile',
            'Clearance_Microsome_AZ',
            'Clearance_Microsome_AZ_drugbank_approved_percentile'
        ],
        'Toxicity': [
            'AMES',
            'Carcinogens_Lagunin',
            'ClinTox',
            'DILI',
            'NR-AR-LBD',
            'NR-AR',
            'NR-AhR',
            'NR-Aromatase',
            'NR-ER-LBD',
            'NR-ER',
            'NR-PPAR-gamma',
            'SR-ARE',
            'SR-ATAD5',
            'SR-HSE',
            'SR-MMP',
            'SR-p53',
            'Skin_Reaction',
            'hERG',
            'LD50_Zhu',
            'Carcinogens_Lagunin_drugbank_approved_percentile',
            'ClinTox_drugbank_approved_percentile',
            'DILI_drugbank_approved_percentile',
            'NR-AR-LBD_drugbank_approved_percentile',
            'NR-AR_drugbank_approved_percentile',
            'NR-AhR_drugbank_approved_percentile',
            'NR-Aromatase_drugbank_approved_percentile',
            'NR-ER-LBD_drugbank_approved_percentile',
            'NR-ER_drugbank_approved_percentile',
            'NR-PPAR-gamma_drugbank_approved_percentile',
            'SR-ARE_drugbank_approved_percentile',
            'SR-ATAD5_drugbank_approved_percentile',
            'SR-HSE_drugbank_approved_percentile',
            'SR-MMP_drugbank_approved_percentile',
            'SR-p53_drugbank_approved_percentile',
            'Skin_Reaction_drugbank_approved_percentile',
            'hERG_drugbank_approved_percentile',
            'LD50_Zhu_drugbank_approved_percentile'
        ]}
    rows = np.where(ADMET_list['index'] == smiles)[0][0]
    cols = properties_mapping[property_class]
    results = pd.DataFrame({'predicted property': cols, 'values': np.round(ADMET_list.loc[rows, cols].astype(float), 4)})
    results = results.reset_index(drop = True)
    return results


def find_all_paths(full_route):
    """根据路径描述字符串查找从A到所有末端的路径"""
    # 1解析路径描述字符串，构建邻接表
    graph = {}
    rxns_lst = full_route.split('|')
    for i in range(len(rxns_lst)):
        start_mole = rxns_lst[i].split('>')[0]
        end_mole = rxns_lst[i].split('>')[-1]
        if end_mole.find('kegg') == -1:
            graph[start_mole] = end_mole.split('.')
        else:
            graph[start_mole] = [end_mole]
    # 2深度优先搜索 (DFS) 查找从start到所有末端的路径
    def dfs(node, current_path):
        current_path.append(node)
        # 如果当前节点没有出边（即末端节点），返回路径
        if node not in graph or not graph[node]:
            return [current_path]
        paths = []
        for neighbor in graph[node]:
            # 递归查找后续路径
            paths.extend(dfs(neighbor, current_path.copy()))
        return paths
    # 3从A开始查找所有路径
    root_mole = full_route.split('>')[0]  # 获取起点A
    return dfs(root_mole, [])

from GUI.CLAIRE.example_inference_EC import rxns_EC_prediction
def predict_EC_number(smiles_list):
    result = rxns_EC_prediction(smiles_list)
    return result
def predict_rxns_EC_number(full_route):
    #full_route = 'COc1ccc([C@H]2O[C@@H](O)[C@H]3[C@@H]2C(=O)O[C@@H]3c2ccc(OC)c(OC)c2)cc1OC>0.9599>COc1ccc(C=O)cc1OC|COc1ccc(C=O)cc1OC>1.0000>COc1cc(C=O)ccc1O.O=C([O-])[O-]|COc1cc(C=O)ccc1O>1.0000>keggpath=kegg.jp/pathway/rn00627+C00755|O=C([O-])[O-]>1.0000>keggpath=kegg.jp/pathway/rn00230+C00288'
    rxns_lst = full_route.split('|')
    print('1. Raw rxns in selected pathway: ', '\n', rxns_lst)

    forward_rxns_lst = []
    for rxn in rxns_lst:
        if rxn.find('kegg') == -1:
            forward_rxn = rxn.split('>')[-1] + '>>' + rxn.split('>')[0]
            forward_rxns_lst.append(forward_rxn)
    print('2. Corresponding forward rxns: ', '\n', forward_rxns_lst)

    rxn_EC_dict = {}
    top_EC_lst, df_all_ec = rxns_EC_prediction(forward_rxns_lst)
    for i in range(len(forward_rxns_lst)):
        rxn_EC_dict[forward_rxns_lst[i].split('>>')[-1]] = top_EC_lst[i]
    print('3. Get predicted EC_number: ', '\n', df_all_ec)
    return df_all_ec, rxn_EC_dict