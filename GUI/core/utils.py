# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:39:16 2024

@author: DELL
"""


import numpy as np
import pandas as pd

from admet_ai import ADMETModel
#from DeepPurpose import utils, DTI



def predict_compound_ADMET_property(smiles_list):
    results = {}
    model = ADMETModel()
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


from GSETRetro.run_single_multi4GUI import run_Muiti_Step_Plan
def predict_compound_derivative_GSETransformer(smiles_list, model_type, beam_size, exp_topk, iterations,route_topk,device):
    result = run_Muiti_Step_Plan(smiles_list, model_type, beam_size, exp_topk, iterations,route_topk, device)
    print(result)
    return result