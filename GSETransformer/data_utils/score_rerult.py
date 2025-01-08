#!/usr/bin/env python
import rdkit.Chem as Chem
import re
from collections import defaultdict
import numpy as np
import tqdm

def read_file(pred_result_path, gt_tgt_path):
    with open(pred_result_path, 'r') as pred_result_in, \
            open(gt_tgt_path, 'r') as gt_tgt_in:
        pred_result_ctt = pred_result_in.readlines()
        gt_tgt_ctt = gt_tgt_in.readlines()

        beam_size = len(pred_result_ctt)//len(gt_tgt_ctt)
        pred_result_output_list = []  # List of beams if beam_size is > 1 else list of smiles
        pred_result_cur_beam = []  # Keep track of the current beam

        for line_pred in pred_result_ctt:
            parse = line_pred.strip().replace(' ', '')  # default parse function
            pred_result_cur_beam.append(parse)
            if len(pred_result_cur_beam) == beam_size:
                if beam_size == 1:
                    pred_result_output_list.append(pred_result_cur_beam[0])
                else:
                    pred_result_output_list.append(pred_result_cur_beam)
                pred_result_cur_beam = []

        tgt_output_list = []
        for line_tgt in gt_tgt_ctt:
            parse = line_tgt.strip().replace(' ', '')  # default parse function
            tgt_output_list.append(parse)

    return pred_result_output_list, tgt_output_list, beam_size

def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None

    if mol is None:
        return ''
    else:
        try:
            cano_smi = Chem.MolToSmiles(mol)
        except:
            return ''
        else:
            return cano_smi

def canonicalize(smiles=None, smiles_list=None):
    """Return the canonicalized version of the given smiles or smiles list"""
    assert (smiles is None) != (smiles_list is None)  # Only take one input

    if smiles is not None:
        return canonicalize_smiles(smiles)
    elif smiles_list is not None:
        # Convert smiles to mol and back to cannonicalize
        new_smiles_list = []

        for smiles in smiles_list:
            new_smiles_list.append(canonicalize_smiles(smiles))
        return new_smiles_list

def match_smiles_set(source_set, target_set):
    if len(source_set) != len(target_set):
        return False

    for smiles in target_set:
        if smiles not in source_set:
            return False
    return True

def match_smiles_lists(pred_list, target_list, beam_size, should_print=True):
    n_data = 0
    n_matched = np.zeros(beam_size)  # Count of matched smiles
    n_invalid = np.zeros(beam_size)  # Count of invalid smiles
    n_repeat = np.zeros(beam_size)   # Count of repeated predictions
    #
    # with open('template/rare_indices.txt', 'r+') as r_file:
    #     rare_rxn_list = json.load(r_file)

    for data_idx, target_smiles in enumerate(tqdm.tqdm(target_list)):
        # if data_idx not in rare_rxn_list:
        #     continue
        n_data += 1
        target_set = set(canonicalize(smiles_list=target_smiles.split('.')))

        pred_beam = pred_list[data_idx]

        beam_matched = False
        prev_sets = []
        for beam_idx, pred_smiles in enumerate(pred_beam):
            pred_set = set(canonicalize(smiles_list=pred_smiles.split('.')))
            if '' in pred_set:
                pred_set.remove('')
            set_matched = match_smiles_set(pred_set, target_set)

            # Check if current pred_set matches any previous sets
            for prev_set in prev_sets:
                if match_smiles_set(pred_set, prev_set):
                    n_repeat[beam_idx] += 1

            if len(pred_set) > 0:
                # Add pred set to list of predictions for current example
                prev_sets.append(pred_set)
            else:
                # If the pred set is empty and the string is not, then invalid
                if pred_smiles != '':
                    n_invalid[beam_idx] += 1

            # Increment if not yet matched beam and the pred set matches
            if set_matched and not beam_matched:
                n_matched[beam_idx] += 1
                beam_matched = True

    if should_print:
        print(f'beam:{beam_size}, num of gt_tgt: {n_data}')
        if 10 <= beam_size < 20:
            chek_list = [1, 3, 5, 10]
        elif 20 <= beam_size < 50:
            chek_list = [1, 3, 5, 10, 20]
        elif beam_size >= 50:
            chek_list = [1, 3, 5, 10, 20, 50]
        for beam_idx in range(beam_size):

            if beam_idx+1 in chek_list:
                match_perc = np.sum(n_matched[:beam_idx+1]) / n_data
                invalid_perc = n_invalid[beam_idx] / n_data
                repeat_perc = n_repeat[beam_idx] / n_data

                print(f'beam: {beam_idx+1}, matched: {match_perc*100:.2f}%, '
                      f'invalid: {invalid_perc*100:.2f}%, repeat: {repeat_perc*100:.2f}%')

    return n_data, n_matched, n_invalid, n_repeat
if __name__ == "__main__":
    pred_path = '/home/zhangmeng/aMy-ONMT003/data/biochem_npl_10xaug_RSmiles/pred_model_step_95000_src_rsmile.txt'
    tgt_path = '/home/zhangmeng/aMy-ONMT003/data/biochem_npl_10xaug_RSmiles/tgt-test_1K.txt'
    smiles_list, target_list, beam = read_file(pred_path, tgt_path)
    match_smiles_lists(smiles_list, target_list, beam)