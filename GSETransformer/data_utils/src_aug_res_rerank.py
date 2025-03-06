
import sys
import os
import re
import random

import os
import sys
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from rdkit import Chem
from rxnmapper import RXNMapper

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
rxn_mapper = RXNMapper()




def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def canonicalize_smiles(smiles):
    smiles = smiles.strip().replace(' ', '')
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return ''
    else:
        if mol is not None:
            [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
            try:
                smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            except:
                return ''
            else:
                return smi
        else:
            return ''

def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


def get_root_id(mol,root_map_number):
    root = -1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            root = i
            break
    return root

def atom_map_src_smi(smi):
    smi = smi.strip().replace(' ', '')
    pad_reaction = smi + '>>' + smi

    try:
        get_atomap_reaction = rxn_mapper.get_attention_guided_atom_maps([pad_reaction])  ## 传入列表，不可字符串
        atomap_reaction = get_atomap_reaction[0]['mapped_rxn']  # 化学式 #返回内含一个字典的列表，lst[0]字典，dct['mapped_rxn']为映射后的化学式
        atom_mapped_src = atomap_reaction.split('>>')[0]
    except:
        return smi  #False, smi
    else:
        return atom_mapped_src  #True, atom_mapped_src

# data atom mapped src and aug time
def atom_mapped_src_aug(if_mapped_smi, aug_time):
    pt = re.compile(r':(\d+)]')
    product = if_mapped_smi
    augmentation = aug_time
    pro_mol = Chem.MolFromSmiles(product)
    """checking data quality"""
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status": 0,
        "src_data": [],
    }
    if len(set(pids)) != len(pids):  # mapping is not 1:1
        return_status["status"] = "error_mapping"
    if "" == product:
        return_status["status"] = "empty_p"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if len(pro_mol.GetAtoms()) == 1:
        return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))

        product_roots = [-1]
        max_times = len(pro_atom_map_numbers)
        times = min(augmentation, max_times)
        if times < augmentation:  # times = max_times
            product_roots.extend(pro_atom_map_numbers)
            product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
        else:  # times = augmentation
            while len(product_roots) < times:
                product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                # pro_atom_map_numbers.remove(product_roots[-1])
                if product_roots[-1] in product_roots[:-1]:
                    product_roots.pop()
        times = len(product_roots)
        assert times == augmentation

        # candidates = []
        for k in range(times):
            pro_root_atom_map = product_roots[k]
            pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
            pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
            product_tokens = smi_tokenizer(pro_smi)
            return_status['src_data'].append(product_tokens)
        assert len(return_status['src_data']) == augmentation
    # if can not do root-aline, do random aug
    else:
        try:
            cano_product = clear_map_canonical_smiles(product)
            return_status['src_data'].append(smi_tokenizer(cano_product))
            pro_mol = Chem.MolFromSmiles(cano_product)
            for i in range(int(augmentation-1)):
                pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True, isomericSmiles=True)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
        except:
            print(f'might be invalid smi:{product}')
            for i in range(int(augmentation)):  #invalid smi
                return_status['src_data'].append(smi_tokenizer(product))
    return return_status['src_data'], return_status["status"]

def compute_rank_rerank(prediction,beam_size, raw=False,alpha=1.0):
    valid_score = [[k for k in range(len(prediction[j]))] for j in range(len(prediction))]
    rank = {}
    max_frag_rank = {}
    highest = {}
    if raw:
        # no test augmentation
        assert len(prediction) == 1
        for j in range(len(prediction)):
            # error detection
            prediction[j] = [i for i in prediction[j] if i[0] != ""]
            for k, data in enumerate(prediction[j]):
                rank[data] = 1 / (alpha * k + 1)
    else:
        for j in range(len(prediction)):
            for k in range(len(prediction[j])):
                # predictions[i][j][k] = canonicalize_smiles_clear_map(predictions[i][j][k])
                if prediction[j][k] == "":
                    valid_score[j][k] = beam_size + 1
            # error detection and deduplication
            de_error = [i[0] for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]) if i[0] != ""]
            prediction[j] = list(set(de_error))
            prediction[j].sort(key=de_error.index)
            for k, data in enumerate(prediction[j]):
                if data in rank:
                    rank[data] += 1 / (alpha * k + 1)
                else:
                    rank[data] = 1 / (alpha * k + 1)
                if data in highest:
                    highest[data] = min(k,highest[data])
                else:
                    highest[data] = k
        for key in rank.keys():
            rank[key] += highest[key] * 1e-8#-1e8
        rank = list(zip(rank.keys(), rank.values()))
        rank.sort(key=lambda x: x[1], reverse=True)
        rank = rank[:beam_size]
        

    rank_pred = [r[0] for r in rank]
    rank_score = [r[1] for r in rank]
        #ranked_results = []
        #ranked_results.append([item[0][0] for item in rank])
    return rank_pred, rank_score


if __name__ == "__main__":
    # smi = 'C=CC(C)(C)[C@@]12C[C@H]3c4nc5ccccc5c(=O)n4[C@H](C)C(=O)N3C1N(C(C)=O)c1ccccc12'.replace(' ', '')
    # # step_01 do atom map for src smi
    # mapped_smi = atom_map_src_smi(smi)
    # print(mapped_smi)
    # # step_02 do root-change aug for mapped src smi
    # aug_src1, _ = atom_mapped_src_aug(mapped_smi, 10)
    # print(aug_src1)
    # aug_src2, _ = atom_mapped_src_aug(smi, 10)
    # print(aug_src2)
    # # step_03 do prediction for root-change auged smi
    # # step_04 got result and rerank the result
    # pred_in = [aug_src1, aug_src2]
    # rank_pred, rank_score= compute_rank_rerank(pred_in, 10)
    # print(rank_pred)
    # print(rank_score)
    pad_reaction = 'C1(CCC(CC1)CBr)OC>>BrCC1CCC(CC1)O'
    get_atomap_reaction = rxn_mapper.get_attention_guided_atom_maps([pad_reaction])  ## 传入列表，不可字符串
    atomap_reaction = get_atomap_reaction[0]['mapped_rxn']
    print(atomap_reaction)
