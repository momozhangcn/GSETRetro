import numpy as np
import pandas as pd
import argparse
import os
import re
import random
import multiprocessing

from rdkit import Chem
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def smi_tokenizer(smi):
    double_atom = '|Cl?|Mg?|Fe?|Se?|Co?|Br?|Si?|Sn?|Li?|Mn?|Xe?|Zn?|Cu?|Na?|Cr?|As?|Al?|Pb?|Te?|Bi?|Mo?|Hg?|Re?|Ge?|Ru?|Zr?'
    pattern = '(\[[^\]]+]'+double_atom+'|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|[a-z]|[A-Z])'
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


def preprocess(save_dir, reactants, products, set_name, augmentation=1, processes=-1):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = [{
        "reactant": i,
        "product": j,
        "augmentation": augmentation} for i, j in zip(reactants, products)]
    src_data = []
    tgt_data = []

    processes = multiprocessing.cpu_count() if processes < 0 else processes
    pool = multiprocessing.Pool(processes=processes)
    results = pool.map(func=multi_process, iterable=data)
    pool.close()
    pool.join()
    for result in tqdm(results):
        src_data.extend(result['src_data'])
        tgt_data.extend(result['tgt_data'])
    print('size', len(src_data))

    if augmentation != 999:
        with open(
                os.path.join(save_dir, 'src-{}.txt'.format(set_name)), 'w') as f:
            for src in src_data:
                f.write('{}\n'.format(src))

        with open(
                os.path.join(save_dir, 'tgt-{}.txt'.format(set_name)), 'w') as f:
            for tgt in tgt_data:
                f.write('{}\n'.format(tgt))
    return src_data, tgt_data


def multi_process(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "src_data": [],
        "tgt_data": [],
    }
    pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
    reactant = reactant.split(".")

    cano_product = clear_map_canonical_smiles(product)
    cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(pro_atom_map_numbers)) > 0 ])
    return_status['src_data'].append(smi_tokenizer(cano_product))
    return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
    pro_mol = Chem.MolFromSmiles(cano_product)
    rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
    # if aum_time > 1, do aug, else just canonical_smiles
    for _ in range(int(augmentation-1)):
        pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
        rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
        rea_smi = ".".join(rea_smi)
        return_status['src_data'].append(smi_tokenizer(pro_smi))
        return_status['tgt_data'].append(smi_tokenizer(rea_smi))
    return return_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='USPTO_50K')
    parser.add_argument("-aug_time", type=int, default=1)
    parser.add_argument("-seed", type=int, default=2024)
    parser.add_argument("-processes", type=int, default=-1)
    parser.add_argument('-subdataset', type=str, nargs='+', default=['test', 'val', 'train'])

    args = parser.parse_args()
    print(f'preprocessing dataset {args.dataset}...')
    assert args.dataset in ['USPTO_50K', 'USPTO_full', 'USPTO-MIT']

    random.seed(args.seed)

    datadir = f'./dataset/{args.dataset}'
    savedir = f'./dataset/{args.dataset}_{args.aug_time}xaug'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, data_set in enumerate(args.subdataset):
        aug_time = 1 if data_set == 'test' else args.aug_time

        csv_path = f"{datadir}/raw_{data_set}.csv"
        csv = pd.read_csv(csv_path)
        reaction_list = list(csv["reactants>reagents>production"])

        # random.shuffle(reaction_list)
        reactant_smarts_list = list(
            map(lambda x: x.split('>')[0], reaction_list))
        reactant_smarts_list = list(
            map(lambda x: x.split(' ')[0], reactant_smarts_list))
        reagent_smarts_list = list(
            map(lambda x: x.split('>')[1], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split('>')[2], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split(' ')[0], product_smarts_list))  # remove ' |f:1...'
        print(f'Have extracted reaction data from {csv_path} file and convert SMARTS to SMILES')

        # reaction_class_list = list(map(lambda x: int(x) - 1, csv['class']))
        sub_react_list = reactant_smarts_list
        sub_prod_list = product_smarts_list
        print(sub_react_list[:10])
        # duplicate multiple product reactions into multiple ones with one product each
        # 多反应物对应的sub_prod_list索引
        multiple_product_indices = [i for i in range(len(sub_prod_list)) if "." in sub_prod_list[i]]
        for index in multiple_product_indices:
            products = sub_prod_list[index].split(".")
            for product in products:
                sub_react_list.append(sub_react_list[index])
                sub_prod_list.append(product)
        for index in multiple_product_indices[::-1]:
            del sub_react_list[index]
            del sub_prod_list[index]
        ## ↑完成对多product拆解
        src_data, tgt_data = preprocess(
            savedir,
            sub_react_list,
            sub_prod_list,
            data_set,
            aug_time,
            processes=args.processes,
        )
