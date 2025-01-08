
import os
import argparse
import re
from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import rdmolops
import torch
import pickle

def get_single_edge_index(smi_in):
    if isinstance(smi_in, (list, tuple)):
        chars_lst = smi_in
        smi = ''.join(smi_in)
    elif isinstance(smi_in, str):
        chars_lst = smi_in.strip().split(' ')
        smi = ''.join(chars_lst)
    # initialize adjacency_matrix of smiles sequence
    smi_token_adjacency = torch.zeros(len(chars_lst), len(chars_lst))
    not_atom_indices = list()
    atom_indices = list()
    atom_lst = []
    for j, cha in enumerate(chars_lst):
        if (len(cha) == 1 and not cha.isalpha()) or (len(cha) > 1 and cha[0] not in ['[', 'B', 'C']):
            not_atom_indices.append(j)
        else:
            atom_indices.append(j)
            atom_lst.append(cha)

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
    except: # for invalid SMILES,random edge_index
        print('invalid SMILES, pad+1')
        length = len(chars_lst)
        for x in range(length):
            if x in atom_indices:
                for y in range(x + 1, length):
                    if y in atom_indices:
                        smi_token_adjacency[x, y] = 1
                        smi_token_adjacency[y, x] = 1
                        break
    else:
        length = len(chars_lst)
        if mol is not None:
            mol_adjacency = rdmolops.GetAdjacencyMatrix(mol)
            for x in range(length):
                for y in range(length):
                    if x in atom_indices and y in atom_indices:
                        smi_token_adjacency[x, y] = mol_adjacency[atom_indices.index(x), atom_indices.index(y)]
                    elif x == y and x in not_atom_indices:
                        smi_token_adjacency[:, y] = 0
                        smi_token_adjacency[x, :] = 0
                        smi_token_adjacency[x, y] = 0
        else:# for invalid SMILES,random edge_index
            print('invalid SMILES, pad+1')
            for x in range(length):
                if x in atom_indices:
                    for y in range(x+1, length):
                        if y in atom_indices:
                            smi_token_adjacency[x, y] = 1
                            smi_token_adjacency[y, x] = 1
                            break

    index_row_x2 = torch.where(smi_token_adjacency == 1)
    edge_index = torch.stack(index_row_x2)
    len_char_4pad = len(chars_lst)
    return [len_char_4pad, edge_index]



def generate_edge_index_pkl(path_arg):
    count_pad = 0
    if isinstance(path_arg, str):
        path_arg = [path_arg]
    for path in path_arg:
        print(f'generate_edge_index_pkl for {path}')
        with open(path, "r") as f_in:
            src_lines = f_in.readlines()
            adj_mat_lst = []
            for line in tqdm(src_lines):
                adj_mat = get_single_edge_index(line)
                adj_mat_lst.append(adj_mat)
            #print(adj_mat_lst)
        with open(path[:-4] + '_edge_index.pkl', 'wb') as f_out:
            pickle.dump(adj_mat_lst, f_out)
        del adj_mat_lst
        print(f'have extracted edge_index for {path}, total {len(src_lines)} , pad for {count_pad}')

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Get file and fn_type ")
    # parser.add_argument('--path_lst', '-path_lst', type=str, nargs='+', default=[], required=True)
    # args = parser.parse_args()
    # print(args)
    # generate_edge_index_pkl(args.path_lst)
    src = ['C S C C C ( O ) ( C C ( = O ) O ) C ( = O ) O',
               'C C C C C C ( = O ) C C ( = O ) C C ( = O ) C C ( * ) = O']

    smi =src[0].replace(' ', '')
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
    print(Chem.MolToSmiles(mol))
    try:
        mol_adjacency = rdmolops.GetAdjacencyMatrix(mol)
    except:
        print('no')
    else:
        print(mol_adjacency, 'ttttttttttttt')

    '''
python generate_edge_index.py -path_lst        \
'/home/zhangmeng/aMy-ONMT003/data/biochem_npl_20xaug_RSMILES/src-test.txt'  '/home/zhangmeng/aMy-ONMT003/data/biochem_npl_20xaug_RSMILES/src-train.txt' '/home/zhangmeng/aMy-ONMT003/data/biochem_npl_20xaug_RSMILES/src-val.txt'  

    '''