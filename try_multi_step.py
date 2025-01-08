from utils.multithread_utils import run_multistep
import os
from time import time
import numpy as np
import argparse
import pandas as pd
import multiprocessing as mp
from utils.multithread_utils import *
from datetime import datetime
import random
import torch

mp.set_start_method('forkserver', force=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--products',             type=str, default='data/test.txt')
    parser.add_argument('-f', '--save_file',            type=str, default='result/debug.txt')
    parser.add_argument('-b', '--blocks',               type=str, default='data/building_block.csv')
    parser.add_argument('-i', '--iterations',           type=int, default=20)
    parser.add_argument('-e', '--exp_topk',             type=int, default=10)
    parser.add_argument('-k', '--route_topk',           type=int, default=10)
    parser.add_argument('-s', '--beam_size',            type=int, default=10)
    parser.add_argument('-m', '--model_type', type=str, default='ensemble',
                        choices=['ensemble', 'retroformer', 'g2s', 'retriever_only', 'megan', 'GSETransformer'])#
    parser.add_argument('-mp', '--model_path',          type=str, default=None)
    parser.add_argument('-r', '--retrieval',            type=str, default='true', choices=['true', 'false'])
    parser.add_argument('-pr', '--path_retrieval',      type=str, default='false', choices=['true', 'false'])#
    parser.add_argument('-d', '--retrieval_db',         type=str, default='data/train_canonicalized_READRetro.txt')#train_canonicalized_clean
    parser.add_argument('-pd', '--path_retrieval_db',   type=str, default='data/pathways.pickle')
    parser.add_argument('-c', '--device',               type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('-n', '--num_threads',          type=int, default=3)
    parser.add_argument("-seed",                        type=int, default=2024)
    args = parser.parse_args()
    example = ['Cn1cnc(CC(=O)O)c1']
    mo_id = [0]
    get_result = run_multistep(thread_id=0, args=args, mol_ids=mo_id, products=example)
    print(get_result)