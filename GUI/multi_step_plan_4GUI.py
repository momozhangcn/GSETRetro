import os
import sys
import pandas as pd
from retro_star.api import RSPlanner
from time import time
import argparse

def run_Muiti_Step_Plan(input_smi, model_type, beam_size, exp_topk, iterations,route_topk, device, model_path=None):
    parser = argparse.ArgumentParser()
    root_dir_path = os.path.dirname(os.path.dirname(__file__))
    #default parameters
    #parser.add_argument('product',                      type=str, default='O=C1C=C2C=CC(O)CC2O1')
    parser.add_argument('-b', '--blocks',               type=str, default=f'{root_dir_path}/data/building_block.csv')
    parser.add_argument('-i', '--iterations',           type=int, default=20)
    parser.add_argument('-e', '--exp_topk',             type=int, default=10)
    parser.add_argument('-k', '--route_topk',           type=int, default=10)
    parser.add_argument('-s', '--beam_size',            type=int, default=10)
    parser.add_argument('-m', '--model_type', type=str, default='GSETransformer',
                        choices=['AugTransformer', 'Chemformer', 'GTA', 'megan', 'TTWTransformer',
                                 'GSETransformer', 'retriever_only', 'GSETransformer+Retriver'])
    parser.add_argument('-mp', '--model_path',          type=str,
                        default=f'{root_dir_path}/GSETransformer/experiments/biochem_npl_20xaug/model_best_acc_step_355000.pt')
    parser.add_argument('-r', '--retrieval',            type=str, default='true', choices=['true', 'false'])
    parser.add_argument('-pr', '--path_retrieval',      type=str, default='true', choices=['true', 'false'])
    parser.add_argument('-d', '--retrieval_db',         type=str, default=f'{root_dir_path}/data/train_canonicalized_full.txt')
    parser.add_argument('-pd', '--path_retrieval_db',   type=str, default=f'{root_dir_path}/data/pathways.pickle')
    parser.add_argument('-c', '--device',               type=str, default='cpu', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    # update parameters
    args.model_type = model_type
    args.beam_size = beam_size
    args.exp_topk = exp_topk
    args.iterations = iterations
    args.route_topk = route_topk
    args.device = device.lower()
    if model_path != None:
        args.model_path = model_path
    print(f'Multi-step planning args: {args}')
    print(F'Loading model from  {args.model_path}')

    t_start = time()
    planner = RSPlanner(
        cuda=args.device=='cuda',
        iterations=args.iterations,
        expansion_topk=args.exp_topk,
        route_topk=args.route_topk,
        beam_size=args.beam_size,
        model_type=args.model_type,
        model_path=args.model_path,
        retrieval=args.retrieval=='true',
        retrieval_db=args.retrieval_db,
        path_retrieval=args.path_retrieval=='true',
        path_retrieval_db=args.path_retrieval_db,
        starting_molecules=args.blocks,
    )
    #args.product = input_smi[0]
    all_result = []
    for i0, single_target_mole in enumerate(input_smi):
        single_result = planner.plan(single_target_mole)
        if single_result != None:
            print(f'Get predicted pathway(s) for No.{i0+1} target molecule')
            for i, route in enumerate(single_result):
                print(f'{i+1} {route}')
            all_result.extend(single_result)

    result = pd.DataFrame(data=all_result, columns=['pathway_prediction'])
    print(f'\033[92mTotal {time() - t_start:.2f} sec elapsed\033[0m')
    return result
