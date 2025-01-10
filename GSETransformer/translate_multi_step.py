#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import os
import sys

root = os.path.abspath('.')
if root not in sys.path:
    sys.path.insert(0, root)
elif os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

try:
    from .data_utils.generate_edge_index import get_single_edge_index
    from .data_utils.src_aug_res_rerank import \
        atom_map_src_smi, atom_mapped_src_aug, compute_rank_rerank, smi_tokenizer, canonicalize_smiles
except:
    from data_utils.generate_edge_index import get_single_edge_index
    from data_utils.src_aug_res_rerank import \
        atom_map_src_smi, atom_mapped_src_aug, compute_rank_rerank, smi_tokenizer, canonicalize_smiles

from onmt.utils.logging import init_logger
#from custom_onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import numpy as np


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser

def prepare_GSET_translator(model_path,  beam, expansion_topk, device_bool):
    parser = _get_parser()
    opt = parser.parse_args()

    if isinstance(model_path, str):
        model_path = [model_path]
    opt.models = model_path
    opt.gpu = device_bool-1
    # if device == 0:
    #     opt.gpu = -1
    # elif device == 1:
    #     opt.gpu = 0
    opt.beam = beam
    opt.n_best = beam
    #opt.batch_size = 1
    opt.expansion_topk = expansion_topk
    opt.multi_step = True
    ArgumentParser.validate_translate_opts(opt)
    translator = build_translator(opt, report_score=True)  # change
    return translator, opt

def run_GSET_translate(src, translator, opt):

    ## smiles→ 1.SMART → 2. 5x aug(RSMILES aug or random aug)→ 3. got result 50x→ 4. 10x ,ok done
    if isinstance(src, str):
        src = smi_tokenizer(src.strip().replace(' ', ''))

    src_aug_time = 10
    if src_aug_time > 1:
        # step_01, do atom map for src smi
        mapped_src_smi = atom_map_src_smi(src)
        # step_02, do root-change aug for mapped src smi
        auged_src_lst, _ = atom_mapped_src_aug(mapped_src_smi, src_aug_time)
        # step_03, do prediction for root-change auged smi  and get raw result
        opt.src = auged_src_lst
    else:
        opt.src = [src]
    # prediction
    src_shards = opt.src
    edge_index_shards = [get_single_edge_index(smi) for smi in opt.src]
    all_scores, all_predictions = translator.translate(
        src=src_shards,
        tgt=None,
        src_dir=opt.src_dir,
        batch_size=opt.batch_size,
        attn_debug=opt.attn_debug,
        edge_index=edge_index_shards,
    )
    # result_process
    if src_aug_time > 1:
        all_predictions_cano = [[canonicalize_smiles(smi) for smi in all_predictions[i]] for i in range(len(all_predictions))]
        # step_04, rerank the result
        rank_pred, rank_score = compute_rank_rerank(all_predictions_cano, opt.expansion_topk)
        cano_preds = rank_pred

        scores_cano = list(np.exp(rank_score))
        sum_scores = sum(scores_cano)
        cano_scores = [score / sum_scores for score in scores_cano]
        # sum_scores = sum(rank_score)
        # cano_scores = [score / sum_scores for score in rank_score]
    else:
        all_scores = all_scores[0]
        all_predictions = all_predictions[0]

        all_predictions_cano = [canonicalize_smiles(smi) for smi in all_predictions]
        scores = [float(each) for each in all_scores]
        all_scores_cano = list(np.exp(scores))

        filter_preds = []
        filter_scores = []
        for pred_smi, pred_score in zip(all_predictions_cano, all_scores_cano):
            if pred_smi != '' and pred_smi not in filter_preds:#去除无效&重复
                filter_preds.append(pred_smi)
                filter_scores.append(pred_score)

        cano_preds = filter_preds[:opt.expansion_topk]
        sum_scores = sum(filter_scores)
        cano_scores = [score / sum_scores for score in filter_scores][:opt.expansion_topk]


    res_dict = {}
    res_dict['reactants'] = cano_preds#preds_cano
    res_dict['scores'] = cano_scores#scores_cano
    templates = []
    res_dict['templates'] = [None for _ in range(len(res_dict['scores']))]
    res_dict['retrieved'] = [False for _ in range(len(res_dict['scores']))]

    return res_dict#, {'reactants': generations, 'scores': scores}

if __name__ == "__main__":
    parser = _get_parser()
    gset_model_path = ['/home/zhangmeng/aMy-ONMT010/experiments/final_biochem_npl_20xaug_RSmiles/model_best_ppl_step_140000.pt']
    device = 'cuda'
    translator, opt = prepare_GSET_translator(gset_model_path, 10, 10, device)
    smiles_lst = 'CC(=O)N/C=C(/CC(=O)O)C(=O)OSP'
    #smiles_lst = ['CC(=O)N/C=C(/CC(=O)O)C(=O)OSP', 'O=C(O)C(=O)CS']
    res = run_GSET_translate(smiles_lst, translator, opt)

    print(res)

    # t_start = time()
    # model, args, vocab, vocab_tokens, device = prepare_g2s()
    # # smi = 'N[C@@H](CNC(=O)C(=O)O)C(=O)O'
    # smi = 'C=CC(C)(C)[C@@]12C[C@H]3c4nc5ccccc5c(=O)n4[C@H](C)C(=O)N3C1N(C(C)=O)c1ccccc12'
    # print(run_g2s(model, args, smi, vocab, vocab_tokens, device))
    # print(f'\033[92mTotal {time() - t_start:.2f} sec elapsed\033[0m')

