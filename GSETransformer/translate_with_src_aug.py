#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat
try:
    from .data_utils.generate_edge_index import get_single_edge_index, generate_edge_index_pkl
    from .data_utils.score_result import read_file, match_smiles_lists
    from .data_utils.src_aug_res_rerank import \
        atom_map_src_smi, atom_mapped_src_aug, compute_rank_rerank, smi_tokenizer, canonicalize_smiles
except:
    from data_utils.generate_edge_index import get_single_edge_index, generate_edge_index_pkl
    from data_utils.score_result import read_file, match_smiles_lists
    from data_utils.src_aug_res_rerank import \
        atom_map_src_smi, atom_mapped_src_aug, compute_rank_rerank, smi_tokenizer, canonicalize_smiles

# must import rdkit/generate_edge_index before onmt
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus, split_matrix
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm


def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)
    opt.multi_step = True
    
    translator = build_translator(opt, report_score=True)  # change
    with open(opt.src, 'r') as f_src_in, open(opt.output, 'w',encoding="utf-8") as f_pred_out:
        src_all = f_src_in.readlines()
        for src in tqdm(src_all):
            src = smi_tokenizer(src.strip().replace(' ', ''))

            src_aug_time = 10
            if src_aug_time > 1:
                # step_01, do atom map for src smi
                mapped_src_smi = atom_map_src_smi(src)
                # step_02, do root-change aug for mapped src smi
                auged_src_lst, _ = atom_mapped_src_aug(mapped_src_smi, src_aug_time)
                # step_03, do prediction for root-change auged smi  and get raw result
                src = auged_src_lst
            else:
                src = [src]

            src_shards = src
            edge_index_shards = [get_single_edge_index(smi) for smi in src]

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
                all_predictions_cano = [[canonicalize_smiles(smi) for smi in all_predictions[i]] for i in
                                        range(len(all_predictions))]
                # step_04, rerank the result
                
                rank_pred, rank_score = compute_rank_rerank(all_predictions_cano, opt.n_best) 
                cano_preds = rank_pred
                
                if len(rank_pred) <10:
                    print(rank_pred)
                scores_cano = list(np.exp(rank_score))
                sum_scores = sum(scores_cano)
                cano_scores = [score / sum_scores for score in scores_cano]

            else:
                all_scores = all_scores[0]
                all_predictions = all_predictions[0]

                all_predictions_cano = [canonicalize_smiles(smi) for smi in all_predictions]
                scores = [float(each) for each in all_scores]
                all_scores_cano = list(np.exp(scores))
                print(all_predictions_cano)
                cano_preds = all_predictions_cano[:opt.n_best]
                sum_scores = sum(all_scores_cano)
                cano_scores = [score / sum_scores for score in all_scores_cano][:opt.n_best]
                
            for pred in cano_preds:
                f_pred_out.writelines(pred+'\n')
    if opt.tgt:
        smiles_list, target_list, beam = read_file(opt.output, opt.tgt)
        match_smiles_lists(smiles_list, target_list, beam)



def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    # edge_index_path = opt.src[:-4] + '_edge_index.pkl'
    # if not os.path.exists(edge_index_path):
    #     generate_edge_index_pkl(opt.src)

    main(opt)

'''/////data_sdd_Biochem//////data_sdd_uspto//////
#gset
CUDA_VISIBLE_DEVICES=3 python translate.py -model experiments/uspto_50k_auged_x20_256stop/model_step_40000.pt   \
-src data/aug_uspto_50k_20x/src-test_aug.txt -tgt data/aug_uspto_50k_20x/tgt-test_aug.txt \
-output data/aug_uspto_50k_20x/GSET_pred_model_step_40000_b50n50.txt -replace_unk  -gpu 0  -beam_size 50 -n_best 50 
#onmt
CUDA_VISIBLE_DEVICES=7 python translate.py 

/home/zhangmeng/aMy-ONMT003/
'''
