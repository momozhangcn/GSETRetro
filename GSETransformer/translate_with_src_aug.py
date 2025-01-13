#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import os
import sys
current_folder = os.path.dirname(__file__)
if current_folder not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from GSETransformer.data_utils.generate_edge_index import get_single_edge_index, generate_edge_index_pkl
from GSETransformer.data_utils.score_result import read_file, match_smiles_lists
from GSETransformer.data_utils.src_aug_res_rerank import \
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
        src_aug_time = opt.src_aug_time
        for src in tqdm(src_all):
            src = smi_tokenizer(src.strip().replace(' ', ''))
            
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
                
                if len(rank_pred) < opt.n_best:
                    rank_pred.extend(['C' for i in range(opt.n_best-len(rank_pred))])
                scores_cano = list(np.exp(rank_score))
                sum_scores = sum(scores_cano)
                cano_scores = [score / sum_scores for score in scores_cano]

            else:
                all_scores = all_scores[0]
                all_predictions = all_predictions[0]

                all_predictions_cano = [canonicalize_smiles(smi) for smi in all_predictions]
                
                scores = [float(each) for each in all_scores]
                all_scores_cano = list(np.exp(scores))
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
    main(opt)


