#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from data_utils.generate_edge_index import generate_edge_index_pkl
from data_utils.score_result import read_file, match_smiles_lists
# must import rdkit/generate_edge_index before onmt
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus, split_matrix
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import os



def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)  #change
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    # add it here
    edge_index_path = opt.src[:-4] + '_edge_index.pkl'
    if not os.path.exists(edge_index_path):
        generate_edge_index_pkl(opt.src)
    edge_index_shards = split_matrix(edge_index_path, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards, edge_index_shards)

    for i, (src_shard, tgt_shard, edge_index_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug,
            edge_index=edge_index_shard,
            )#add
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
CUDA_VISIBLE_DEVICES=7 python translate.py -model experiments/uspto_50k_auged_x20/model_step_95000.pt   \
-src /home/zhangmeng/aMy-ONMT003/data/aug_uspto_50k_20x/src-test_aug.txt \
-tgt /home/zhangmeng/aMy-ONMT003/data/aug_uspto_50k_20x/tgt-test_aug.txt \
-output /home/zhangmeng/aMy-ONMT003/data/aug_uspto_50k_20x/onmt_pred_model_step_95000_b10n50.txt -replace_unk  -gpu 0  -beam_size 10 -n_best 50

/home/zhangmeng/aMy-ONMT003/
'''
