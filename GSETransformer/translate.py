#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat
from data_utils.generate_edge_index import generate_edge_index_pkl
# must import rdkit/generate_edge_index before onmt
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus, split_matrix
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import os


def DIY_evaluate(output_path, target_path):
    with open(target_path, 'r') as tgt_file:
        with open(output_path, 'r') as prdt_file:
            count = 0
            i_seq = []
            tgt_lst = tgt_file.readlines()
            prdt_lst = prdt_file.readlines()

            n_beam = int(len(prdt_lst) / len(tgt_lst))
            print(f'beam = {n_beam}')
            if n_beam < 20:
                check_lst = [1, 3, 5, 10]
            elif 20 <= n_beam < 50:
                check_lst = [1, 3, 5, 10, 20]
            else:
                check_lst = [1, 3, 5, 10, 20, 50]
            for k in check_lst:  # ,15,20,20,50
                count_k = 0
                count_correct = 0
                for i, line in enumerate(tgt_lst):
                    line = line.replace(' ', '').strip('\n')  # 去除空格#
                    left_b = i * n_beam
                    right_b = i * n_beam + k  # +1
                    correct_pred = False
                    for aline in prdt_lst[left_b:right_b]:
                        # for h,chrs in enumerate(aline):
                        #     if chrs ==',':
                        #         aline = aline[:h]
                        aline = aline.replace(' ', '').strip('\n')  # 去除空格#
                        # print(aline)
                        if aline == line:
                            # print(i)
                            count_k += 1
                            correct_pred = True  # 存在正确预测，洗
                            break
                acc = (count_k / len(tgt_lst)) * 100
                print(f'top_{k:2d}, acc:{acc:5.2f}%')

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
        DIY_evaluate(opt.output, opt.tgt)


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
