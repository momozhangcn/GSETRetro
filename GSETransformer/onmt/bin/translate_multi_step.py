#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat
from data_utils.generate_edge_index import get_single_edge_index
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)  # change

    # src_shards = split_corpus(opt.src, opt.shard_size)
    # tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
    #     if opt.tgt is not None else repeat(None)
    # # add it here
    # edge_index_path = opt.src[:-4] + '_edge_index.pkl'
    # if not os.path.exists(edge_index_path):
    #     generate_edge_index_pkl(opt.src)
    # edge_index_shards = split_matrix(edge_index_path, opt.shard_size)
    src_shards = opt.src
    tgt_shards = repeat(None)
    edge_index_shards = [get_single_edge_index(smi) for smi in opt.src]
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
        )  # add

def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser

if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    opt.model = ''
    opt.src = []
    translate(opt)
