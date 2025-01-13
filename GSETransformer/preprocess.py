#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import os.path
import sys
import gc
import torch
from GSETransformer.data_utils.generate_edge_index import generate_edge_index_pkl

from functools import partial
from collections import Counter, defaultdict

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus, split_matrix
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab, _load_vocab


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def build_save_dataset(corpus_type, fields, src_reader, tgt_reader, opt, edge_index_reader):
    assert corpus_type in ['train', 'valid']
    if corpus_type == 'train':
        counters = defaultdict(Counter)
        srcs = opt.train_src # a list
        tgts = opt.train_tgt # a list, same below
        edge_index_s = [srcs[0][:-4]+'_edge_index.pkl']
        if not os.path.exists(edge_index_s[0]):
            generate_edge_index_pkl(opt.train_src)
        ids = opt.train_ids
    else:
        srcs = [opt.valid_src]
        tgts = [opt.valid_tgt]
        edge_index_s = [srcs[0][:-4] + '_edge_index.pkl']
        if not os.path.exists(edge_index_s[0]):
            generate_edge_index_pkl(srcs)
        ids = [None]
    for src, tgt, edge_index, maybe_id in zip(srcs, tgts, edge_index_s, ids):
        logger.info("Reading source and target files: %s %s. edge_index file: %s" % (src, tgt, edge_index))

        src_shards = split_corpus(src, opt.shard_size)
        tgt_shards = split_corpus(tgt, opt.shard_size)
        edge_index_shards = split_matrix(edge_index, opt.shard_size)
        shard_pairs = zip(src_shards, tgt_shards, edge_index_shards)#
        dataset_paths = []
        if (corpus_type == "train" or opt.filter_valid) and tgt is not None:
            filter_pred = partial(
                inputters.filter_example, use_src_len=opt.data_type == "text",
                max_src_len=opt.src_seq_length, max_tgt_len=opt.tgt_seq_length)
        else:
            filter_pred = None

        if corpus_type == "train":
            existing_fields = None
            if opt.src_vocab != "":
                try:
                    logger.info("Using existing vocabulary...")
                    existing_fields = torch.load(opt.src_vocab)
                except torch.serialization.pickle.UnpicklingError:
                    logger.info("Building vocab from text file...")
                    src_vocab, src_vocab_size = _load_vocab(
                        opt.src_vocab, "src", counters,
                        opt.src_words_min_frequency)
            else:
                src_vocab = None

            if opt.tgt_vocab != "":
                tgt_vocab, tgt_vocab_size = _load_vocab(
                    opt.tgt_vocab, "tgt", counters,
                    opt.tgt_words_min_frequency)
            else:
                tgt_vocab = None
        for i, (src_shard, tgt_shard, edge_index_shard) in enumerate(shard_pairs):
            assert len(src_shard) == len(tgt_shard)
            assert len(src_shard) == len(edge_index_shard)
            logger.info("Building shard %d." % i)
            dataset = inputters.Dataset(
                fields,
                readers=([src_reader, tgt_reader, edge_index_reader]    #c
                         if tgt_reader else [src_reader, edge_index_reader]),   #c
                data=([("src", src_shard), ("tgt", tgt_shard), ("edge_index", edge_index_shard)]    #c
                      if tgt_reader else [("src", src_shard), ("edge_index", edge_index_shard)]),   #c
                dirs=([opt.src_dir, None, None] #c
                      if tgt_reader else [opt.src_dir, None]),  #c
                sort_key=inputters.str2sortkey[opt.data_type],
                filter_pred=filter_pred
            )
            # for ex_i, ex_ctt in enumerate(dataset.examples):
            #     print('got here',ex_ctt.__dict__)
            #     break
            if corpus_type == "train" and existing_fields is None:
                for ex in dataset.examples:
                    for name, field in fields.items():
                        try:
                            f_iter = iter(field)
                        except TypeError:
                            f_iter = [(name, field)]
                            all_data = [getattr(ex, name, None)]
                        else:
                            all_data = getattr(ex, name)
                        for (sub_n, sub_f), fd in zip(
                                f_iter, all_data):
                            has_vocab = (sub_n == 'src' and src_vocab) or \
                                        (sub_n == 'tgt' and tgt_vocab)
                            if (hasattr(sub_f, 'sequential')
                                    and sub_f.sequential and not has_vocab):
                                val = fd
                                counters[sub_n].update(val)
            if maybe_id:
                shard_base = corpus_type + "_" + maybe_id
            else:
                shard_base = corpus_type
            data_path = "{:s}.{:s}.{:d}.pt".\
                format(opt.save_data, shard_base, i)
            dataset_paths.append(data_path)



            logger.info(" * saving %sth %s data shard to %s."
                        % (i, shard_base, data_path))

            dataset.save(data_path)
            # check # check 是否添加成功
            # for ex_i, ex_ctt in enumerate(dataset.examples):
            #     print('got here',ex_ctt.__dict__)
            #     break
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        if existing_fields is None:
            fields = _build_fields_vocab(
                fields, counters, opt.data_type,
                opt.share_vocab, opt.vocab_size_multiple,
                opt.src_vocab_size, opt.src_words_min_frequency,
                opt.tgt_vocab_size, opt.tgt_words_min_frequency)
        else:
            fields = existing_fields
        torch.save(fields, vocab_path)


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    if not(opt.overwrite):
        check_existing_pt_files(opt)

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = 0
    tgt_nfeats = 0
    for src, tgt in zip(opt.train_src, opt.train_tgt):
        src_nfeats += count_features(src) if opt.data_type == 'text' \
            else 0
        tgt_nfeats += count_features(tgt)  # tgt always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        tgt_nfeats,
        dynamic_dict=opt.dynamic_dict,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc)
    #fields['graph'] = torchtext.data.Field(sequential=False)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    tgt_reader = inputters.str2reader["text"].from_opt(opt)
    edge_index_reader = inputters.str2reader["graph"].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset(
        'train', fields, src_reader, tgt_reader, opt, edge_index_reader)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, src_reader, tgt_reader, opt, edge_index_reader)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
'''#
python preprocess.py -train_src data/final_biochem_npl_clean_20xaug_RSmiles/src-train.txt -train_tgt data/final_biochem_npl_clean_20xaug_RSmiles/tgt-train.txt \
-valid_src data/final_biochem_npl_clean_20xaug_RSmiles/src-val.txt  -valid_tgt data/final_biochem_npl_clean_20xaug_RSmiles/tgt-val.txt  \
-save_data data/final_biochem_npl_clean_20xaug_RSmiles/final_biochem_npl_clean_20xaug_RSmiles  \
-src_seq_length 1000 -tgt_seq_length 1000 -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
'''