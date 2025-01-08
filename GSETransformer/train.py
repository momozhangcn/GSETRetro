#!/usr/bin/env python
"""Train models."""
import os
import signal
import torch

import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.train_single import main as single_main
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple

from itertools import cycle
'''
-data data/biochem/biochem_token  -save_model experiments/biochem -seed 2020 -gpu_ranks 0 -batch_size 1024 -save_checkpoint_steps 5000  -keep_checkpoint 41  -train_steps 200000 -valid_steps 2000 -report_every 1000  -param_init 0 -param_init_glorot   -batch_type tokens -normalization tokens   -dropout 0.3 -max_grad_norm 0 -accum_count 4  -optim adam -adam_beta1 0.9 -adam_beta2 0.998  -decay_method noam -warmup_steps 8000  -learning_rate 2 -label_smoothing 0.0  -enc_layers 6 -dec_layers 6 -rnn_size 256 -word_vec_size 256  -encoder_type transformer -decoder_type transformer  -share_embeddings -position_encoding -max_generator_batches 0   -global_attention general -global_attention_function softmax  -self_attn_type scaled-dot -max_relative_positions 4 -heads 8 -transformer_ff 2048  -early_stopping 30  
'''

def main(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    if len(opt.data_ids) > 1:
        train_shards = []
        for train_id in opt.data_ids:
            shard_base = "train_" + train_id
            train_shards.append(shard_base)
        train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
    else:
        if opt.data_ids[0] is not None:
            shard_base = "train_" + opt.data_ids[0]
        else:
            shard_base = "train"
        train_iter = build_dataset_iter(shard_base, fields, opt)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            q = mp.Queue(opt.queue_size)
            queues += [q]
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, q, semaphore), daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producer = mp.Process(target=batch_producer,
                              args=(train_iter, queues, semaphore, opt,),
                              daemon=True)
        producer.start()
        error_handler.add_child(producer.pid)

        for p in procs:
            p.join()
        producer.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def batch_producer(generator_to_serve, queues, semaphore, opt):
    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)
    # generator_to_serve = iter(generator_to_serve)

    def pred(x):
        """
        Filters batches that belong only
        to gpu_ranks of current node
        """
        for rank in opt.gpu_ranks:
            if x[0] % opt.world_size == rank:
                return True

    generator_to_serve = filter(
        pred, enumerate(generator_to_serve))

    def next_batch(device_id):
        new_batch = next(generator_to_serve)
        semaphore.acquire()
        return new_batch[1]

    b = next_batch(0)

    for device_id, q in cycle(enumerate(queues)):
        b.dataset = None
        if isinstance(b.src, tuple):
            b.src = tuple([_.to(torch.device(device_id))
                           for _ in b.src])
        else:
            b.src = b.src.to(torch.device(device_id))
        b.tgt = b.tgt.to(torch.device(device_id))
        b.indices = b.indices.to(torch.device(device_id))
        b.alignment = b.alignment.to(torch.device(device_id)) \
            if hasattr(b, 'alignment') else None
        b.src_map = b.src_map.to(torch.device(device_id)) \
            if hasattr(b, 'src_map') else None

        # hack to dodge unpicklable `dict_keys`
        b.fields = list(b.fields)
        q.put(b)
        b = next_batch(device_id)


def run(opt, device_id, error_queue, batch_queue, semaphore):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id, batch_queue, semaphore)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
'''
CUDA_VISIBLE_DEVICES=5 python  train.py -data data/final_biochem_npl_clean_20xaug_RSmiles/final_biochem_npl_clean_20xaug_RSmiles \
                 -save_model experiments/final_biochem_npl_clean_20xaug_RSmiles/model \
                 -seed 2024 -gpu_ranks 0 \
                 -save_checkpoint_steps 5000  \
                 -train_steps 500000 -valid_steps 5000 -report_every 1000 \
                 -param_init 0 -param_init_glorot \
                 -batch_size 4096 -batch_type tokens -normalization tokens \
                 -dropout 0.3 -max_grad_norm 0 -accum_count 4 \
                 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 \
                 -decay_method noam -warmup_steps 8000  \
                 -learning_rate 2 -label_smoothing 0.0 \
                 -enc_layers 6 -dec_layers 6 -rnn_size 512 -word_vec_size 512 \
                 -encoder_type transformer -decoder_type transformer \
                 -share_embeddings -position_encoding -max_generator_batches 32 \
                 -global_attention general -global_attention_function softmax \
                 -self_attn_type scaled-dot -max_relative_positions 4 \
                 -heads 8 -transformer_ff 2048   -early_stopping 20 -keep_checkpoint 10 \
                 -tensorboard -tensorboard_log_dir runs/final_biochem_npl_clean_20xaug_RSmiles 2>&1 | tee runs/final_biochem_npl_clean_20xaug_RSmiles.log

# -keep_checkpoint 31 #-early_stopping 20


'''