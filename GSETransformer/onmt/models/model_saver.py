import os
import torch
import torch.nn as nn

from collections import deque
from onmt.utils.logging import logger

from copy import deepcopy


def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)
            #self.best_model_checkpoint_queue = deque([], maxlen=keep_checkpoint)
            keep_checkpoint4best_model = max(keep_checkpoint//2, 1)
            self.model_best_ppl_checkpoint_queue = deque([], maxlen=keep_checkpoint4best_model)
            self.model_best_acc_checkpoint_queue = deque([], maxlen=keep_checkpoint4best_model)

    def save(self, step, moving_average=None, best_step=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """
        if self.keep_checkpoint == 0 or all([step == self.last_saved_step, best_step == None]):
            return

        if moving_average:
            save_model = deepcopy(self.model)
            for avg, param in zip(moving_average, save_model.parameters()):
                param.data.copy_(avg.data)
        else:
            save_model = self.model

        assert best_step in [None, 'ppl', 'acc']
        chkpt, chkpt_name = self._save(step, save_model, best_step)
        self.last_saved_step = step

        if moving_average:
            del save_model

        if self.keep_checkpoint > 0:
            # best_checkpoint = '%s_step_%d.pt' % (self.base_path, step)
            # if best_step is not None:
            #     best_checkpoint = '%s_step_%d.pt' % (self.base_path, best_step)
            if best_step == None:
                if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                    todel = self.checkpoint_queue.popleft()
                    self._rm_checkpoint(todel)
                self.checkpoint_queue.append(chkpt_name)
            elif best_step == 'ppl':
                if len(self.model_best_ppl_checkpoint_queue) == self.model_best_ppl_checkpoint_queue.maxlen:
                    best_ppl_model_todel = self.model_best_ppl_checkpoint_queue.popleft()
                    self._rm_checkpoint(best_ppl_model_todel)
                self.model_best_ppl_checkpoint_queue.append(chkpt_name)
            elif best_step == 'acc':
                if len(self.model_best_acc_checkpoint_queue) == self.model_best_acc_checkpoint_queue.maxlen:
                    best_acc_model_todel = self.model_best_acc_checkpoint_queue.popleft()
                    self._rm_checkpoint(best_acc_model_todel)
                self.model_best_acc_checkpoint_queue.append(chkpt_name)

    def _save(self, step, model, best_step):# add ', model, best_step'
        """Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model, best_step):
        real_model = (model.module
                      if isinstance(model, nn.DataParallel)
                      else model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()

        # NOTE: We need to trim the vocab to remove any unk tokens that
        # were not originally here.

        vocab = deepcopy(self.fields)
        for side in ["src", "tgt"]:
            keys_to_pop = []
            if hasattr(vocab[side], "fields"):
                unk_token = vocab[side].fields[0][1].vocab.itos[0]
                for key, value in vocab[side].fields[0][1].vocab.stoi.items():
                    if value == 0 and key != unk_token:
                        keys_to_pop.append(key)
                for key in keys_to_pop:
                    vocab[side].fields[0][1].vocab.stoi.pop(key, None)

        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': vocab,
            'opt': self.model_opt,
            'optim': self.optim.state_dict(),
        }

        if best_step == None:
            logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
            checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        else:       # name model witth best_step == ppl or acc
            logger.info(f"Saving checkpoint {self.base_path}_best_{best_step}_step_{step}.pt")
            checkpoint_path = f'{self.base_path}_best_{best_step}_step_{step}.pt'

        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)
