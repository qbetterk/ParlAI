#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import os
import pdb

import torch
from parlai.agents.nar.modules import NarModel
from parlai.agents.bart.bart import BartAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel, PPLMetric
from parlai.core.agents import compare_init_model_opts
from parlai.utils.io import PathManager
from parlai.utils.misc import warn_once
from parlai.utils.torch import IdentityLayer, padded_tensor
from parlai.utils.typing import TShared

from parlai.core.metrics import (
    Metric,
    SumMetric,
    AverageMetric,
    BleuMetric,
    FairseqBleuMetric,
)

class NarAgent(BartAgent):
    """
    BART Agent.

    Relies on the BART model implemented in fairseq.

    If you have a fine-tuned BART model from fairseq, you can specify the
    `--init-fairseq-model` arg, which will convert your fine-tuned model
    to a ParlAI model.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        super().__init__(opt, shared)
        torch.autograd.set_detect_anomaly(True)

    def build_model(self) -> NarModel:
        """
        Build and return model.
        """
        model = NarModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model


    def compute_loss(self, batch, return_output=False):
        """
        Override TGA.compute_loss to ignore start token.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, encoder_states, *_ = model_output

        if scores.size(1) != batch.label_vec.size(1):
            # ignore start
            scores = scores[:, 1:, :]
            preds = preds[:, 1:]

        # cross entropy
        score_view = scores.reshape(-1, scores.size(-1))
        nll_loss = self.criterion(score_view, batch.label_vec.view(-1))
        nll_loss = nll_loss.view(scores.shape[:-1]).sum(dim=1)
        loss = nll_loss

        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        # length loss
        if batch['labels'] is not None:
            length_lprobs = encoder_states[-1]
            # length_target = torch.LongTensor([[len(label.split(",")) - 1] for label in batch['labels']]).to(length_lprobs.device)
            length_target = target_tokens.unsqueeze(-1)
            length_target[length_target < 0] = 0
            len_loss = -length_lprobs.gather(dim=-1, index=length_target)
            self.record_local_metric('len_loss', AverageMetric.many(len_loss, target_tokens.new_ones(target_tokens.shape)))

            # total loss by summing up
            loss += torch.squeeze(len_loss) *10

        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('nll_loss', AverageMetric.many(nll_loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric('token_acc', AverageMetric.many(correct, target_tokens))
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def _construct_token_losses(self, labels, model_output):
        """
        Override TGA._construct_token_losses to ignore start token.
        """
        # Get non-aggregated losses
        scores, _, _ = model_output
        scores = scores[:, 1:, :]  # ignore start token
        score_view = scores.reshape(-1, scores.size(-1))
        losses = self.criterion(score_view, labels.view(-1)).view(len(labels), -1)

        # Zip decoded tokens with losses
        token_losses = []
        for i, label in enumerate(labels):
            token_losses.append(
                list(
                    zip(
                        [self.dict[token] for token in label.tolist()],
                        losses[i].tolist(),
                    )
                )
            )
        return token_losses