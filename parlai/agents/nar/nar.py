#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import os

import torch
from parlai.agents.nar.modules import NARModel
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.utils.io import PathManager
from parlai.utils.misc import warn_once
from parlai.utils.torch import IdentityLayer, padded_tensor


class NARAgent(TransformerGeneratorAgent):
    """
    BART Agent.

    Relies on the BART model implemented in fairseq.

    If you have a fine-tuned BART model from fairseq, you can specify the
    `--init-fairseq-model` arg, which will convert your fine-tuned model
    to a ParlAI model.
    """


