#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DST on Google The Schema-Guided Dialogue(SGD) Dataset implementation for ParlAI.
"""

import os
import json
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, BleuMetric
from .utils.reformat import reformat_parlai

from .build import build

def _path(opt):
    # set up path to data (specific to each dataset)
    data_dir = os.path.join(opt['datapath'], 'google_sgd_dst')
    data_path = os.path.join(data_dir, 'data_reformat.json')

    # build the data if it does not exist
    build(opt)

    # reformat data for DST
    reformat_parlai(data_dir)

    return data_path, data_dir


class Google_SGD_DST_Teacher(FixedDialogTeacher):
    """
    MultiWOZ DST Teacher.
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        opt['datafile'], jsons_path = _path(opt)
        self._setup_data(opt['datafile'], jsons_path)
        self.id = 'google_sgd_dst'
        self.reset()

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _setup_data(self, data_path, jsons_path):
        print('loading: ' + data_path)

        # # # loading directly from test file or val file
        if self.datatype.startswith('test'):
            test_path = data_path.replace(".json", "_test.json")
            test_data = self._load_json(test_path)
            self.messages = list(test_data.values())
        elif self.datatype.startswith('valid'):
            valid_path = data_path.replace(".json", "_dev.json")
            valid_data = self._load_json(valid_path)
            self.messages = list(valid_data.values())
        else:
            train_path = data_path.replace(".json", "_train.json")
            train_data = self._load_json(train_path)
            self.messages = list(train_data.values())


    def num_examples(self):
        examples = 0
        for data in self.messages:
            examples += len(data)
        return examples

    def num_episodes(self):
        return len(self.messages)

    def get(self, episode_idx, entry_idx=0):
        # log_idx = entry_idx
        entry = self.messages[episode_idx][entry_idx]['context']
        episode_done = entry_idx == len(self.messages[episode_idx]) - 1
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': [self.messages[episode_idx][entry_idx]['slots_inf']],
        }
        return action

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom slot_type slot_val", ... ]
        """
        slots_list = []

        # # # split according to ","
        str_split = slots_string.split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        slots_list = [slot.strip() for slot in str_split]

        return slots_list

    def custom_evaluation(self, teacher_action: Message, labels, model_response: Message):
        resp = model_response.get('text')
        if not resp:
            return

        # # # extract ground truth from labels
        slots_truth = self._extract_slot_from_string(labels[0])
        
        # # # extract generated slots from model_response
        slots_pred = self._extract_slot_from_string(resp)

        self.metrics.add('joint goal acc', AverageMetric(set(slots_truth) == set(slots_pred)))

class DefaultTeacher(Google_SGD_DST_Teacher):
    pass
