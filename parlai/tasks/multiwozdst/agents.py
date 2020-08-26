#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DST on Multiwoz2.1 Dataset implementation for ParlAI.
"""
import sys, os
import json

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from .build import build
from .utils.trade_proc import trade_process
from .utils.reformat import reformat_parlai


def _path(opt):
    # set up path to data (specific to each dataset)
    data_dir = os.path.join(opt['datapath'], 'multiwozdst', 'MULTIWOZ2.1')
    # data_dir = os.path.join('/checkpoint/kunqian/multiwoz/data/MultiWOZ_2.1/')
    data_path = os.path.join(data_dir, 'data_reformat_trade_turn_sa_ha.json')

    # build the data if it does not exist
    build(opt)

    # process the data with TRADE's code, if it does not exist
    if not os.path.exists(os.path.join(data_dir, 'dials_trade.json')):
        trade_process(data_dir)

    # reformat data for DST
    if not os.path.exists(data_path):
        reformat_parlai(data_dir)

    return data_path, data_dir


class MultiWozDSTTeacher(FixedDialogTeacher):
    """
    MultiWOZ DST Teacher.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt['datafile'], jsons_path = _path(opt)
        self.id = 'multiwozdst'

        # # # reading args
        self.decode_all = opt.get('decode_all', False)
        self.just_test = opt.get('just_test', False)

        # import pdb
        # pdb.set_trace()
        self._setup_data(opt['datafile'], jsons_path)

        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('MultiWozDST Teacher Args')
        agent.add_argument(
            '-dall',
            '--decode_all',
            type='bool',
            default=False,
            help="True if one would like to decode dst for all samples in training data, probably for \
            training a correction model (default: False).",
        )
        agent.add_argument(
            '--just_test',
            type='bool',
            default=False,
            help="True if one would like to test agents with small amount of data (default: False).",
        )

    def _load_txt(self, file_path):
        with open(file_path) as df:
            data=df.read().lower().split("\n")
            data.remove('')
        return data

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _setup_data(self, data_path, jsons_path):
        # # # loading directly from test file or val file
        if self.decode_all:
            all_data = self._load_json(data_path)
            self.messages = list(all_data.values())
        elif self.datatype.startswith('test'):
            test_path = data_path.replace(".json", "_test.json")
            test_data = self._load_json(test_path)
            self.messages = list(test_data.values())
        elif self.datatype.startswith('valid'):
            valid_path = data_path.replace(".json", "_valid.json")
            valid_data = self._load_json(valid_path)
            self.messages = list(valid_data.values())
        else:
            train_path = data_path.replace(".json", "_train.json")
            train_data = self._load_json(train_path)
            self.messages = list(train_data.values())
        
        if self.just_test:
            self.messages = self.messages[:10]

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
        domains    = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]
        # slot_types = ['book time', 'leaveat', 'name', 'internet', 'book stay', 
        #               'pricerange', 'arriveby', 'area', 'destination', 'day', 
        #               'food', 'departure', 'book day', 'book people', 'department', 
        #               'stars', 'parking', 'type']
        slot_types = ["stay", "price", "addr",  "type", "arrive", "day", "depart", "dest",
                    "area", "leave", "stars", "department", "people", "time", "food", 
                    "post", "phone", "name", 'internet', 'parking',
                    'book stay', 'book people','book time', 'book day',
                    'pricerange', 'destination', 'leaveat', 'arriveby', 'departure']
        slots_list = []

        # # # remove start and ending token
        str_split = slots_string.strip().split()
        if str_split != [] and str_split[0] in ["<bs>", "</bs>"]:
            str_split = str_split[1:]
        if "</bs>" in str_split:
            str_split = str_split[:str_split.index("</bs>")]

        # # # split according to ","
        str_split = " ".join(str_split).split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in domains:
                domain = slot[0]
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1]+" "+slot[2]
                    slot_val  = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val  = " ".join(slot[2:])
                if not slot_val == 'dontcare':
                    slots_list.append(domain+"--"+slot_type+"--"+slot_val)
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

    def num_examples(self):
        # each turn be seen as a individual dialog
        return len(self.messages)

    def num_episodes(self):
        return len(self.messages)

    def get(self, episode_idx, entry_idx=0):
        # log_idx = entry_idx
        entry = self.messages[episode_idx]['context']
        episode_done = True
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': [self.messages[episode_idx]['slots_inf']],
            'dial_id': self.messages[episode_idx]['dial_id'],
            'turn_num': self.messages[episode_idx]['turn_num'],
        }
        return action


class DefaultTeacher(MultiWozDSTTeacher):
    """
    Default teacher.
    """
    pass
