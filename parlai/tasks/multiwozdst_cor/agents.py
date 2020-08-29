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
# from .build import build
# from .utils.trade_proc import trade_process
from .utils.split_decoded_data import split_decoded_data
from .utils.add_err import AddErr




class MultiWozDSTCORTeacher(FixedDialogTeacher):
    """
    MultiWOZ DST Correction Teacher.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'multiwozdst_cor'

        # # # loading args
        self.decode_all = opt.get('decode_dall', False)
        self.just_test = opt.get('just_test', False)
        self.repeat_minor_err = opt.get('repeat_minor_err', False)
        self.add_err = opt.get('add_err', False)
        self.err_data_path = opt.get('err_data_path', None)
        self.split_decoded_data = opt.get('split_decoded_data', False)
        self.data_name = opt.get('data_name', None)

        opt['datafile'], data_dir = self._path(opt)
        self._setup_data(opt['datafile'], data_dir)
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
        agent.add_argument(
            '--repeat_minor_err',
            type='bool',
            default=False,
            help="True if one would like to repeat turns with one or two err (default: False).",
        )
        agent.add_argument(
            '--err_data_path',
            type=str,
            default=None,
            help="specify the data with generated errs (default: None).",
        )
        agent.add_argument(
            '--split_decoded_data',
            type='bool',
            default=False,
            help="True if one would like to create training data (default: False).",
        )
        agent.add_argument(
            '--data_name',
            type=str,
            default=None,
            help="specify the data file name (default: None).",
        )
        agent.add_argument(
            '--add_err',
            type='bool',
            default=False,
            help="True if one would like to create errors (default: False).",
        )


    def _path(self, opt):
        # # set up path to data (specific to each dataset)
        # data_dir = os.path.join(opt['datapath'], 'multiwozdst', 'MULTIWOZ2.1')
        # # data_dir = os.path.join('/checkpoint/kunqian/multiwoz/data/MultiWOZ_2.1/')
        # data_path = os.path.join(data_dir, 'data_reformat_trade_turn_sa_ha.json')
        data_dir = os.path.join(opt['datapath'], 'multiwozdst_cor')
        if self.data_name is not None:
            data_name = self.data_name
        else:
            data_name = 'dials_nodict_bs1.json'
        data_path = os.path.join(data_dir, data_name)
        # data_path = os.path.join(data_dir, 'dials_owndict_bs8.json')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if self.split_decoded_data:
            multiwozdst_dir = os.path.join(opt['datapath'], 'multiwozdst', 'MULTIWOZ2.1')

            if self.err_data_path is None:
                err_data_path = os.path.join("./experiment/gen_gpt2_nodict/", 'result_decode_all.jsonl')
            else:
                err_data_path = self.err_data_path

            split_decoded_data(data_path, multiwozdst_dir, err_data_path)
        return data_path, data_dir

    def _load_txt(self, file_path):
        with open(file_path) as df:
            data=df.read().lower().split("\n")
            data.remove('')
        return data

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _setup_data(self, data_path, data_dir):
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
            # # # repeat turns with one/two err
            if self.repeat_minor_err:
                adderr = AddErr(train_path, data_dir)
                self.messages = adderr.repeat_err()
            # # # manually add err following dist from err file
            elif self.add_err:
                adderr = AddErr(train_path, data_dir)
                self.messages = adderr.add_err()
            else:
                train_data = self._load_json(train_path)
                self.messages = list(train_data.values())
                
        if self.just_test:
            # if self.datatype.startswith('valid'):
            #     self.messages = self.messages[:200]
            # else:
            #     self.messages = self.messages[:20]
            self.messages = self.messages[:50]
    

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
        # import pdb
        # pdb.set_trace()
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


class DefaultTeacher(MultiWozDSTCORTeacher):
    """
    Default teacher.
    """
    pass
