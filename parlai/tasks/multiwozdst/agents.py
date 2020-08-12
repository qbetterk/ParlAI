#!/usr/bin/env python3
#
import sys, os
import json

from parlai.core.teachers import FixedDialogTeacher
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

    # process the data with TRADE's code
    trade_process(data_dir)

    # reformat data for DST
    reformat_parlai(data_dir)

    return data_path, data_dir


class MultiWozDSTTeacher(FixedDialogTeacher):
    """
    MultiWOZ DST Teacher.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt['datafile'], jsons_path = _path(opt)
        self._setup_data(opt['datafile'], jsons_path)
        self.id = 'multiwozdst'
        self.reset()

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
        print('loading: ' + data_path)

        # # # loading directly from test file or val file
        if self.datatype.startswith('test'):
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

    # def customize_evaluation(self)

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
        }
        return action


class DefaultTeacher(MultiWozDSTTeacher):
    pass
