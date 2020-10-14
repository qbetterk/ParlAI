#!/usr/bin/env python3
#
import os, sys, json
import math, argparse, random, re
from tqdm import tqdm
import pdb


class SplitDecodedData(object):
    def __init__(self, data_path, multiwozdst_dir, decoded_data_path):
        # # decoded data path
        self.decoded_data_path = decoded_data_path
        # # # val & test list
        self.multiwozdst_dir = multiwozdst_dir
        self.val_list_path = os.path.join(self.multiwozdst_dir, "valListFile.json")
        self.test_list_path = os.path.join(self.multiwozdst_dir, "testListFile.json")
        self.val_list = self._load_txt(self.val_list_path)
        self.test_list = self._load_txt(self.test_list_path)

        # # # # path to store reformat and splited data
        self.filter_type = ""
        # self.filter_type = "miss"
        # self.data_path = data_path.replace(".json", f"_fo{self.filter_type}.json")
        self.data_path = data_path
        self.train_data_path = self.data_path.replace(".json", "_train.json")
        self.valid_data_path = self.data_path.replace(".json", "_valid.json")
        self.test_data_path = self.data_path.replace(".json", "_test.json")

    def _load_list_of_json(self, file_path):
        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _load_txt(self, file_path):
        with open(file_path) as df:
            data=df.read().lower().split("\n")
            data.remove('')
        return data

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom slot_type slot_val,", ... ]
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
                    slots_list.append(domain+" "+slot_type+" "+slot_val+",")
        return slots_list

    def _filter_err(self, gt_slots, gen_slots, filter_type=""):
        """
        filter out missing or extra err
        """
        gt_list = self._extract_slot_from_string(gt_slots)
        gen_list = self._extract_slot_from_string(gen_slots)

        if filter_type == "":
            return gt_slots, gen_slots
        if filter_type == "extra":
            # filter out all extra slots
            new_gen_list = sorted(list(set(gt_list).intersection(set(gen_list))))
            # import pdb
            # pdb.set_trace()
            return gt_slots, " ".join(new_gen_list)
        if filter_type == "miss":
            # filter out all miss slots
            new_gt_list = sorted(list(set(gt_list).intersection(set(gen_list))))
            return " ".join(new_gt_list), gen_slots
        
    def split(self):
        decoded_all_dials = self._load_list_of_json(self.decoded_data_path)
        self.dials_all, self.dials_train, self.dials_val, self.dials_test = {}, {}, {}, {}
        for dial in decoded_all_dials:
            if "dial_id" not in dial["dialog"][0][0]:
                continue
            dial_id = dial["dialog"][0][0]['dial_id']
            turn_num = dial["dialog"][0][0]['turn_num']
            context = dial["dialog"][0][0]['text']
            gt_slots = dial["dialog"][0][0]['eval_labels'][0]
            gen_slots = dial["dialog"][0][1]['text']

            if dial_id not in self.test_list:
                gt_slots, gen_slots = self._filter_err(gt_slots, gen_slots, self.filter_type)

            reformat_dial = {
                "dial_id"   : dial_id,
                "turn_num"  : turn_num,
                "slots_inf" : gt_slots,
                "slots_err" : gen_slots,
                "context"   : context,
            }

            self.dials_all[dial_id + "-" + str(turn_num)] = reformat_dial
            if dial_id in self.test_list:
                self.dials_test[dial_id + "-" + str(turn_num)] = reformat_dial
            elif dial_id in self.val_list:
                self.dials_val[dial_id + "-" + str(turn_num)] = reformat_dial
            else:
                self.dials_train[dial_id + "-" + str(turn_num)] = reformat_dial

        
        with open(self.train_data_path, "w") as tf:
            json.dump(self.dials_train, tf, indent=2)
        with open(self.valid_data_path, "w") as tf:
            json.dump(self.dials_val, tf, indent=2)
        with open(self.test_data_path, "w") as tf:
            json.dump(self.dials_test, tf, indent=2)
        with open(self.data_path, "w") as tf:
            json.dump(self.dials_all, tf, indent=2)


def split_decoded_data(data_dir, multiwozdst_dir, decoded_data_path):
    split = SplitDecodedData(data_dir, multiwozdst_dir, decoded_data_path)
    split.split()

