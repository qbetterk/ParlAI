#!/usr/bin/env python3
#
import os, sys, json
import math, argparse, random, re
from tqdm import tqdm
import pdb


"""
my:{'arrive', 'parking', 'name', 'phone', 'stay', 'time', 'type', 'depart', 'stars', 'post', 'dest', 'leave', 'addr', 'people', 'day', 'internet', 'price', 'food', 'area', 'department'}
trade: ['area', 'arriveby', 'book day', 'book people', 'book stay', 'book time', 
        'day', 'department', 'departure', 'destination', 'food', 'internet', 
        'leaveat', 'name', 'parking', 'pricerange', 'stars', 'type'] 18


my - trade:{'stay', 'price', 'dest', 'leave', 'people', 'arrive', 'depart',          'post', 'phone', 'time', 'addr'}
trade - my:{'book stay', 'pricerange', 'destination', 'leaveat', 'book people', 'arriveby', 'departure'}

['attraction-area', 'attraction-name', 'attraction-type', 
 'hospital-department', 
 'hotel-area', 'hotel-book day', 'hotel-book people', 'hotel-book stay', 'hotel-internet', 
 'hotel-name', 'hotel-parking', 'hotel-pricerange', 'hotel-stars', 'hotel-type', 
 'restaurant-area', 'restaurant-book day', 'restaurant-book people', 'restaurant-book time', 
 'restaurant-food', 'restaurant-name', 'restaurant-pricerange', 
 'taxi-arriveby', 'taxi-departure', 'taxi-destination', 'taxi-leaveat', 
 'train-arriveby', 'train-book people', 'train-day', 'train-departure', 
 'train-destination', 'train-leaveat']

 total 31 type 6 domain
"""
SLOT_TYPE_MAPPING = {'pricerange' :'price', 
                     'destination':'dest', 
                     'leaveat'    :'leave', 
                     'arriveby'   :'arrive', 
                     'departure'  :"depart",
                     'book stay'  :'stay', 
                     'book people':'people', 
                     "book time"  :"time",
                     "book day"   :"day"
                     }

class Reformat_Multiwoz(object):
    """
    reformat multiwoz (maybe sgd later) into 
    utt-to-slots
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_path = os.path.join(self.data_dir, "data.json")
        self.reformat_data_name = "data_reformat_trade_turn.json"
        # if args.order != "dtv":
        #     self.reformat_data_name = self.reformat_data_name.replace(".json", "_O"+args.order+".json")
        self.reformat_data_path = os.path.join(self.data_dir, self.reformat_data_name)
        self.order = "dtv"
        # self.slot_accm = args.slot_accm
        # self.hist_accm = args.hist_accm
        self.slot_accm = True
        self.hist_accm = True

        self.val_path = os.path.join(self.data_dir, "valListFile.json")
        self.test_path = os.path.join(self.data_dir, "testListFile.json")
        self.val_list = self.load_txt(self.val_path)
        self.test_list = self.load_txt(self.test_path)

        # # # normally there would not be the case of 
        # # # slot_accm == 1 while hist_accm == 0
        if self.slot_accm:
            self.reformat_data_path = self.reformat_data_path.replace(".json", "_sa.json")
        if self.hist_accm:
            self.reformat_data_path = self.reformat_data_path.replace(".json", "_ha.json")

        self.slot_type_set = set()
        # self.load_dials()

    def load_dials(self, data_path=None):
        if data_path is None:
            data_path = self.data_path
        with open(data_path) as df:
            self.dials = json.loads(df.read().lower())
    
    def load_txt(self, file_path):
        with open(file_path) as df:
            data=df.read().lower().split("\n")
            data.remove('')
        return data

    def remove_triplets(self, b_list, mid_idx):
        """
        triplet = domain slot_type slot_value
        mid_idx = idx of slot_type
        """
        if mid_idx == 1:
            return b_list[3:]
        elif mid_idx == len(b_list) - 2:
            return b_list[:-3]
        else:
            return b_list[:mid_idx-1] + b_list[mid_idx+2:]

    def reformat_from_trade_proc_to_turn(self):
        """
        following trade's code for normalizing multiwoz*
        now the data has format:
        file=[{
            "dialogue_idx": dial_id,
            "domains": [dom],
            "dialogue": [
                    {
                        "turn_idx": 0,
                        "domain": "hotel",
                        "system_transcript": "system response",
                        "transcript": "user utterance",
                        "system_act": [],
                        "belief_state": [{
                            "slots":[["domain-slot_type","slot_vale"]],
                            "act":  "inform"
                        }, ...], # accumulated
                        "turn_label": [["domain-slot_type","slot_vale"],...],    # for current turn
                    },
                    ...
                ],
                ...
            },
            ]
        and output with format like:
        file={
            dial_id-turn_num:
                    {
                        "dial_id": dial_id
                        "turn_num": 0,
                        "slots_inf": slot sequence ("dom slot_type1, dom slot_type2 ..."),
                        "slots_err": slot sequence ("dom slot_type1, ..."),
                        "context" : "User: ... Sys: ... User:..."
                    },
            ...
            }
        """
        self.data_trade_proc_path = os.path.join(self.data_dir, "dials_trade.json")
        self.load_dials(data_path = self.data_trade_proc_path)
        self.dials_form = {}
        self.dials_train, self.dials_val, self.dials_test = {}, {}, {}

        for dial in tqdm(self.dials):
            # self.dials_form[dial["dialogue_idx"]] = []
            context = []
            for turn in dial["dialogue"]:
                turn_form = {}
                # # # turn number
                turn_form["turn_num"] = turn["turn_idx"]

                # # # # dial_id
                turn_form["dial_id"] = dial["dialogue_idx"]
                
                # # # slots/dialog states
                slots_inf = []
                if not self.slot_accm:
                    # # # dialog states only for the current turn, extracted based on "turn_label"
                    for slot in turn["turn_label"]:
                        domain    = slot[0].split("-")[0]
                        slot_type = slot[0].split("-")[1]
                        slot_val  = slot[1]

                        slots_inf += [domain]

                else:
                    # # # ACCUMULATED dialog states, extracted based on "belief_state"
                    for state in turn["belief_state"]:
                        if state["act"] == "inform":
                            domain = state["slots"][0][0].split("-")[0]
                            slot_type = state["slots"][0][0].split("-")[1]
                            slot_val  = state["slots"][0][1]

                            slots_inf += [domain, ","]

                slots_inf = list(set(slots_inf))
                turn_form["slots_inf"] = " ".join(slots_inf)

                # # # import error
                turn_form["slots_err"] = ""

                # # # dialog history
                if turn["system_transcript"] != "":
                    context.append("<system> " + turn["system_transcript"])
                
                if not self.hist_accm:
                    context = context[-1:]

                # # # adding current turn to dialog history
                context.append("<user> " + turn["transcript"])

                turn_form["context"] = " ".join(context)

                self.dials_form[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form
                if dial["dialogue_idx"] in self.test_list:
                    self.dials_test[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form
                elif dial["dialogue_idx"] in self.val_list:
                    self.dials_val[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form
                else:
                    self.dials_train[dial["dialogue_idx"] + "-" + str(turn_form["turn_num"])] = turn_form

        self.reformat_train_data_path = self.reformat_data_path.replace(".json", "_train.json")
        self.reformat_valid_data_path = self.reformat_data_path.replace(".json", "_valid.json")
        self.reformat_test_data_path = self.reformat_data_path.replace(".json", "_test.json")

        with open(self.reformat_train_data_path, "w") as tf:
            json.dump(self.dials_train, tf, indent=2)
        with open(self.reformat_valid_data_path, "w") as tf:
            json.dump(self.dials_val, tf, indent=2)
        with open(self.reformat_test_data_path, "w") as tf:
            json.dump(self.dials_test, tf, indent=2)
        with open(self.reformat_data_path, "w") as tf:
            json.dump(self.dials_form, tf, indent=2)

    def save_dials(self):
        with open(self.reformat_data_path, "w") as tf:
            json.dump(self.dials_form, tf, indent=2)
        print(f"Saved reformatted data to {self.reformat_data_path} ...")


def reformat_parlai(data_dir, force_reformat=False):
    # args = Parse_args()
    # args.data_dir = data_dir
    if os.path.exists(os.path.join(data_dir, 'data_reformat_trade_turn_sa_ha.json')) and \
       os.path.exists(os.path.join(data_dir, 'data_reformat_trade_turn_sa_ha_train.json')) and \
       os.path.exists(os.path.join(data_dir, 'data_reformat_trade_turn_sa_ha_valid.json')) and \
       os.path.exists(os.path.join(data_dir, 'data_reformat_trade_turn_sa_ha_test.json')) and \
       not force_reformat:
        pass
        # print("already reformat data before, skipping this time ...")
    else:
        reformat = Reformat_Multiwoz(data_dir)
        reformat.reformat_from_trade_proc_to_turn()


def Parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",   default="multiwoz")
    parser.add_argument(      "--slot_accm", default=1, type=int)
    parser.add_argument(      "--hist_accm", default=1, type=int)
    parser.add_argument(      "--trade", default=1, type=int)
    parser.add_argument(      "--order", default="dtv", type=str,
                                         help="slot order, default as : domain, slot_type, slot_value")
    parser.add_argument(      "--mask_predict", default=0, type=int)
    parser.add_argument(      "--reformat_data_name", default=None)
    parser.add_argument(      "--save_dial", default=True, type=bool)
    parser.add_argument(      "--data_dir", default="/checkpoint/kunqian/multiwoz/data/MultiWOZ_2.1/")

    args = parser.parse_args()
    return args

def main():
    args = Parse_args()

    if args.dataset == "multiwoz":
        reformat = Reformat_Multiwoz(args)
        if args.reformat_data_name is not None:
            reformat.reformat_data_path = os.path.join(reformat.data_dir, args.reformat_data_name)
        if args.trade:
            if args.mask_predict:
                reformat.reformat_trade_to_mask_err()
            else:
                # reformat.reformat_from_trade_proc()
                reformat.reformat_from_trade_proc_to_turn()
                # print(sorted(list(reformat.slot_type_set)))
        else:
            reformat.reformat_slots()
    elif args.dataset == "sgd":
        reformat = Reformat_SGD(args)
        reformat.reformat_slots()
    

if __name__ == "__main__":
    main()

        
        
