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
    def __init__(self, data_dir, reformat_data_name="data_reformat_trade_turn.json"):
        self.data_dir = data_dir
        self.data_path = os.path.join(self.data_dir, "data.json")
        self.reformat_data_name = reformat_data_name
        # if args.order != "dtv":
        #     self.reformat_data_name = self.reformat_data_name.replace(".json", "_O"+args.order+".json")
        self.reformat_data_path = os.path.join(self.data_dir, self.reformat_data_name)
        self.order = "dtv"
        
        self.val_path = os.path.join(self.data_dir, "valListFile.json")
        self.test_path = os.path.join(self.data_dir, "testListFile.json")
        self.val_list = self.load_txt(self.val_path)
        self.test_list = self.load_txt(self.test_path)

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
                        "slots_inf": slot sequence ("dom slot_type1 slot_val1, dom slot_type2 ..."),
                        "slots_err": slot sequence ("dom slot_type1 slot_type2, ..."),
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
                
                # # # ACCUMULATED dialog states, extracted based on "belief_state"
                for state in turn["belief_state"]:
                    if state["act"] == "inform":
                        domain = state["slots"][0][0].split("-")[0]
                        slot_type = state["slots"][0][0].split("-")[1]
                        slot_val  = state["slots"][0][1]
                        
                        slots_inf += [domain, slot_type, slot_val, ","]

                turn_form["slots_inf"] = self.translate_dst(slots_inf)

                # # # import error
                turn_form["slots_err"] = ""

                # # # dialog history
                if turn["system_transcript"] != "":
                    context.append("<system> " + turn["system_transcript"])

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


    def translate_dst(self, slots_inf):
        """
        all_slot_type = [
            'attraction-area', 'attraction-name', 'attraction-type', 

            'hospital-department', 

            'hotel-area', 'hotel-book day', 'hotel-book people', 'hotel-book stay', 'hotel-internet', 
            'hotel-name', 'hotel-parking', 'hotel-pricerange', 'hotel-stars', 'hotel-type', 

            'restaurant-area', 'restaurant-book day', 'restaurant-book people', 'restaurant-book time', 
            'restaurant-food', 'restaurant-name', 'restaurant-pricerange', 

            'taxi-arriveby', 'taxi-departure', 'taxi-destination', 'taxi-leaveat', 

            'train-arriveby', 'train-book people', 'train-day', 'train-departure', 
            'train-destination', 'train-leaveat'
            ]
        """
        templates = {
            "attraction-area" : "attraction area is [slot_val]",
            "attraction-name" : "attraction name is [slot_val]",
            "attraction-type" : "attraction type is [slot_val]",
            "hospital-department" : "hospital department is [slot_val]",
            "hotel-area" : "hotel area is [slot_val]",
            "hotel-name" : "hotel name is [slot_val]",
            "hotel-pricerange" : "hotel price is [slot_val]",
            "hotel-stars" : "hotel stars is [slot_value]",
            "hotel-type" :  "hotel type is [slot_value]",
            "hotel-internet" :  "hotel internet is [slot_value]",
            "hotel-parking" :  "hotel parking is [slot_value]",
            "hotel-book day" :  "hotel book day is [slot_value]",
            "hotel-book people" :  "hotel book people is [slot_value]",
            "hotel-book stay" :  "hotel book stay is [slot_value]",
            "restaurant-area" : "restaurant area is [slot_val]",
            "restaurant-name" : "restaurant name is [slot_val]",
            "restaurant-food" : "restaurant food is [slot_val]",
            "restaurant-pricerange" : "restaurant price is [slot_val]",
            "restaurant-book day" :  "restaurant book day is [slot_value]",
            "restaurant-book people" :  "restaurant book people is [slot_value]",
            "restaurant-book time" :  "restaurant book time is [slot_value]",
            "taxi-arriveby" :  "taxi arriveby is [slot_value]",
            "taxi-leaveat" :  "taxi leaveat is [slot_value]",
            "taxi-departure" :  "taxi departure is [slot_value]",
            "taxi-destination" :  "taxi destination is [slot_value]",
            "train-arriveby" :  "train arriveby is [slot_value]",
            "train-leaveat" :  "train leaveat is [slot_value]",
            "train-departure" :  "train departure is [slot_value]",
            "train-destination" :  "train destination is [slot_value]",
            "train-day" :  "train day is [slot_value]",
            "train-book people" :  "train book people is [slot_value]",
        }
        templates = {
            "attraction-area" : "attraction area is [slot_val]",
            "attraction-name" : "attraction name is [slot_val]",
            "attraction-type" : "attraction type is [slot_val]",
            "hospital-department" : "hospital department is [slot_val]",
            "hotel-area" : "hotel area is [slot_val]",
            "hotel-name" : "hotel name is [slot_val]",
            "hotel-pricerange" : "hotel pricerange is [slot_val]",
            "hotel-stars" : "hotel stars is [slot_value]",
            "hotel-type" :  "hotel type is [slot_value]",
            "hotel-internet" :  "hotel internet is [slot_value]",
            "hotel-parking" :  "hotel parking is [slot_value]",
            "hotel-book day" :  "book hotel on [slot_value]",
            "hotel-book people" :  "book hotel for [slot_value] people",
            "hotel-book stay" :  "book hotel for [slot_value] stay",
            "restaurant-area" : "restaurant area is [slot_val]",
            "restaurant-name" : "restaurant name is [slot_val]",
            "restaurant-food" : "restaurant food is [slot_val]",
            "restaurant-pricerange" : "restaurant pricerange is [slot_val]",
            "restaurant-book day" :  "book restaurant on [slot_value]",
            "restaurant-book people" :  "book restaurant for [slot_value] people",
            "restaurant-book time" :  "book restaurant for [slot_value] time",
            "taxi-arriveby" :  "taxi arrive by [slot_value]",
            "taxi-leaveat" :  "taxi leave at [slot_value]",
            "taxi-departure" :  "taxi departure is [slot_value]",
            "taxi-destination" :  "taxi destination is [slot_value]",
            "train-arriveby" :  "train arrive by [slot_value]",
            "train-leaveat" :  "train leave at [slot_value]",
            "train-departure" :  "train departure is [slot_value]",
            "train-destination" :  "train destination is [slot_value]",
            "train-day" :  "train is on [slot_value]",
            "train-book people" :  "book train for [slot_value] people",
        }
        
        slots_trans = []
        
        if len(slots_inf) % 4 != 0:
            print("incomplete slots seq")
            import pdb
            pdb.set_trace()
            return ""

        for i in range(len(slots_inf) // 4):
            domain  = slots_inf[4 * i]
            slot_type = slots_inf[4 * i + 1]
            slot_val = slots_inf[4 * i + 2]

            slots_trans += [templates[f"{domain}-{slot_type}"].replace("[slot_value]", slot_val),","]
        return " ".join(slots_trans)

def reformat_parlai(data_dir, force_reformat=False):
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

        
        
