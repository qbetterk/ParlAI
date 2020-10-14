#!/usr/bin/env python3
#
import os, sys, json
import math, argparse, random, re
from tqdm import tqdm
import pdb

DOMAINS    = ["attraction", "hotel", "hospital", "restaurant", "taxi", "train"]
SLOT_TYPES = ["stay", "price", "addr",  "type", "arrive", "day", "depart", "dest",
            "area", "leave", "stars", "department", "people", "time", "food", 
            "post", "phone", "name", 'internet', 'parking',
            'book stay', 'book people','book time', 'book day',
            'pricerange', 'destination', 'leaveat', 'arriveby', 'departure']

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
    def __init__(self, data_dir, multiwoz_dir):
        self.data_dir = data_dir
        self.multiwoz_dir = multiwoz_dir
        self.data_path = os.path.join(self.multiwoz_dir, "data.json")
        self.reformat_data_name = "data_reformat_trade_turn.json"
        self.reformat_data_path = os.path.join(self.data_dir, self.reformat_data_name)

        self.val_path = os.path.join(self.multiwoz_dir, "valListFile.json")
        self.test_path = os.path.join(self.multiwoz_dir, "testListFile.json")
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

    def _load_ontology(self, ontology_path):
        """
        load ontology file from multiwoz
        input format: {
                dom - slot_type : [val1, val2, ...],
                ...
        }
        output format: {
            dom: {
                slot_type: [val1, val2, val3, ...]
            }
        }

        """
        with open(self.ontology_path) as ot:
            orig_ontology = json.loads(ot.read().lower())

        ontology = {}
        for dom_type, vals in orig_ontology.items():
            dom, slot_type = dom_type.split("-")
            # format slot type, e.g. "price range" --> "pricerange"
            if slot_type not in SLOT_TYPES:
                if slot_type.replace(" ", "") in SLOT_TYPES:
                    slot_type = slot_type.replace(" ", "")
                # else:
                #     pdb.set_trace()
            if dom not in ontology:
                ontology[dom] = {}
            ontology[dom][slot_type] = vals
        return ontology

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
            dial_id-turn_num-dom-slot_type:
                    {
                        "dial_id": dial_id
                        "turn_num": 0,
                        "slots_inf": slot_val,
                        "slots_err": "",
                        "slots_domtype" : "dom slot_type",
                        "context" : "User: ... Sys: ... User:..."
                    },
            ...
            }
        """
        self.data_trade_proc_path = os.path.join(self.multiwoz_dir, "dials_trade.json")
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
                
                # # # dialog history
                if turn["system_transcript"] != "":
                    context.append("<system> " + turn["system_transcript"])

                # # # adding current turn to dialog history
                context.append("<user> " + turn["transcript"])

                turn_form["context"] = " ".join(context)

                # # # slots/dialog states
                slots_inf = []
                # # # ACCUMULATED dialog states, extracted based on "belief_state"
                for state in turn["belief_state"]:
                    if state["act"] == "inform":
                        domain = state["slots"][0][0].split("-")[0]
                        slot_type = state["slots"][0][0].split("-")[1]
                        slot_val  = state["slots"][0][1]
                        
                        slots_inf += [domain, slot_type, slot_val, ","]

                slots_inf_alltype, slots_dom_alltype = self.find_dtv_alltype(slots_inf)

                for i in range(len(slots_inf_alltype) // 4):
                    domain  = slots_inf_alltype[4 * i]
                    slot_type = slots_inf_alltype[4 * i + 1]
                    slot_val = slots_inf_alltype[4 * i + 2]

                    dial_id = dial["dialogue_idx"]
                    turn_num = str(turn_form["turn_num"])

                    dial_key = f"{dial_id}-{turn_num}-{domain}-{slot_type}"

                    turn_form["slots_inf"] = slot_val
                    turn_form["slots_dom_type"] = f"{domain} {slot_type}"
                    # # # import error
                    turn_form["slots_err"] = ""

                    self.dials_form[dial_key] = turn_form.copy()
                    if dial["dialogue_idx"] in self.test_list:
                        self.dials_test[dial_key] = turn_form.copy()
                    elif dial["dialogue_idx"] in self.val_list:
                        self.dials_val[dial_key] = turn_form.copy()
                    else:
                        self.dials_train[dial_key] = turn_form.copy()

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

    def find_dtv_alltype(self, slots_inf):
        """
        given list of slots, return all possible slot type 
        related to domain
        input: slots_dom = [dom1, typ1, val1, ",", dom2, typ21, val21, ","]
        output: slots_inf_alltype = [dom1, typ1, val1, ",", dom1, typ2, "None", ",", ... , dom2, typ21, val21, ",", dom2, type22, "None", ",", ...]
                slots_dom_alltype = [dom1, typ1, ",", dom1, typ2, ",", ... ]
        """
        if slots_inf == []:
            return [], []

        slots_inf_alltype_dict = {}
        slots_inf_alltype, slots_dom_alltype = [], []
        # loading ontology file
        self.ontology_path = os.path.join(self.multiwoz_dir, "ontology.json")
        self.ontology = self._load_ontology(self.ontology_path)
        # assert list length:
        if len(slots_inf) % 4 != 0:
            print("incomplete slots seq")
            import pdb
            pdb.set_trace()
            return [], []

        for i in range(len(slots_inf) // 4):
            domain  = slots_inf[4 * i]
            slot_type = slots_inf[4 * i + 1]
            slot_val = slots_inf[4 * i + 2]
            if domain not in slots_inf_alltype_dict:
                # init dict for dom
                slots_inf_alltype_dict[domain] = {}
                for slot_type_ in self.ontology[domain]:
                    slots_inf_alltype_dict[domain][slot_type_] = "none"
            # add slot val
            slots_inf_alltype_dict[domain][slot_type] = slot_val
        # transfer dict into list
        for domain in slots_inf_alltype_dict:
            for slot_type, slot_val in slots_inf_alltype_dict[domain].items():
                slots_inf_alltype += [domain, slot_type, slot_val, ","]
                slots_dom_alltype += [domain, slot_type, ","]
        # import pdb
        # pdb.set_trace()
        return slots_inf_alltype, slots_dom_alltype

def reformat_parlai(data_dir, multiwoz_dir, force_reformat=False):
    # args = Parse_args()
    # args.data_dir = data_dir
    if os.path.exists(os.path.join(data_dir, 'data_reformat_trade_turn.json')) and \
       os.path.exists(os.path.join(data_dir, 'data_reformat_trade_turn_train.json')) and \
       os.path.exists(os.path.join(data_dir, 'data_reformat_trade_turn_valid.json')) and \
       os.path.exists(os.path.join(data_dir, 'data_reformat_trade_turn_test.json')) and \
       not force_reformat:
        pass
        # print("already reformat data before, skipping this time ...")
    else:
        reformat = Reformat_Multiwoz(data_dir, multiwoz_dir)
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

        
        
