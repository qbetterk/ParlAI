#!/usr/bin/env python3
#
import os, sys, json
import math
import pdb
from collections import defaultdict, OrderedDict
	
# from fuzzywuzzy import fuzz

domains = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]
slot_types = ["stay", "price", "addr", "type", "arrive", "day", "depart", "dest",
                "area", "leave", "stars", "department", "people", "time", "food",
                "post", "phone", "name",]
slot_types = ['food', 'day', 'dest', 'price', 'department', 'name', 'leave', 
              'type', 'stay', 'people', 'time', 'phone', 'depart', 'internet', 
              'post', 'stars', 'area', 'arrive', 'addr', 'parking']
#  # # # for trade data
slot_types = ["stay", "price", "addr",  "type", "arrive", "day", "depart", "dest",
            "area", "leave", "stars", "department", "people", "time", "food", 
            "post", "phone", "name", 'internet', 'parking',
            'book stay', 'book people','book time', 'book day',
            'pricerange', 'destination', 'leaveat', 'arriveby', 'departure']

class Clean_Analyze_result(object):
    def __init__(self, path_result=None):
        if path_result is None:
            path_result = "/data/home/kunqian/projs/ParlAI/experiment/dst_test/world_log_level123_multi_0.jsonl"
        self.result_path = path_result
        self.result_clean_path = ".".join(self.result_path.split(".")[:-1]) + "_clean.json"
        self.result_clean = None

        self.analyze_result_path = ".".join(self.result_path.split(".")[:-1]) + "_analyze.json"
        self.err_result_path = ".".join(self.result_path.split(".")[:-1]) + "_err.json"

    def load_raw_result(self):
        with open(self.result_path) as df:
            self.result = json.loads(df.read().lower())
        return self.result

    def _load_jsonl(self, data_path):
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        return data


    def clean(self, raw_result_path=None, save_clean_result=False, save_clean_path=None):
        """
        format raw_result (string of slots) into triplets
        input : [
            {
                "dial_id"   : dial_id,
                "turn_num"  : turn_num,
                "slots_inf" : "dom1 type1 val1, dom2 ...",
                "slots_err" : "dom1 type1 valx, dom2 ...",
                "context"   : "<user> ... <system> ... <user> ... <bs> slots_err </bs>",
            },..
            ]
        output: {"dial_id":{"turn_num":{"ground_truth": ["dom-slot_type-slot_val", ...],
                                       "generated_seq": ["dom-slot_type-slot_val", ...]}, ...}, ...}
        """
        if raw_result_path is None:
            raw_result_path = self.result_path
        if save_clean_path is None:
            save_clean_path = ".".join(raw_result_path.split(".")[:-1]) + "_clean.json"

        raw_result = self._load_jsonl(raw_result_path)
        self.result_clean = []
        count = defaultdict(int)
        total_turn = 0
        correct_turn = 0
        slots_acc = 0

        # pdb.set_trace()
        for dial in raw_result:

            total_turn += 1
            context = dial["dialog"][0][0]['text']
            domain  = dial["dialog"][0][0]['domain']
            gen_slots = dial["dialog"][0][1]['text']
            gt_slots = dial["dialog"][0][0]['eval_labels'][0]

            gt_list = gt_slots.split(",")
            gt_list = [f"{domain} name {item.strip()}" for item in gt_list]
            gen_list = [item.strip() for item in gen_slots.split(",")]
            if gen_list[-1] == "":
                gen_list = gen_list[:-1]

                
            self.result_clean.append({
                    "context" : context,
                    "domain " : domain,
                    "gt     " : " , ".join(gt_list),
                    "gen    " : " , ".join(gen_list)
                })

            if set(gen_list) == set(gt_list):
                correct_turn += 1


        joint_goal_acc = correct_turn / float(total_turn)
    
        
        print(f"joint goal acc: {joint_goal_acc}")


        with open(save_clean_path, "w") as tf:
            json.dump(self.result_clean[:100], tf, indent=2)



def main():
    clean = Clean_Analyze_result()
    clean.clean(raw_result_path="/data/home/kunqian/projs/ParlAI/experiment/dst_test/world_log_level12_0.jsonl")

if __name__ == "__main__":
    main()



