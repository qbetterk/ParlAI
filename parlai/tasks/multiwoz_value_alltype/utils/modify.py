#!/usr/bin/env python3
#
import os, sys, json
import math, argparse, random, re
import pdb

DOMAINS    = ["attraction", "hotel", "hospital", "restaurant", "taxi", "train"]
# slot_types = ['book time', 'leaveat', 'name', 'internet', 'book stay', 
#               'pricerange', 'arriveby', 'area', 'destination', 'day', 
#               'food', 'departure', 'book day', 'book people', 'department', 
#               'stars', 'parking', 'type']
SLOT_TYPES = ["stay", "price", "addr",  "type", "arrive", "day", "depart", "dest",
            "area", "leave", "stars", "department", "people", "time", "food", 
            "post", "phone", "name", 'internet', 'parking',
            'book stay', 'book people','book time', 'book day',
            'pricerange', 'destination', 'leaveat', 'arriveby', 'departure']

class Modify(object):
    def __init__(self, data_path, data_dir):
        self.data_path = data_path
        self.data_dir  = data_dir

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
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

    def _load_jsonl(self, data_path):
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def replace_domtype(self, dials_list, gen_type_result_path):
        """
        this function is used when doing iterative correction,
        by replacing error slots in the test data with the 
        generated result, specified by 'gen_type_result_path',
        usually from the last iteration.
        input dials list should be in format:
        gen_result = [{
            "dialog" : [[{
                "id" : "multiwozdst_cor",
                "text" : "dialog history <bs> gen dst </bs>",
                "eval_labels" : "ground truth dst",
                "dial_id" : dial_id,
                "turn_num": turn_num,
            },{
                "text" : generated result,
                ... : ...
            }]]
            "context": ...
        }]
        output dials list should be in format:
        dials_list = [{
                "dial_id"   : dial_id,
                "turn_num"  : turn_num,
                "slots_inf" : "dom1 type1 val1, dom2 ...",
                "slots_domtype" : "dom1 type1, dom2 ...",   <----- to be replaced
                "context"   : "<user> ... <system> ... <user> ... <bs> slots_err </bs>",
            },..
            ]
        """
        # load generated test result (list)
        gen_result = self._load_jsonl(gen_type_result_path)
        # transfer generated test result in to dict, for easier searching
        gen_result_dict = {}
        for dial in gen_result:
            # pdb.set_trace()
            dial_id = dial["dialog"][0][0]['dial_id']
            turn_num = dial["dialog"][0][0]['turn_num']
            gen_result_dict[f"{dial_id}-{turn_num}"] = dial["dialog"][0][1]['text']
        # replace err slots with generated ones
        for dial in dials_list:
            dial_id = dial["dial_id"]
            turn_num = dial["turn_num"]
            dial["slots_domtype"] = gen_result_dict[f"{dial_id}-{turn_num}"]
        return dials_list
