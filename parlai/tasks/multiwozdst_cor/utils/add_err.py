#!/usr/bin/env python3
#
import os, sys, json
import math, argparse, random, re
import pdb

DOMAINS    = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]
# slot_types = ['book time', 'leaveat', 'name', 'internet', 'book stay', 
#               'pricerange', 'arriveby', 'area', 'destination', 'day', 
#               'food', 'departure', 'book day', 'book people', 'department', 
#               'stars', 'parking', 'type']
SLOT_TYPES = ["stay", "price", "addr",  "type", "arrive", "day", "depart", "dest",
            "area", "leave", "stars", "department", "people", "time", "food", 
            "post", "phone", "name", 'internet', 'parking',
            'book stay', 'book people','book time', 'book day',
            'pricerange', 'destination', 'leaveat', 'arriveby', 'departure']

class AddErr(object):
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
            if dom not in ontology:
                ontology[dom] = {}
            ontology[dom][slot_type] = vals
        return ontology

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
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
            if len(slot) > 2 and slot[0] in DOMAINS:
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

    def repeat_err(self, select_times=5, select_port=0.8):
        """
        repeat turns with only one or two err
        basicly put all those turns into the list pool self.target_dials
        randomly choose k% from this pool for n times (k,n can be parameters)
        and append to self.dials then return self.dials
        self.dials = [{
                "dial_id"   : dial_id,
                "turn_num"  : turn_num,
                "slots_inf" : "dom1 type1 val1, dom2 ...",
                "slots_err" : "dom1 type1 valx, dom2 ...",
                "context"   : "<user> ... <system> ... <user> ... <bs> slots_err </bs>",
            },..
            ]
        """
        self.dials = list(self._load_json(self.data_path).values())

        self.target_dials = []
        for dial in self.dials:
            slots_inf = self._extract_slot_from_string(dial["slots_inf"])
            slots_err = self._extract_slot_from_string(dial["slots_err"])

            miss_slots = list(set(slots_inf) - set(slots_err))
            extr_slots = list(set(slots_err) - set(slots_inf))

            err_num = len(miss_slots) + len(extr_slots)

            if 0 < err_num < 3:
                self.target_dials.append(dial)

        select_num   = int(select_port * len(self.target_dials))
        # pdb.set_trace()
        for i in range(select_times):
            self.dials += random.choices(self.target_dials, k=select_num)
        random.shuffle(self.dials)
        return self.dials

    def add_err(self):
        """
        add errors according to the err distribution
        of result from generation model. both input and
        output should be in the format like:
        self.dials = [{
                "dial_id"   : dial_id,
                "turn_num"  : turn_num,
                "slots_inf" : "dom1 type1 val1, dom2 ...",
                "slots_err" : "dom1 type1 valx, dom2 ...",
                "context"   : "<user> ... <system> ... <user> ... <bs> slots_err </bs>",
            },..
            ]
        """
        self.ontology_path = os.path.join(self.data_dir, "../multiwozdst/MULTIWOZ2.1/ontology.json")
        self.ontology = self._load_ontology()

        self.dials = list(self._load_json(self.data_path).values())

        self.target_dials = []
        for dial in self.dials:
            slots_inf = self._extract_slot_from_string(dial["slots_inf"])
            
            # use generated err or manually created err
            if random.choices([0,1], weights=[1,1], k=1)[0]:
                slots_err = self._add_err(slots_inf)
                dial["slots_err"] = slots_err

            self.target_dials.append(dial)
        return self.target_dials

    def _creat_err(self, slots_inf):
        """
        create error by adding, replacing, removing
        for training correction model
        input : slots_inf = [dom1--slot_type1--slot_val1, dom2--slot_type2--slot_val2, ...]
        output: "dom1 type1 valx, dom2 ..."
        """
        slots_list = [slot.split("--") for slot in slots_inf]
        # # # missing err
        # miss_num = random.choices(range(len(slots_list)), weights=None, k=1)

        # # # extra err
        
        # # # match err (replace dom/slot_type/slot_val)
        
    def _create_matched_err(self, slots_list):
        """
        create error by replacing domain name or slot type
        or slot value
        input  : slots_list = [[dom1 type1 val1], ...]
        output : slots_list = [[dom1 type1 valx], ...]

        param: err num per turn
               err ratio over types(add/remove/replace) 
               err ratio over domains
        """

        # # # case of no slot
        if len(slots_list) == 0:
            return slots_list

        err_num = 1
        err_idxs = random.choices(range(slots_list), weights=None, k=err_num)

        for err_idx in err_idxs:
            [domain, slot_type, slot_val] = slots_inf[err_idx]

            for key_ in self.ontology:
                if key_.startswith(domain) and slot_type in key_.split("-")[-1]:
                    # # # skip if slot_type contains only one slot_val
                    if len(self.ontology[key_]) > 1:
                        vals = self.ontology[key_][:]
                        if slot_val in vals:
                            vals.remove(slot_val)
                        slots_inf[err_idx * 4 + self.order.index("v")] = random.choice(vals)
                    break
        
        return " ".join(slots_inf)