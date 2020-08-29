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

class AddErr(object):
    def __init__(self, data_path, data_dir):
        self.data_path = data_path
        self.data_dir  = data_dir
        self.err_analyze_filepath = "./experiment/gen_gpt2_nodict/result_decode_all_bs8_analyze.json"

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
                else:
                    pdb.set_trace()
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
        # load ontology file
        self.ontology_path = os.path.join(self.data_dir, "../multiwozdst/MULTIWOZ2.1/ontology.json")
        self.ontology = self._load_ontology(self.ontology_path)
        # load err distribution
        self.err_dist = self._load_json(self.err_analyze_filepath)
        # load dialogs
        self.dials = list(self._load_json(self.data_path).values())
        # setting uniform distribution
        self.uniform = True

        self.target_dials = []
        for dial in self.dials:
            slots_inf = self._extract_slot_from_string(dial["slots_inf"])
            
            # use generated err or manually created err
            dist = [1, 3] # [wi err, wo err]
            if not self.uniform:
                dist = [self.err_dist["count_err_num"]["total_turn"], self.err_dist["count_err_num"]["total_turn_num_w_err"]]
            if random.choices([0,1], weights=dist, k=1)[0]:
                # with err
                slots_err = self._creat_err(slots_inf)
                dial["slots_err"] = slots_err
            else:
                # without err
                dial["slots_err"] = dial["slots_inf"]

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
        slots_list = self._create_miss_err(slots_list)
        # # # extra err
        slots_list = self._create_extra_err(slots_list)
        # # # match err (replace dom/slot_type/slot_val)
        slots_list = self._create_matched_err(slots_list)

        # # # format list into string
        slots_str = ", ".join([" ".join(slot) for slot in slots_list]) if slots_list!=[] else ""
        return slots_str
        
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
        # # # set match err distribution dict
        self.match_dict = self.err_dist["count_err_type"]["match"]
        # # # decide num of matched err
        err_num_range = min(2, len(slots_list) + 1)   # 1 ~ n + 1
        if self.uniform:
            dist = None
        else:
            dist = []
            for num in range(err_num_range):
                dist.append(self.match_dict["slot_num"].get(str(len(slots_list)), {}).get(str(num), 0))
        err_num = random.choices(range(err_num_range), weights=dist, k=1)[0]
        # # # choose err slot index
        err_idxs = random.choices(range(len(slots_list)), weights=None, k=err_num)
        # # # replace slot one by one
        for err_idx in err_idxs:
            slots_list[err_idx] = self._replace_slot(slots_list[err_idx])
        
        return slots_list

    def _replace_slot(self, slot):
        """
        replacing domain name or slot type or slot value
        for a single slot

        input  : slot = [dom1 type1 val1]
        output : slot = [dom1 type1 valx]
        """
        [dom, slot_type, slot_val] = slot
        # choose dom or type or val to replace
        # TODO: change prior probability to conditional prob on current
        #  dom slot_type p(.| dom, slot_type)
        dist = [self.err_dist["count_err_type"]["overall"]["domain"],
                self.err_dist["count_err_type"]["overall"]["slot_type"],
                self.err_dist["count_err_type"]["overall"]["slot_val"]] 
        part = random.choices(["domain", "slot_type", "slot_val"], weights=dist, k=1)[0]

        if part == "domain":
            # TODO: change prior probability (freq + add-one) to conditional prob on current
            #  dom slot_type p(.| dom, slot_type)
            if self.uniform:
                dist = None
            else:
                dist = []
                for dom_ in DOMAINS:
                    dual_dom = "--".join(sorted([dom, dom_]))
                    dist.append(self.match_dict.get(dual_dom, {}).get("total", 0) + 1)
            dom_err = random.choices(DOMAINS, weights=dist, k=1)[0]
            return [dom_err, slot_type, slot_val]

        if part == "slot_type":
            # do not change for domain police and hospital
            if (dom not in self.ontology or
                dom not in self.match_dict["slot_type"]):
                return [dom, slot_type, slot_val]

            type_candidates = list(self.ontology[dom].keys())
            # TODO: change prior probability (freq + add-0.1) to conditional prob on current
            #  dom slot_type p(.| dom, slot_type)
            if self.uniform:
                dist = None
            else:
                dist = []
                for slot_type_ in type_candidates:
                    dual_type = "--".join(sorted([slot_type, slot_type_]))
                    dist.append(self.match_dict.get(dom, {}).get(dual_type, 0) + 0.5)
            type_err = random.choices(type_candidates, weights=dist, k=1)[0]
            return [dom, type_err, slot_val]
        
        if part == "slot_val":
            # pdb.set_trace()
            val_candidates = self.ontology.get(dom, {}).get(slot_type, [])
            if not val_candidates:
                pdb.set_trace()
            # TODO: change prior probability (uniform) to conditional prob on current
            #  dom slot_type p(.| dom, slot_type)
            # compute probability
            dist = None
            # repalce
            val_err = random.choices(val_candidates, weights=dist, k=1)[0]
            return [dom, slot_type, val_err]

    def _create_extra_err(self, slots_list):
        """
        create error adding a triplet
        first decide extra err num and then 
        dom --> slot_type --> slot_val
        input  : slots_list = [trip1, trip2]
        output : slots_list = [trip1, trip3, trip2]
        """
        self.extra_dict = self.err_dist["count_err_type"]["extra"]
        # # # decide num of matched err
        err_num_range = min(2, len(slots_list) + 1)
        if self.uniform:
            dist = None
        else:
            dist = []
            for num in range(err_num_range):
                dist.append(self.extra_dict["slot_num"].get(str(len(slots_list)), {}).get(str(num), 0))
        err_num = random.choices(range(err_num_range), weights=dist, k=1)[0]
        # # # choose err slot place index
        # dist = [1 for i in range(len(slots_list))] if err_num != 0 else None
        dist = None
        err_idxs = random.choices(range(len(slots_list)), weights=dist, k=err_num)
        for err_idx in sorted(err_idxs, reverse=True):
            slots_list.insert(err_idx, self._generate_slot())

        return slots_list

    def _generate_slot(self):
        """
        generate a single slot for extra slot err
        input  : 
        output : [dom type val]
        """
        # # # decide domain
        if self.uniform:
            dist = None
        else:
            dist = []
            for dom in DOMAINS:
                dist.append(self.extra_dict.get(dom, {}).get("total", 0))
        dom_err = random.choices(DOMAINS, weights=dist, k=1)[0]

        # # # slot_type
        slot_type_candidates = list(self.ontology[dom_err].keys())
        if self.uniform:
            dist = None
        else:
            dist = []
            for slot_type in slot_type_candidates:
                dist.append(self.extra_dict.get(dom_err, {}).get(slot_type, 0))
        slot_type_err = random.choices(slot_type_candidates, weights=dist, k=1)[0]

        # # # slot_val
        slot_val_err = random.choices(self.ontology[dom_err][slot_type_err], k=1)[0]

        return [dom_err, slot_type_err, slot_val_err]
        
    def _create_miss_err(self, slots_list):
        """
        create error by removing a triplet
        input  : slots_list = [trip1, trip2, trip3]
        output : slots_list = [trip1, trip2]
        """
        # # # case of no slot
        if len(slots_list) == 0:
            return slots_list

        self.miss_dict = self.err_dist["count_err_type"]["miss"]
        # # # decide num of miss err
        err_num_range = min(2, len(slots_list))
        if self.uniform:
            dist = None
        else:
            dist = []
            for num in range(err_num_range):
                dist.append(self.miss_dict["slot_num"].get(str(len(slots_list)), {}).get(str(num), 0))
        err_num = random.choices(range(err_num_range), weights=dist, k=1)[0]
        # # # decide which to remove
        if self.uniform:
            dist = None
        else:
            dist = []
            for slot in slots_list:
                dist.append(self.miss_dict.get(slot[0], {}).get(slot[1], 0) + 1)
        err_idxs = random.choices(range(len(slots_list)), weights=dist, k=err_num)
        # # # removing from end to start in list
        for err_idx in sorted(err_idxs, reverse=True):
            if err_idx >= len(slots_list):
                pdb.set_trace()
            del slots_list[err_idx]
        return slots_list

        