#!/usr/bin/env python3
#
import os, sys, json
import math, argparse, random, re
import pdb

DOMAINS    = ["attraction", "hotel", "hospital", "restaurant", "taxi", "train"]

SLOT_TYPES = ["stay", "price", "addr",  "type", "arrive", "day", "depart", "dest",
            "area", "leave", "stars", "department", "people", "time", "food", 
            "post", "phone", "name", 'internet', 'parking',
            'book stay', 'book people','book time', 'book day',
            'pricerange', 'destination', 'leaveat', 'arriveby', 'departure']
def weighted_sample(cand_list, weights=None, k=1):
    return list(set(random.choices(cand_list, weights=weights, k=k)))
    # return random.choices(cand_list, weights=weights, k=k)

class AddErr(object):
    def __init__(self, data_path, data_dir):
        self.data_path = data_path
        self.data_dir  = data_dir
        # load err distribution
        self.err_analyze_filepath = "/checkpoint/kunqian/parlai/gen_gpt2_nodict/result_decode_all_bs8_analyze.json"
        self.err_dist = self._load_json(self.err_analyze_filepath)
        # load ontology file
        self.ontology_path = os.path.join(self.data_dir, "../multiwoz_dst/MULTIWOZ2.1/ontology.json")
        self.ontology = self._load_ontology(self.ontology_path)
        # setting uniform distribution
        self.uniform = True
        # number range of diff err
        self.extra_err_num_range = [1, 1]
        self.miss_err_num_range = [1, 1]
        self.match_err_num_range = [1, 1]

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
            self.dials += random.sample(self.target_dials, k=select_num)
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
        # load dialogs
        self.dials = list(self._load_json(self.data_path).values())

        self.target_dials = []
        for dial in self.dials:
            # use generated err or manually created err
            slots_str, miss_err_str, extr_err_str = self.creat_err(dial["slots_inf"])
            dial["slots_err"] = slots_str
            dial["miss_err"]  = miss_err_str
            dial["extra_err"] = extr_err_str

            self.target_dials.append(dial)

        with open("./experiment/tmp/created_errs.json", "w") as tf:
            json.dump(self.target_dials, tf, indent=2)
        return self.target_dials

    def creat_err(self, slots_inf):
        """
        create error by adding, replacing, removing
        for training correction model
        input : slots_inf = [dom1--slot_type1--slot_val1, dom2--slot_type2--slot_val2, ...]
        output: "dom1 type1 valx, dom2 ..."
        """
        slots_inf = self._extract_slot_from_string(slots_inf)
        slots_list = [slot.split("--") for slot in slots_inf]
        miss_err_list, extr_err_list = [], []
        dist = [1,1,0,1]
        err_type_cands = ["miss", "match", "extra", "none"]
        err_type = random.choices(err_type_cands, weights=dist, k=1)[0]
        if err_type == "miss":
            # # # missing err
            slots_list, miss_err_list = self._create_miss_err(slots_list)
        elif err_type == "match":
            # # # match err
            slots_list, miss_err_list, extr_err_list = self._create_matched_err(slots_list)
        elif err_type == "extra":
            # # # extra err
            slots_list, extr_err_list = self._create_extra_err(slots_list)

        # # # format list into string
        slots_str    = " ".join([" ".join(slot) + "," for slot in slots_list]) if slots_list!=[] else ""
        miss_err_str = " ".join([" ".join(slot) + "," for slot in miss_err_list]) if miss_err_list!=[] else ""
        extr_err_str = " ".join([" ".join(slot) + "," for slot in extr_err_list]) if extr_err_list!=[] else ""
        return slots_str, miss_err_str, extr_err_str
        
    def _create_matched_err(self, slots_list):
        """
        create error by replacing domain name or slot type
        or slot value, or serval of them
        input  : slots_list = [[dom1 type1 val1], ...]
        output : slots_list = [[dom1 typex valx], ...]

        param: err num per turn
               err ratio over types(add/remove/replace) 
               err ratio over domains
        """
        miss_err_list, extr_err_list = [], []
        # # # case of no slot
        if len(slots_list) == 0:
            return slots_list, miss_err_list, extr_err_list
        # # # set match err distribution dict
        self.match_dict = self.err_dist["count_err_type"]["match"]
        # # # decide num of matched err
        [err_num_min, err_num_max] = self.match_err_num_range
        err_num_max = min(err_num_max, len(slots_list) + 1)
        # no error
        if not err_num_max:
            return slots_list, miss_err_list, extr_err_list

        if self.uniform:
            dist = None
        else:
            dist = []
            for num in range(err_num_min, err_num_max + 1):
                dist.append(self.match_dict["slot_num"].get(str(len(slots_list)), {}).get(str(num), 0))
        err_num = weighted_sample(range(err_num_min, err_num_max + 1), weights=dist, k=1)[0]
        # # # choose err slot index
        err_idxs = weighted_sample(range(len(slots_list)), weights=None, k=err_num)
        # # # replace slot one by one
        for err_idx in err_idxs:
            miss_err_list.append(slots_list[err_idx])
            slots_list[err_idx] = self._replace_slot(slots_list, err_idx)
            extr_err_list.append(slots_list[err_idx])
        
        return slots_list, miss_err_list, extr_err_list

    def _replace_slot(self, slots_list, err_idx):
        """
        replacing domain name or slot type or slot value
        for a single slot

        input  : slot = [dom1 type1 val1]
        output : slot = [dom1 type1 valx]
        """
        [dom, slot_type, slot_val] = slots_list[err_idx]
        dom_set = set([slot[0] for slot in slots_list])
        type_set = set([slot[1] for slot in slots_list])
        # # # for domain
        # keep the same domain for now
        dom_err = dom

        # do not change for domain police and hospital
        if (dom_err not in self.ontology or
            dom_err not in self.match_dict["slot_type"]):
            return [dom, slot_type, slot_val]

        type_candidates = set(self.ontology[dom_err].keys()) - type_set
        type_candidates.add(slot_type)
        type_candidates = list(type_candidates)
        # TODO: change prior probability (freq + add-0.1) to conditional prob on current
        #  dom slot_type p(.| dom, slot_type)
        if self.uniform:
            dist = None
        else:
            dist = []
            for slot_type_ in type_candidates:
                dual_type = "--".join(sorted([slot_type, slot_type_]))
                dist.append(self.match_dict.get(dom_err, {}).get(dual_type, 0) + 0.5)
        type_err = weighted_sample(type_candidates, weights=dist, k=1)[0]
        
        val_candidates = self.ontology.get(dom_err, {}).get(type_err, [])
        # TODO: change prior probability (uniform) to conditional prob on current
        #  dom slot_type p(.| dom, slot_type)
        # compute probability
        dist = None
        # repalce
        if random.choice([0, 1]):
            val_err = slot_val
        else:
            val_err = weighted_sample(val_candidates, weights=dist, k=1)[0]

        return [dom_err, type_err, val_err]

    def _create_extra_err(self, slots_list):
        """
        create error adding a triplet
        first decide extra err num and then 
        dom --> slot_type --> slot_val
        input  : slots_list = [trip1, trip2]
        output : slots_list = [trip1, trip3, trip2]
        """
        self.extra_dict = self.err_dist["count_err_type"]["extra"]
        extr_err_list = []
        # # # decide num of extra err
        [err_num_min, err_num_max] = self.extra_err_num_range
        if not err_num_max:
            return slots_list, extr_err_list

        if self.uniform:
            dist = None
        else:
            dist = []
            for num in range(err_num_min, err_num_max + 1):
                dist.append(self.extra_dict["slot_num"].get(str(len(slots_list)), {}).get(str(num), 0))
        err_num = weighted_sample(range(err_num_min, err_num_max + 1), weights=dist, k=1)[0]
        # # # choose place index to insert err slot, range 0 ~ len(slots_list)
        dist = None
        err_idxs = weighted_sample(range(len(slots_list) + 1), weights=dist, k=err_num)
        # # # generate extra err and insert them to slots_list and append them to extr_err_list
        for err_idx in sorted(err_idxs, reverse=True):
            extr_slot = self._generate_slot(slots_list)

            slots_list.insert(err_idx, extr_slot)
            extr_err_list.insert(0, extr_slot)

        return slots_list, extr_err_list

    def _generate_slot(self, slots_list):
        """
        generate a single slot for extra slot err
        input  : 
        output : [dom type val]
        """
        # domains already shown up in slots_list
        dom_list = list(set([slot[0] for slot in slots_list]))
        # # # decide domain
        if self.uniform:
            dist = None
        else:
            dist = []
            for dom in DOMAINS:
                dist.append(self.extra_dict.get(dom, {}).get("total", 0))
        if slots_list == []: 
            # sample among all possible domains
            dom_err = weighted_sample(DOMAINS, weights=dist, k=1)[0]
        else:
            # sample within shown up domains
            dom_err = weighted_sample(dom_list, weights=dist, k=1)[0]

        # # # slot_type
        slot_type_candidates = list(self.ontology[dom_err].keys())
        if self.uniform:
            dist = None
        else:
            dist = []
            for slot_type in slot_type_candidates:
                dist.append(self.extra_dict.get(dom_err, {}).get(slot_type, 0))
        slot_type_err = weighted_sample(slot_type_candidates, weights=dist, k=1)[0]

        # # # slot_val
        slot_val_err = weighted_sample(self.ontology[dom_err][slot_type_err], k=1)[0]

        return [dom_err, slot_type_err, slot_val_err]
        
    def _create_miss_err(self, slots_list):
        """
        create error by removing a triplet
        input  : slots_list = [trip1, trip2, trip3]
        output : slots_list = [trip1, trip2]
        """
        miss_err_list = []
        # # # case of no slot
        if len(slots_list) == 0:
            return slots_list, miss_err_list

        self.miss_dict = self.err_dist["count_err_type"]["miss"]
        # # # decide num of miss err
        [err_num_min, err_num_max] = self.miss_err_num_range
        err_num_max = min(err_num_max, len(slots_list) + 1)
        if not err_num_max:
            return slots_list, miss_err_list

        if self.uniform:
            dist = None
        else:
            dist = []
            for num in range(err_num_min, err_num_max + 1):
                dist.append(self.miss_dict["slot_num"].get(str(len(slots_list)), {}).get(str(num), 0))
        err_num = weighted_sample(range(err_num_min, err_num_max + 1), weights=dist, k=1)[0]
        # # # decide which slot to remove
        if self.uniform:
            dist = None
        else:
            dist = []
            for slot in slots_list:
                dist.append(self.miss_dict.get(slot[0], {}).get(slot[1], 0) + 1)
        err_idxs = weighted_sample(range(len(slots_list)), weights=dist, k=err_num)
        # # # removing from end to start in list
        for err_idx in sorted(err_idxs, reverse=True):
            miss_err_list.insert(0, slots_list[err_idx])
            del slots_list[err_idx]
        return slots_list, miss_err_list

    def replace_err(self, dials_list, gen_test_result_path):
        """
        this function is used when doing iterative correction,
        by replacing error slots in the test data with the 
        generated result, specified by 'gen_test_result_path',
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
                "slots_err" : "dom1 type1 valx, dom2 ...",   <----- to be replaced
                "context"   : "<user> ... <system> ... <user> ... <bs> slots_err </bs>",
            },..
            ]
        """
        # load generated test result (list)
        gen_result = self._load_jsonl(gen_test_result_path)
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
            dial["slots_err"] = gen_result_dict[f"{dial_id}-{turn_num}"]
        return dials_list
