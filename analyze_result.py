#!/usr/bin/env python3
#
import os, sys, json
import math
import pdb
from collections import defaultdict, OrderedDict
	
from fuzzywuzzy import fuzz

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

    def _extract_slot_from_string(self, slots_string, context=None):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
        domains    = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]
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
                # if context is not None and slot_type != "internet" and slot_type != "parking" and slot_val not in context:
                #     continue
                # if slot_type == "leaveat" or slot_type == "arriveby":
                #     continue
                # if slot_type == "type":
                #     continue
                # if slot_val.split()[0] == "the":
                #     slot_val = " ".join(slot_val.split()[1:])
                # if "|" in slot_val:
                #     continue
                if not slot_val == 'dontcare':# and slot_type != "departure" and slot_type != "destination":
                    slots_list.append(domain+"--"+slot_type+"--"+slot_val)
        return slots_list

    def clean(self, raw_result=None, save_clean_result=False, save_clean_path=None, accm=0, replace_truth=0, truth_path=None, order='dtv', save_err=False):
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
        if raw_result is None:
            raw_result = self._load_jsonl(self.result_path)

        if replace_truth:
            data_path = '/checkpoint/kunqian/multiwoz/data/MultiWOZ_2.1/data_reformat_slot_accm_hist_accm.json'
            if truth_path is not None:
                data_path = truth_path
            with open(data_path) as df:
                ori_data = json.loads(df.read().lower())

        self.err = {}
        self.result_clean = {}
        count = defaultdict(int)
        for dial in raw_result:
            if "dial_id" not in dial["dialog"][0][0]:
                continue
            dial_id = dial["dialog"][0][0]['dial_id']
            turn_num = dial["dialog"][0][0]['turn_num']
            context = dial["dialog"][0][0]['text']
            gen_slots = dial["dialog"][0][1]['text']
            if not replace_truth:
                gt_slots = dial["dialog"][0][0]['eval_labels'][0]
            else:
                gt_slots = ori_data[f"{dial_id}-{turn_num}"]["slots_inf"]

            if dial_id not in self.result_clean:
                self.result_clean[dial_id] = {}
            if turn_num not in self.result_clean[dial_id]:
                self.result_clean[dial_id][turn_num] = {}

            # # # reformat ground truth
            slots_truth = self._extract_slot_from_string(gt_slots, context)

            # # # # reformat generated slots
            slots_pred = self._extract_slot_from_string(gen_slots, context)

            # # # fuzzy match
            for idx in range(len(slots_pred)):
                slot_pred = slots_pred[idx]
                dom_type_pred = "--".join(slot_pred.split("--")[:2])
                slot_val_pred = slot_pred.split("--")[-1]
                for slot_truth in slots_truth:
                    dom_type_truth = "--".join(slot_truth.split("--")[:2])
                    slot_val_truth = slot_truth.split("--")[-1]
                    if dom_type_pred == dom_type_truth and slot_val_pred != slot_val_truth:
                        if fuzz.partial_ratio(slot_val_pred, slot_val_truth) >= 90 \
                            and len(slot_val_pred.split("|")) == len(slot_val_truth.split("|")):
                            # print(slot_pred)
                            # print(slot_truth)
                            # pdb.set_trace()
                            count[dom_type_pred] += 1
                            slots_pred[idx] = slot_truth
                            # pdb.set_trace()
                            break
            # # print(count)

            self.result_clean[dial_id][turn_num]["ground_truth"] = sorted(list(set(slots_truth)))
            self.result_clean[dial_id][turn_num]["generated_seq"] = sorted(list(set(slots_pred)))

            self.result_clean[dial_id][turn_num]["tru_seq"] = ", ".join(sorted(list(set(slots_truth))))
            self.result_clean[dial_id][turn_num]["gen_seq"] = ", ".join(sorted(list(set(slots_pred))))

            self.result_clean[dial_id][turn_num]["context"] = context

            if save_err and set(slots_truth)!=set(slots_pred):
                miss_err = list(set(slots_truth)-set(slots_pred))
                extr_err = list(set(slots_pred)-set(slots_truth))
                if dial_id not in self.err:
                    self.err[dial_id] = {}
                self.err[dial_id][turn_num] = {
                    "context"  : context,
                    "miss_err" : ", ".join(miss_err),
                    "extr_err" : ", ".join(extr_err),
                    "oracle_s" : ", ".join(sorted(list(set(slots_truth)))),
                }

        # for dom_type in count:
        #     print(dom_type+": "+str(count[dom_type]))

        if save_err:
            with open(self.err_result_path, "w") as ef:
                json.dump(self.err, ef, indent=2)

        if save_clean_result:
            if save_clean_path is None:
                save_clean_path = self.result_clean_path
            with open(save_clean_path, "w") as tf:
                json.dump(self.result_clean, tf, indent=2)

    def clean_all_type_seq(self, raw_result=None, save_clean_result=False, save_clean_path=None):

        """
        format raw_result (string of slots) into triplets
        input : [
            {
                "dialog" :
                    [
                        [
                            {
                                "dial_id"   : dial_id,
                                "turn_num"  : turn_num,
                                "eval_labels" : ["true slot value"],
                                "text"   : "(<user> ... <system> ... <user> ... )<bs> domain slot_type </bs>",
                            },
                            {
                                "text" : "generated slot value",
                                ...
                            }
                        ],
                        ...
                    ]
            }
        ]
        output: {"dial_id":{"turn_num":{"ground_truth": ["dom-slot_type-slot_val", ...],
                                       "generated_seq": ["dom-slot_type-slot_val", ...]}, ...}, ...}
        """
        if raw_result is None:
            raw_result = self._load_jsonl(self.result_path)
        # import pdb
        # pdb.set_trace()

        self.result_clean = {}
        for dial in raw_result:
            if "dial_id" not in dial["dialog"][0][0]:
                continue
            
            slots_truth, slots_pred = [], []
            for slot_num in range(len(dial["dialog"])):
                dial_id = dial["dialog"][slot_num][0]['dial_id']
                turn_num = dial["dialog"][slot_num][0]['turn_num']
                context = dial["dialog"][slot_num][0]['text']
                gt_val = dial["dialog"][slot_num][0]['eval_labels'][0]
                gen_val = dial["dialog"][slot_num][1]['text']

                # init dict
                if dial_id not in self.result_clean:
                    self.result_clean[dial_id] = {}
                if turn_num not in self.result_clean[dial_id]:
                    self.result_clean[dial_id][turn_num] = {}
                if context.split("<bs>")[0] != "":
                    self.result_clean[dial_id][turn_num]["text"] = context.split("<bs>")[0]
                # extract domain and slot_type
                if "<bs>" not in context:
                    continue
                dom_type = context.split("<bs>")[-1].split("</bs>")[0].strip().split()
                if dom_type == []:
                    continue
                domain = dom_type[0]
                slot_type = " ".join(dom_type[1:])

                # # # reformat ground truth
                if gt_val != 'dontcare' and gt_val != 'none':
                    slots_truth.append(domain+"--"+slot_type+"--"+gt_val)

                # # # reformat generated slots
                if gen_val != 'dontcare' and gen_val != 'none':
                    slots_pred.append(domain+"--"+slot_type+"--"+gen_val)

                # pdb.set_trace()

            self.result_clean[dial_id][turn_num]["ground_truth"] = sorted(list(set(slots_truth)))
            self.result_clean[dial_id][turn_num]["generated_seq"] = sorted(list(set(slots_pred)))

            self.result_clean[dial_id][turn_num]["tru_seq"] = ", ".join(sorted(list(set(slots_truth))))
            self.result_clean[dial_id][turn_num]["gen_seq"] = ", ".join(sorted(list(set(slots_pred))))
            

        if save_clean_result:
            if save_clean_path is None:
                save_clean_path = self.result_clean_path
            with open(save_clean_path, "w") as tf:
                json.dump(self.result_clean, tf, indent=2)

    def clean_all_type_sep(self, raw_result=None, save_clean_result=False, save_clean_path=None):

        """
        format raw_result (string of slots) into triplets
        input : [
            {
                "dialog" :
                    [
                        [
                            {
                                "dial_id"   : dial_id,
                                "turn_num"  : turn_num,
                                "eval_labels" : ["true slot value"],
                                "text"   : "(<user> ... <system> ... <user> ... )<bs> domain slot_type </bs>",
                            },
                            {
                                "text" : "generated slot value",
                                ...
                            }
                        ],
                        ...
                    ]
            }
        ]
        output: {"dial_id":{"turn_num":{"ground_truth": ["dom-slot_type-slot_val", ...],
                                       "generated_seq": ["dom-slot_type-slot_val", ...]}, ...}, ...}
        """
        if raw_result is None:
            raw_result = self._load_jsonl(self.result_path)
        # import pdb
        # pdb.set_trace()

        self.result_clean = {}
        result_dict = {}
        for dial in raw_result:
            if "dial_id" not in dial["dialog"][0][0]:
                continue
            
            slots_truth, slots_pred = [], []

            dial_id = dial["dialog"][0][0]['dial_id']
            turn_num = dial["dialog"][0][0]['turn_num']
            context = dial["dialog"][0][0]['text']
            gt_val = dial["dialog"][0][0]['eval_labels'][0]
            gen_val = dial["dialog"][0][1]['text']

            # init dict
            if dial_id not in result_dict:
                result_dict[dial_id] = {}
            if turn_num not in result_dict[dial_id]:
                result_dict[dial_id][turn_num] = {"text": context, "dst":{}}

            # extract domain and slot_type
            if "<bs>" not in context:
                continue
            dom_type = context.split("<bs>")[-1].split("</bs>")[0].strip().split()
            if dom_type == []:
                continue
            domain = dom_type[0]
            slot_type = " ".join(dom_type[1:])

            # init dict
            if domain not in result_dict[dial_id][turn_num]["dst"]:
                result_dict[dial_id][turn_num]["dst"][domain] = {}
            if slot_type not in result_dict[dial_id][turn_num]["dst"][domain]:
                result_dict[dial_id][turn_num]["dst"][domain][slot_type] = []
            else:
                pdb.set_trace()

            result_dict[dial_id][turn_num]["dst"][domain][slot_type] += [gt_val, gen_val]

        for dial_id, dial in result_dict.items():
            self.result_clean[dial_id] = {}
            for turn_num, turn in dial.items():
                self.result_clean[dial_id][turn_num] = {}
                slots_truth, slots_pred = [] , []
                for domain in turn["dst"]:
                    for slot_type in turn["dst"][domain]:
                        [gt_val, gen_val] = turn["dst"][domain][slot_type]

                        # # # reformat ground truth
                        if gt_val != 'dontcare' and gt_val != 'none':
                            slots_truth.append(domain+"--"+slot_type+"--"+gt_val)
                        # # # reformat generated slots
                        if gen_val != 'dontcare' and gen_val != 'none':
                            slots_pred.append(domain+"--"+slot_type+"--"+gen_val)

                self.result_clean[dial_id][turn_num]["ground_truth"] = sorted(list(set(slots_truth)))
                self.result_clean[dial_id][turn_num]["generated_seq"] = sorted(list(set(slots_pred)))

                self.result_clean[dial_id][turn_num]["tru_seq"] = ", ".join(sorted(list(set(slots_truth))))
                self.result_clean[dial_id][turn_num]["gen_seq"] = ", ".join(sorted(list(set(slots_pred))))
                self.result_clean[dial_id][turn_num]["dialog"] = turn["text"]

        if save_clean_result:
            if save_clean_path is None:
                save_clean_path = self.result_clean_path
            with open(save_clean_path, "w") as tf:
                json.dump(self.result_clean, tf, indent=2)

    def clean_err(self, raw_result=None, save_clean_result=False, save_clean_path=None, accm=0, replace_truth=0, order='dtv'):
        """
        format raw_result (string of slots) into triplets
        input : [
            {
                "dial_id"   : dial_id,
                "turn_num"  : turn_num,
                "slots_inf" : "dom1 type1 val1, dom2 ...",
                "slots_err" : "dom1 type1 valx, dom2 ...",
                "slots_kperr" : "dom1 type1 valx",  # only errs
                "context"   : "<user> ... <system> ... <user> ... <bs> slots_err </bs>",
            },..
            ]
        output: {"dial_id":{"turn_num":{"ground_truth": ["dom-slot_type-slot_val", ...],
                                       "generated_seq": ["dom-slot_type-slot_val", ...]}, ...}, ...}
        """
        if raw_result is None:
            raw_result = self._load_jsonl(self.result_path)
        # pdb.set_trace()

        self.result_clean = {}

        # # # original no err
        no_err  = 0             # total
        no_err_no_cor = 0       # no change
        no_err_fa_cor = 0       # false alarm
        no_err_fa_cor_add = 0   # fa, adding and get extra err
        no_err_fa_cor_rem = 0   # fa, removing and get miss err

        # # # original with err
        err     = 0             # total
        err_no_cor = 0          # no change
        err_no_cor_mi = 0       # no change with miss err
        err_no_cor_ex = 0       # no change with extra err
        err_fa_cor = 0          # change but still wrong
        err_fa_cor_add = 0      # change but still wrong by adding
        err_fa_cor_rem = 0      # change but still wrong by removing
        err_su_cor = 0          # success correction
        err_su_cor_add = 0      # success correction by adding to fix miss err
        err_su_cor_rem = 0      # success correction by removing to fix extra err

        for dial in raw_result:
            if "dial_id" not in dial["dialog"][0][0]:
                continue
            # index
            dial_id = dial["dialog"][0][0]['dial_id']
            turn_num = dial["dialog"][0][0]['turn_num']

            if dial_id not in self.result_clean:
                self.result_clean[dial_id] = {}
            if turn_num not in self.result_clean[dial_id]:
                self.result_clean[dial_id][turn_num] = {}

            # extract slots
            context = dial["dialog"][0][0]['text']
            # previous generated slots
            gen_slots = context.split("<bs>")[-1].split("</bs>")[0].strip()
            # ground truth slots
            gt_slots = dial["dialog"][0][0]['slots_inf']
            # ground truth err slots (compared with gen_slots)
            gt_err_slots = dial["dialog"][0][0]['eval_labels'][0]
            # generated err
            gen_err_slots = dial["dialog"][0][1]['text']

            # # # turn ground truth into list
            slots_truth = self._extract_slot_from_string(gt_slots)
            # # # turn previous generated slots into list
            slots_gen = self._extract_slot_from_string(gen_slots)
            # # # turn generated err slots into list
            slots_miss_err, slots_extra_err = [], []
            # # # turn ground truth err slots into list
            slots_miss_err_gt, slots_extra_err_gt = [], []
            # # format corrected result 
            # if ("kpextra" in self.result_path or 
            #     ("kpe" in self.result_path and "kperr" not in self.result_path)):
            #     slots_extra_err = self._extract_slot_from_string(gen_err_slots)
            #     slots_extra_err_gt = self._extract_slot_from_string(gt_err_slots)
            # if "kpmiss" in self.result_path:
            #     slots_miss_err = self._extract_slot_from_string(gen_err_slots)
            #     slots_miss_err_gt = self._extract_slot_from_string(gt_err_slots)
            # if "kperr" in self.result_path:
            gen_err_extra   = gen_err_slots.split("<m>")[-1].split("<e>")[0].strip()
            gen_err_miss    = gen_err_slots.split("<m>")[-1].split("<e>")[-1].strip()
            slots_extra_err = self._extract_slot_from_string(gen_err_extra)
            slots_miss_err  = self._extract_slot_from_string(gen_err_miss)
            # for ground truth err
            gen_err_extra_gt   = gt_err_slots.split("<m>")[-1].split("<e>")[0].strip()
            gen_err_miss_gt    = gt_err_slots.split("<m>")[-1].split("<e>")[-1].strip()
            slots_extra_err_gt = self._extract_slot_from_string(gen_err_extra_gt)
            slots_miss_err_gt  = self._extract_slot_from_string(gen_err_miss_gt)

            # format corrected result 
            slots_cor = list(set(slots_gen) - set(slots_extra_err))
            slots_cor = slots_cor + slots_miss_err
            
            self.result_clean[dial_id][turn_num]["ground_truth"] = sorted(list(set(slots_truth)))
            self.result_clean[dial_id][turn_num]["generated_seq"] = sorted(list(set(slots_cor)))

            self.result_clean[dial_id][turn_num]["tru_seq"] = ", ".join(sorted(list(set(slots_truth))))
            self.result_clean[dial_id][turn_num]["gen_seq"] = ", ".join(sorted(list(set(slots_cor))))


            # analyze correction
            if len(slots_miss_err_gt) + len(slots_extra_err_gt) == 0:
                # original with no err
                no_err += 1
                if len(slots_extra_err) + len(slots_miss_err) == 0:
                    # no change (true positive)
                    no_err_no_cor += 1
                else:
                    # false alarm
                    no_err_fa_cor += 1
                    if len(slots_miss_err) > 0:
                        # fa by adding slot (falsely detect miss err)
                        no_err_fa_cor_add += 1
                    if len(slots_extra_err) > 0:
                        # fa by removing slot (falsely detect extra err)
                        no_err_fa_cor_rem += 1
            else:
                # original with err
                err += 1
                if len(slots_extra_err) + len(slots_miss_err) == 0:
                    # no change (False Positive)
                    err_no_cor += 1
                    if len(slots_miss_err_gt) > 0:
                        # no change with miss err
                        err_no_cor_mi += 1
                    if len(slots_extra_err_gt) > 0:
                        # no change with extra err
                        err_no_cor_ex += 1
                elif set(slots_miss_err) == set(slots_miss_err_gt) and \
                     set(slots_extra_err) == set(slots_extra_err_gt):
                    # success correction
                    err_su_cor += 1
                    if len(slots_miss_err) > 0:
                        # success correction by adding to fix miss err
                        err_su_cor_add += 1
                    if len(slots_extra_err) > 0:
                        # success correction by removing to fix extra err
                        err_su_cor_rem += 1
                else:
                    # change but still wrong
                    err_fa_cor += 1
                    if set(slots_miss_err) != set(slots_miss_err_gt) and len(set(slots_miss_err)) > 0:
                        # change but still wrong by adding
                        err_fa_cor_add += 1
                    if set(slots_extra_err) != set(slots_extra_err_gt) and len(set(slots_extra_err)) > 0:
                        # change but still wrong by removing
                        err_fa_cor_rem += 1



        print(f"no err turns:{no_err} with {no_err_no_cor} no correction and {no_err_fa_cor} ({no_err_fa_cor_add} add and {no_err_fa_cor_rem} remove) wrong corrections")
        print(f"err turns:{err} with {err_no_cor} ({err_no_cor_mi} miss and {err_no_cor_ex} extra) no correction \n\
                and {err_fa_cor} ({err_fa_cor_add} add and {err_fa_cor_rem} remove) wrong corrections and {err_su_cor} ({err_su_cor_add} add and {err_su_cor_rem} remove) success corrections")
                



        # saving
        if save_clean_result:
            if save_clean_path is None:
                save_clean_path = self.result_clean_path
            with open(save_clean_path, "w") as tf:
                json.dump(self.result_clean, tf, indent=2)

    def load_result_clean(self):
        with open(self.result_clean_path) as df:
            self.result_clean = json.loads(df.read().lower())

    def load_result_err(self):
        with open(self.err_result_path) as df:
            self.result_err = json.loads(df.read().lower())

    def compute_slot_acc_turn(self, truth, pred, total_slot_type=30):
        """
        compute slots acc for each turn:
            does each slot_type are predicted correctly,
            (including slot_types that does not show up)
        """
        wrong_predict = set(pred) - set(truth)
        miss_predict = set(truth) - set(pred)
        correct_predict_num = total_slot_type - len(wrong_predict) - len(miss_predict)
        return correct_predict_num/float(total_slot_type)

    def compute_metrics(self, clean_result=None):
        """
        compute joint_goal_acc and slots_acc
        """
        # # # load cleaned results
        if clean_result is None:
            if self.result_clean is None:
                self.load_result_clean()
            clean_result = self.result_clean
        
        total_turn = 0
        correct_turn = 0
        slots_acc = 0
        prec_type, reca_type = 0, 0
        prec_val, reca_val = 0, 0
        total_tp_type, total_tp_fp_type, total_tp_fn_type = 0, 0, 0 
        total_tp_val, total_tp_fp_val, total_tp_fn_val = 0, 0, 0 
        gen_type_num = 0

        turn_n = 0


        for dial_id, dial in clean_result.items():
            for turn_num, turn in dial.items():
                total_turn += 1
                # # # slots acc
                slots_acc += self.compute_slot_acc_turn(turn["ground_truth"], turn["generated_seq"])

                # # # precision & recall 
                # for slot_type 
                gt_type  = set(["--".join(slot.split("--")[:2]) for slot in turn["ground_truth"]])
                gen_type = set(["--".join(slot.split("--")[:2]) for slot in turn["generated_seq"]])
                prec_type += len(gen_type.intersection(gt_type)) / float(len(gen_type) + 1e-10)
                reca_type += len(gen_type.intersection(gt_type)) / float(len(gt_type) + 1e-10)
                total_tp_type += len(gen_type.intersection(gt_type))
                total_tp_fp_type += len(gen_type)
                total_tp_fn_type += len(gt_type)
                # for slot_value
                gt_val   = set([slot.split("--")[2] for slot in turn["ground_truth"]])
                gen_val  = set([slot.split("--")[2] for slot in turn["generated_seq"]])
                prec_val += len(gen_val.intersection(gt_val)) / float(len(gen_val) + 1e-10)
                reca_val += len(gen_val.intersection(gt_val)) / float(len(gt_val) + 1e-10)
                total_tp_val += len(gen_val.intersection(gt_val))
                total_tp_fp_val += len(gen_val)
                total_tp_fn_val += len(gt_val)


                # # # joint goal acc
                if set(turn["generated_seq"]) == set(turn["ground_truth"]):
                    correct_turn += 1

        joint_goal_acc = correct_turn / float(total_turn)
        slots_acc /= float(total_turn)
        prec_type /= total_turn
        reca_type /= total_turn
        prec_val /= total_turn
        reca_val /= total_turn
        precision_type = total_tp_type / (total_tp_fp_type + 1e-10)
        recall_type = total_tp_type / (total_tp_fn_type + 1e-10)
        precision_val = total_tp_val / (total_tp_fp_val + 1e-10)
        recall_val = total_tp_val / (total_tp_fn_val + 1e-10)
    
        
        print(f"joint goal acc: {joint_goal_acc}, slots acc: {slots_acc}")

        # print(f"slot type : turn precision: {prec_type}, recall: {reca_type}; total precision: {precision_type}, recall: {recall_type}")
        # print(f"slot value: turn precision: {prec_val} , recall: {reca_val} ; total precision: {precision_val} , recall: {recall_val}")

    def Count_Err_Num(self, clean_result):
        """
        compute the ratio of diff error num in each turn
        (how many turns contains one/two/... (missing/extra) slots)
        """
        stats = {"extra":defaultdict(int), "miss":defaultdict(int), "err":defaultdict(int)}
        # tn --> turn_num
        extra_tn, miss_tn, total_tn = 0, 0, 0
        extra_1_tn, miss_1_tn = 0, 0
        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                extra_predict = set(turn["generated_seq"]) - set(turn["ground_truth"])
                miss_predict = set(turn["ground_truth"]) - set(turn["generated_seq"])

                extra_num = len(extra_predict)
                miss_num = len(miss_predict)
                err_num = len(extra_predict) + len(miss_predict)
                stats["extra"][extra_num] += 1
                stats["miss"][miss_num] += 1
                stats["err"][err_num] += 1
                
                if len(extra_predict) > 0:
                    extra_tn += 1
                    if len(extra_predict) == 1:
                        extra_1_tn += 1
                if len(miss_predict) > 0:
                    miss_tn += 1
                    if len(miss_predict) == 1:
                        miss_1_tn += 1
                
                total_tn += 1
        stats["total_turn"] = sum(list(stats["extra"].values()))
        stats["total_extra_turn_num"] = sum([stats["extra"][i] * 1 for i in stats["extra"]])-stats["extra"][0]
        stats["total_miss_turn_num"]  = sum([stats["miss"][i] * 1 for i in stats["miss"]])-stats["miss"][0]
        stats["total_turn_num_w_err"] = sum([stats["err"][i] * 1 for i in stats["err"]])-stats["err"][0]
        stats["total_extra_err_num"]  = sum([stats["extra"][i] * int(i) for i in stats["extra"]])
        stats["total_miss_err_num"]   = sum([stats["miss"][i] * int(i) for i in stats["miss"]])
        stats["total_err_num_w_err"]  = sum([stats["err"][i] * int(i) for i in stats["err"]])
        # for _key in ["extra", "miss", "err"]:
        #     total_num = float(sum(list(stats[_key].values())))
        #     for num in stats[_key]:
        #         stats[_key][num] = [stats[_key][num], stats[_key][num]/total_num]

        stats["extra"] = OrderedDict(sorted(stats["extra"].items(), key=lambda t: t[0]))
        stats["miss"] = OrderedDict(sorted(stats["miss"].items(), key=lambda t: t[0]))
        stats["err"] = OrderedDict(sorted(stats["err"].items(), key=lambda t: t[0]))
        return stats

    def Count_Err_Type(self, clean_result):
        """
        To explore the ratio of err from domain or slot_type or slot_val.
        This should only focus on the turns with both missing and extra slots.
        Those turns with either only missing or extra slots should not be considered.
        ground truth:    dom1 slot_type1 slot_val1
        generated slot:  dom1 slot_type1 slot_val2 --> val err
                    or   dom1 slot_type2 slot_val1 --> typ err
                    or   dom2 slot_type1 slot_val1 --> dom err

            other case   dom1 slot_type2 slot_val2
                         dom2 slot_type1 slot_val2
                         dom2 slot_type2 slot_val1
                         dom2 slot_type2 slot_val2

        other thinking:  d1 t1 v1    <--->  d1 t1 v2 ,  d1 t1 v3
            or                              d1 t1 v2 ,  d1 t2 v1
            ....
        """
        extra_slot_path = self.analyze_result_path.replace(".json", "_extra.json")
        miss_slot_path = self.analyze_result_path.replace(".json", "_miss.json") # unmatched missing
        extra_slot_turn = {"domain":[], "slot_type":[], "slot_val":[], "other":[]}
        unmatched_miss_slot_turn = []

        stats = {
                "overall": {
                    "domain"    :0,   # for matched
                    "slot_type" :0,   # for matched
                    "slot_val"  :0,   # for matched
                    "extra_slot_num":0,
                    "match_slot_num":0,
                    "miss_slot_num" :0,
                    },
                "match" : {
                    "domain"    : {},  # dom1-dom2-slot_type
                    "slot_type" : {},  # dom-slot_type1-slot_typ2
                    "slot_val"  : {"dom_type":defaultdict(int),},
                    "slot_num"  : {},
                    },
                "extra" : {
                            "slot_num" :{},
                            "slot_type": defaultdict(int),
                            "taxi--departure":defaultdict(int),
                           },
                "miss"  : {
                            "slot_num":{},
                            "slot_type": defaultdict(int),
                            "taxi--departure":defaultdict(int),
                            },
        }
        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                extra_predict = set(turn["generated_seq"]) - set(turn["ground_truth"])
                miss_predict = set(turn["ground_truth"]) - set(turn["generated_seq"])
                extra_predict_split = [slot.split("--") for slot in extra_predict]
                miss_predict_split = [slot.split("--") for slot in miss_predict]
                stats["overall"]["extra_slot_num"] += len(extra_predict)
                stats["overall"]["miss_slot_num"] += len(miss_predict)

                matched_slot = []
                for extra_slot in extra_predict_split:
                    match_flag = 0
                    stats["extra"]["slot_type"][extra_slot[0] + "--" + extra_slot[1]] += 1
                    for miss_slot in miss_predict_split:
                        # for matched errs
                        if extra_slot[0] == miss_slot[0]:
                            if extra_slot[1] == miss_slot[1]:
                                stats["overall"]["slot_val"] += 1
                                # detailed count    dom:{type: num}
                                if extra_slot[0] not in stats["match"]["slot_val"]:
                                    stats["match"]["slot_val"][extra_slot[0]] = {"total" : 0, extra_slot[1] : 0}
                                elif extra_slot[1] not in stats["match"]["slot_val"][extra_slot[0]]:
                                    stats["match"]["slot_val"][extra_slot[0]][extra_slot[1]] = 0
                                stats["match"]["slot_val"][extra_slot[0]][extra_slot[1]] += 1
                                stats["match"]["slot_val"][extra_slot[0]]["total"] += 1
                                stats["match"]["slot_val"]["dom_type"][extra_slot[0]+"_"+extra_slot[1]] += 1
                                # saving
                                if miss_slot not in matched_slot:
                                    matched_slot.append(miss_slot)
                                match_flag = 1
                                extra_slot_turn["slot_val"].append({
                                        "dial_id": dial_id,
                                        "turn_num": tn,
                                        "extr_slot" : "--".join(extra_slot),
                                        "miss_slot" : "--".join(miss_slot),
                                        "gen_slot"  : ", ".join(sorted(turn["generated_seq"])),
                                        "tru_slot"  : ", ".join(sorted(turn["ground_truth"]))
                                })
                                break
                                
                            elif extra_slot[2] == miss_slot[2]:
                                stats["overall"]["slot_type"] += 1
                                # detailed count    dom:{type1--type2: num}
                                dual_type = "--".join(sorted([extra_slot[1], miss_slot[1]]))
                                if extra_slot[0] not in stats["match"]["slot_type"]:
                                    stats["match"]["slot_type"][extra_slot[0]] = {"total" : 0, dual_type : 0}
                                elif dual_type not in stats["match"]["slot_type"][extra_slot[0]]:
                                    stats["match"]["slot_type"][extra_slot[0]][dual_type] = 0
                                stats["match"]["slot_type"][extra_slot[0]][dual_type] += 1
                                stats["match"]["slot_type"][extra_slot[0]]["total"] += 1
                                # saving
                                if miss_slot not in matched_slot:
                                    matched_slot.append(miss_slot)
                                match_flag = 1
                                extra_slot_turn["slot_type"].append({
                                        "dial_id": dial_id,
                                        "turn_num": tn,
                                        "extr_slot" : "--".join(extra_slot),
                                        "miss_slot" : "--".join(miss_slot),
                                        "gen_slot"  : ", ".join(sorted(turn["generated_seq"])),
                                        "tru_slot"  : ", ".join(sorted(turn["ground_truth"]))
                                })
                                break
                                
                        elif extra_slot[1] == miss_slot[1]:
                            if extra_slot[2] == miss_slot[2]:
                                # overall count
                                stats["overall"]["domain"] += 1
                                # detailed count    dom1-dom2:{type: num}
                                # dual_dom = "--".join([extra_slot[0], miss_slot[0]])
                                dual_dom = "--".join(sorted([extra_slot[0], miss_slot[0]]))
                                if dual_dom not in stats["match"]["domain"]:
                                    stats["match"]["domain"][dual_dom] = {"total" : 0, extra_slot[1] : 0}
                                elif extra_slot[1] not in stats["match"]["domain"][dual_dom]:
                                    stats["match"]["domain"][dual_dom][extra_slot[1]] = 0
                                stats["match"]["domain"][dual_dom][extra_slot[1]] += 1
                                stats["match"]["domain"][dual_dom]["total"] += 1
                                # saving
                                if miss_slot not in matched_slot:
                                    matched_slot.append(miss_slot)
                                match_flag = 1
                                extra_slot_turn["domain"].append({
                                        "dial_id": dial_id,
                                        "turn_num": tn,
                                        "extr_slot" : "--".join(extra_slot),
                                        "miss_slot" : "--".join(miss_slot),
                                        "gen_slot"  : ", ".join(sorted(turn["generated_seq"])),
                                        "tru_slot"  : ", ".join(sorted(turn["ground_truth"]))
                                })
                                break
                    if match_flag == 1:
                        stats["overall"]["match_slot_num"] += 1
                    else:
                        # saving
                        extra_slot_turn["other"].append({
                                "dial_id": dial_id,
                                "turn_num": tn,
                                "extr_slot" : "--".join(extra_slot),
                                "gen_slot"  : ", ".join(sorted(turn["generated_seq"])),
                                "tru_slot"  : ", ".join(sorted(turn["ground_truth"]))
                        })

                    # # # # for unmatched extra slots
                    # if extra_slot[0] not in stats["extra"]:
                    #     stats["extra"][extra_slot[0]] = {"total" : 0, extra_slot[1] : 0}
                    # elif extra_slot[1] not in stats["extra"][extra_slot[0]]:
                    #     stats["extra"][extra_slot[0]][extra_slot[1]] = 0
                    # stats["extra"][extra_slot[0]][extra_slot[1]] += 1
                    # stats["extra"][extra_slot[0]]["total"] += 1


                # # # for unmatched miss slots
                for miss_slot in miss_predict_split:
                    stats["miss"]["slot_type"][miss_slot[0] + "--" + miss_slot[1]] += 1
                    
        stats["miss"]["slot_type"] = OrderedDict(sorted(stats["miss"]["slot_type"].items(), key=lambda t: t[0]))
        stats["extra"]["slot_type"] = OrderedDict(sorted(stats["extra"]["slot_type"].items(), key=lambda t: t[0]))
        stats["match"]["slot_val"]["dom_type"] = OrderedDict(sorted(stats["match"]["slot_val"]["dom_type"].items(), key=lambda t: t[0]))

        stats["extra"]["taxi--departure"] = OrderedDict(sorted(stats["extra"]["taxi--departure"].items(), key=lambda t: t[1], reverse=True))
        stats["miss"]["taxi--departure"] = OrderedDict(sorted(stats["miss"]["taxi--departure"].items(), key=lambda t: t[1], reverse=True))
                                
        return stats

    def Count_Dontcare(self, clean_result=None):
        """
        count the num of "dom1 typ1 do not care" generated while 
        "dom1 typ1 xxx" does not in ground truth 
        or "dom1 typ1 do not care" in ground truth while not
        generating "dom1 typ1 xxx"

        format of "do not care":
                do nt care
                don't care
        """
        
        total_turn = 0.
        correct_turn = 0
        correct_turn_gen = 0    # ignoring generated do_not_care
        dontcare_gen = 0
        correct_turn_tru = 0    # ignoring groundtruth do_not_care
        dontcare_tru = 0
        dontcare_list = set()   # store diff kinds of do_not_care
        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                # # # list all kinds of "do_not_care"
                for key in turn:
                    for slot in turn[key]:
                        if "care" in slot:
                            dontcare_list.add(slot.split("--")[-1])
                

                total_turn += 1
                # # # joint goal acc
                if set(turn["generated_seq"]) == set(turn["ground_truth"]):
                    correct_turn += 1

                else:
                    extra_predict = set(turn["generated_seq"]) - set(turn["ground_truth"])
                    miss_predict = set(turn["ground_truth"]) - set(turn["generated_seq"])

                    # # # joint goal acc ignoring generated do_not_care
                    flag1 = 1
                    for extra_slot in extra_predict:
                        if "care" not in extra_slot:
                            flag1 = 0
                            break
                    if flag1 == 1:
                        dontcare_gen += 1
                        if len(miss_predict) == 0:
                            correct_turn_gen += 1
                    

                    # # # joint goal acc ignoring groundtruth do_not_care
                    flag2 = 1
                    for miss_slot in miss_predict:
                        if "care" not in miss_slot:
                            flag2 = 0
                            break
                    if flag2 == 1:
                        dontcare_tru += 1
                        if len(extra_predict) == 0:
                            correct_turn_tru += 1

                    # if flag1 == 1 and flag2 == 1:
                    #     pdb.set_trace()
            
        joint_goal_acc = correct_turn / total_turn
        joint_goal_acc_wo_gen_dotca = correct_turn_gen / total_turn
        joint_goal_acc_wo_tru_dotca = correct_turn_tru / total_turn

        stats = {
                 "dontcare_list" : list(dontcare_list),
                 "joint_goal_acc" : joint_goal_acc,
                 "joint_goal_acc_wo_gen_dotca" : joint_goal_acc_wo_gen_dotca,
                 "joint_goal_acc_wo_tru_dotca" : joint_goal_acc_wo_tru_dotca,
                 "correct_turn" : correct_turn,
                 "correct_turn_gen" : correct_turn_gen,
                 "correct_turn_tru" : correct_turn_tru,
                 "dontcare_gen" : dontcare_gen,
                 "dontcare_tru" : dontcare_tru,
                 "total_turn" : total_turn
                 }

        return stats
        # print(dontcare_list)
        # print(joint_goal_acc, joint_goal_acc_wo_gen_dotca, joint_goal_acc_wo_tru_dotca)
        # print(correct_turn, correct_turn_gen, correct_turn_tru, total_turn)

    def Count_Correct_Ratio(self, clean_result, gen_data_path=None):

        gen_data_path = "data/multiwozdst_cor/dials_nodict_bs1_test.json"
        gen_data_path = "data/multiwozdst_cor_err/dials_nodict_bs1_test.json"
        with open(gen_data_path) as df:
            origin_data = json.loads(df.read().lower())
        # # # original no err
        no_err  = 0             # total
        no_err_no_cor = 0       # no change
        no_err_fa_cor = 0       # false alarm
        no_err_fa_cor_add = 0   # fa, adding and get extra err
        no_err_fa_cor_rem = 0   # fa, removing and get miss err

        # # # original with err
        err     = 0             # total
        err_no_cor = 0          # no change
        err_no_cor_mi = 0       # no change with miss err
        err_no_cor_ex = 0       # no change with extra err
        err_fa_cor = 0          # change but still wrong
        err_fa_cor_add = 0      # change but still wrong by adding
        err_fa_cor_rem = 0      # change but still wrong by removing
        err_su_cor = 0          # success correction
        err_su_cor_add = 0      # success correction by adding to fix miss err
        err_su_cor_rem = 0      # success correction by removing to fix extra err

        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                if dial_id+"-"+str(tn) not in origin_data:
                    continue
                err_slots_str = origin_data[dial_id+"-"+str(tn)]["slots_err"]
                err_slots_list = self._extract_slot_from_string(err_slots_str)
                
                tru_slots = turn["ground_truth"]
                gen_slots = turn["generated_seq"]

                if set(err_slots_list) == set(tru_slots):
                    # original with no err
                    no_err += 1
                    if set(tru_slots) == set(gen_slots):
                        # no change
                        no_err_no_cor += 1
                    else:
                        # false alarm
                        no_err_fa_cor += 1
                        if len(set(err_slots_list) - set(gen_slots)) > 0:
                            # fa, removing and get miss err
                            no_err_fa_cor_rem += 1
                        if len(set(gen_slots) - set(err_slots_list)) > 0:
                            # fa, adding and get extra err
                            no_err_fa_cor_add += 1
                else:
                    # original with err
                    err += 1
                    if set(gen_slots) == set(err_slots_list):
                        # no change
                        err_no_cor += 1
                        if len(set(tru_slots) - set(gen_slots)) > 0:
                            # no change with miss err
                            err_no_cor_mi += 1
                        if len(set(gen_slots) - set(tru_slots)) > 0:
                            # no change with extra err
                            err_no_cor_ex += 1
                    elif set(gen_slots) == set(tru_slots):
                        # success correction
                        err_su_cor += 1
                        if len(set(tru_slots) - set(err_slots_list)) > 0:
                            # success correction by adding to fix miss err
                            err_su_cor_add += 1
                        if len(set(err_slots_list) - set(tru_slots)) > 0:
                            # success correction by removing to fix extra err
                            err_su_cor_rem += 1
                    else:
                        # change but still wrong
                        err_fa_cor += 1
                        if len(set(gen_slots) - set(err_slots_list)) > 0:
                            # change but still wrong by adding
                            err_fa_cor_add += 1
                        if len(set(err_slots_list) - set(gen_slots)) > 0:
                            # change but still wrong by removing
                            err_fa_cor_rem += 1

        succ_corr = no_err_no_cor + err_su_cor
        fail_corr = no_err_fa_cor + err_no_cor + err_fa_cor
        total_turn= no_err + err

        stats = {
            "no_err":no_err,
            "no_err_no_cor":no_err_no_cor,
            "no_err_fa_cor":no_err_fa_cor,
            "err":err,
            "err_no_cor":err_no_cor,
            "err_su_cor":err_su_cor,
            "err_fa_cor":err_fa_cor,
            "succ_corr":succ_corr,
            "fail_corr":fail_corr,
            "total_turn":total_turn
        }

        print(f"no err turns:{no_err} with {no_err_no_cor} no correction and {no_err_fa_cor} ({no_err_fa_cor_add} add and {no_err_fa_cor_rem} remove) wrong corrections")
        print(f"err turns:{err} with {err_no_cor} ({err_no_cor_mi} miss and {err_no_cor_ex} extra) no correction \n\
                and {err_fa_cor} ({err_fa_cor_add} add and {err_fa_cor_rem} remove) wrong corrections and {err_su_cor} ({err_su_cor_add} add and {err_su_cor_rem} remove) success corrections")
                

        return stats

    def Count_Basic_Info(self, clean_result):
        total_gen_slot_num = 0
        total_tru_slot_num = 0
        total_turn_num = 0
        total_slot_type_num = defaultdict(int)
        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                total_turn_num += 1
                total_gen_slot_num += len(list(set(turn["generated_seq"])))
                total_tru_slot_num += len(list(set(turn["ground_truth"])))

                for slot in set(turn["ground_truth"]):
                    domain, slot_type, slot_val = slot.split("--")
                    # if domain not in total_slot_type_num:
                    #     total_slot_type_num[domain] = {}
                    # if slot_type not in total_slot_type_num[domain]:
                    #     total_slot_type_num[domain][slot_type] = 0
                    # total_slot_type_num[domain][slot_type] += 1
                    total_slot_type_num[domain+"--"+slot_type] += 1
        total_slot_type_num = OrderedDict(sorted(total_slot_type_num.items(), key=lambda t: t[0]))

        stats = {
            "total_turn_num" : total_turn_num,
            "total_gen_slot_num" : total_gen_slot_num,
            "total_tru_slot_num" : total_tru_slot_num,
            "total_slot_type_num": total_slot_type_num,
        }
        return stats

    def Count_Other(self, clean_result):
        stats = {
            "attraction": {"dom":defaultdict(int), "val":defaultdict(int), "other":defaultdict(int)}, 
            "hotel"     : {"dom":defaultdict(int), "val":defaultdict(int), "other":defaultdict(int)}, 
            "restaurant": {"dom":defaultdict(int), "val":defaultdict(int), "other":defaultdict(int)}, 
            "none" : {"miss":0, "extra":0},
            "train-leaveat" : defaultdict(int),
            "hotel-type" : defaultdict(int),
            "miss_refer" : {'refer': defaultdict(int),'hist': defaultdict(int),'not_in_text': defaultdict(int),},
            "extr_refer" : {'refer': defaultdict(int),'hist': defaultdict(int),'not_in_text': defaultdict(int),},
            "extra" : {
                "attraction": {"dom":defaultdict(int), "val":defaultdict(int), "other":defaultdict(int)}, 
                "hotel"     : {"dom":defaultdict(int), "val":defaultdict(int), "other":defaultdict(int)}, 
                "restaurant": {"dom":defaultdict(int), "val":defaultdict(int), "other":defaultdict(int)}, 
                "taxi": {"dom":defaultdict(int), "val":defaultdict(int), "other":defaultdict(int)}, 
                }
            }
        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                extra_predict = set(turn["generated_seq"]) - set(turn["ground_truth"])
                miss_predict  = set(turn["ground_truth"]) - set(turn["generated_seq"])

                # # # for slot_type name
                for miss_slot in miss_predict:
                    [dom, slot_type, slot_val] = miss_slot.split("--")
                    if slot_type == "name":
                        flag = 0
                        for extra_slot in extra_predict:
                            if extra_slot.split("--")[1] == "name":
                                flag = 1
                                if extra_slot.split("--")[0] == dom:
                                    stats[dom]["val"]["total_num"] += 1
                                    if slot_val == "none":
                                        stats[dom]["val"]["none"] += 1
                                    else:
                                        key_name = slot_val+"--"+extra_slot.split("--")[2]
                                        stats[dom]["val"][key_name] += 1
                                else:
                                    stats[dom]["dom"]["total_num"] += 1
                                    stats[dom]["dom"][extra_slot.split("--")[0]] += 1
                                break
                        if flag == 0:
                            if slot_val == "none":
                                stats[dom]["val"]["total_num"] += 1
                                stats[dom]["val"]["none"] += 1
                            else:
                                stats[dom]["other"]["total_num"] += 1


                                # referring
                                if slot_val in turn["context"]:
                                    utt_list = turn["context"].split("<")
                                    # utt_list.reverse()
                                    for utt in utt_list:
                                        if slot_val in utt:
                                            if utt.startswith("system"):
                                                stats[dom]["other"]["refer"] += 1
                                                # print(f"dialog id: {dial_id}, turn number: {tn}")
                                                # print("extra slot: " + ", ".join(extra_predict))
                                                # print("miss  slot: " + ", ".join(miss_predict))
                                                # print(utt)
                                                # pdb.set_trace()
                                            else:
                                                stats[dom]["other"]["history"] += 1

                                                # # stats[dom]["other"]["last"] += 1
                                                # print(f"dialog id: {dial_id}, turn number: {tn}")
                                                # print("extra slot: " + ", ".join(extra_predict))
                                                # print("miss  slot: " + ", ".join(miss_predict))
                                                # print(utt)
                                                # pdb.set_trace()
                                            break
                                else:
                                    stats[dom]["other"]["not_show_up"] += 1

                for extra_slot in extra_predict:
                    [dom, slot_type, slot_val] = extra_slot.split("--")
                    if slot_type == "name":
                        flag = 0
                        for miss_slot in miss_predict:
                            if miss_slot.split("--")[1] == "name":
                                flag = 1
                                if miss_slot.split("--")[0] == dom:
                                    stats["extra"][dom]["val"]["total_num"] += 1
                                    if slot_val == "none":
                                        stats["extra"][dom]["val"]["none"] += 1
                                    else:
                                        key_name = slot_val+"--"+miss_slot.split("--")[2]
                                        stats["extra"][dom]["val"][key_name] += 1
                                else:
                                    stats["extra"][dom]["dom"]["total_num"] += 1
                                    stats["extra"][dom]["dom"][miss_slot.split("--")[0]] += 1
                                break
                        if flag == 0:
                            if slot_val == "none":
                                stats["extra"][dom]["val"]["total_num"] += 1
                                stats["extra"][dom]["val"]["none"] += 1
                            else:
                                stats["extra"][dom]["other"]["total_num"] += 1

                                # referring
                                if slot_val in turn["context"]:
                                    utt_list = turn["context"].split("<")
                                    for utt in utt_list:
                                        if slot_val in utt:
                                            if utt.startswith("system"):
                                                stats["extra"][dom]["other"]["refer"] += 1
                                                # print(f"dialog id: {dial_id}, turn number: {tn}")
                                                # print("extra slot: " + ", ".join(extra_predict))
                                                # print("miss  slot: " + ", ".join(miss_predict))
                                                # print(turn["context"])
                                                # pdb.set_trace()
                                                

                                            else:
                                                stats["extra"][dom]["other"]["history"] += 1

                                            break
                                else:
                                    stats["extra"][dom]["other"]["not_show_up"] += 1

                # # # hotel-type
                for miss_slot in miss_predict:
                    [dom, slot_type, slot_val] = miss_slot.split("--")
                    if dom == "hotel" and slot_type == "type":
                        stats["hotel-type"][slot_val] += 1

                # # # train-leaveat
                for miss_slot in miss_predict:
                    [dom, slot_type, slot_val] = miss_slot.split("--")
                    if dom == "train" and slot_type == "leaveat":
                        if slot_val == "none":
                            stats["train-leaveat"]["none"] += 1
                        else:
                            flag = 0
                            for extra_slot in extra_predict:
                                if extra_slot.split("--")[0] == "train" and extra_slot.split("--")[1]:
                                    stats["train-leaveat"]["val"] += 1
                                    # key_name = slot_val+"--"+extra_slot.split("--")[2]
                                    # stats["train-leaveat"][key_name] += 1
                                    flag = 1
                            if flag == 0:
                                stats["train-leaveat"]["other"] += 1


                # # # referring
                # referring
                for miss_slot in miss_predict:
                    [dom, slot_type, slot_val] = miss_slot.split("--")
                    dom_type = dom+"--"+slot_type
                    if slot_val in turn["context"]:
                        utt_list = turn["context"].split("<")
                        # utt_list.reverse()
                        for utt in utt_list:
                            if slot_val in utt:
                                if utt.startswith("system"):
                                    # stats["miss_refer"]["refer"] += 1
                                    stats["miss_refer"]["refer"][dom_type] += 1
                                else:
                                    # stats["miss_refer"]["history"] += 1
                                    stats["miss_refer"]["hist"][dom_type] += 1
                                break
                        stats["miss_refer"]["not_in_text"][dom_type] += 1
                for extra_slot in extra_predict:
                    [dom, slot_type, slot_val] = extra_slot.split("--")
                    dom_type = dom+"--"+slot_type
                    if slot_val in turn["context"]:
                        utt_list = turn["context"].split("<")
                        # utt_list.reverse()
                        for utt in utt_list:
                            if slot_val in utt:
                                if utt.startswith("system"):
                                    # stats["extr_refer"]["refer"] += 1
                                    stats["extr_refer"]["refer"][dom_type] += 1
                                else:
                                    # stats["extr_refer"]["history"] += 1
                                    stats["extr_refer"]["hist"][dom_type] += 1
                                break
                    else:
                        stats["extr_refer"]["not_in_text"][dom_type] += 1


                
                # # # none value
                for miss_slot in miss_predict:
                    [dom, slot_type, slot_val] = miss_slot.split("--")
                    if slot_val == "none":
                        stats["none"]["miss"] += 1
                for extra_slot in extra_predict:
                    [dom, slot_type, slot_val] = extra_slot.split("--")
                    if slot_val == "none":
                        stats["none"]["extra"] += 1
                

        stats["miss_refer"] = OrderedDict(sorted(stats["miss_refer"].items(), key=lambda t: t[0]))
        stats["extr_refer"] = OrderedDict(sorted(stats["extr_refer"].items(), key=lambda t: t[0]))



        return stats

    def analyze(self, clean_result_path=None):
        """
        analyze results
        """
        # # # load cleaned results
        if clean_result_path is None:
            if self.result_clean is None:
                self.load_result_clean()
            clean_result = self.result_clean
        else:
            with open(clean_result_path) as df:
                clean_result = json.loads(df.read().lower())
            self.analyze_result_path = clean_result_path.replace("_clean.json", "_analyze.json")

        # self.load_result_err()
        # clean_result = self.result_err
        

        # # # basic info
        count_basic_info = self.Count_Basic_Info(clean_result)
        self.update_results(key_="count_basic_info", value_=count_basic_info)

        # # # turn num dist over err num per turn
        count_err_num = self.Count_Err_Num(clean_result)
        self.update_results(key_="count_err_num", value_=count_err_num)

        # # # err of domain vs. slot_type vs. slot_val
        count_err_type = self.Count_Err_Type(clean_result)
        self.update_results(key_="count_err_type", value_=count_err_type)

        # # # result if ignore do_not_care
        count_dontcare = self.Count_Dontcare(clean_result)
        self.update_results(key_="count_dontcare", value_=count_dontcare)

        # # # err correction ratio of succ & fail cases
        count_cor_ratio = self.Count_Correct_Ratio(clean_result)
        self.update_results(key_="count_cor_ratio", value_=count_cor_ratio)

        # # # other tmp ans special things
        count_other = self.Count_Other(clean_result)
        self.update_results(key_="count_other", value_=count_other)

    def update_results(self, key_=None, value_=None, dict_=None):
        if os.path.exists(self.analyze_result_path):
            with open(self.analyze_result_path, "r") as rf:
                results = json.loads(rf.read().lower())
        else:
            results = {}
        
        if dict_ is not None:
            results.update(dict_)
        else:
            results[key_] = value_

        with open(self.analyze_result_path, "w") as tf:
            json.dump(results, tf, indent=2)

def mean_and_dev(list):
    mean = sum(list) / float(len(list))
    var = sum([(i - mean)**2 for i in list]) / float(len(list))
    dev = math.sqrt(var)
    return mean, dev

def main():
    # path = "./experiment/cor_gpt2/model.result.jsonl"
    # path = "./experiment/gpt2_dst_nodict/result_test_bs1_2gpu.json"
    # path = "./experiment/gpt2_dst_owndict/result.jsonl"
    # path = "./experiment/gen_gpt2_nodict/result_decode_all_bs8.jsonl"
    # path = "./experiment/gen_gpt2_nodict/result_decode_all.jsonl"
    # path = "./experiment/gen_gpt2_nodict/result_test_ep8.jsonl"                       # 0.5567
    # path = "./experiment/gen_gpt2_nodict/result_test.jsonl"                       # 0.5567
    # path = "./experiment/gen_gpt2_nodict/result_valid.jsonl"                       # 0.5567
    # # path = "./experiment/cor_gpt2_nod_gen_errbs1_fomi/result_test_ep5.jsonl"    # 0.5570
    # # path = "./experiment/cor_gpt2_nod_gen_errbs1_foex/result_test_ep4.jsonl"    # 0.5542
    # path = "./experiment/cor_gpt2_nod_gen_errbs1/result_test.jsonl"             # 0.5553
    # # path = "./experiment/cor_gpt2_nod_fresh_errbs1/result_test.jsonl"           # 0.5557
    # # path = "./experiment/cor_gpt2_nod_gen4_errbs1/result_test_ep5.jsonl"        # 0.5566
    # path = "./experiment/cor_gpt2_nod_gen_errbs8/result_test_ep9.jsonl"         # 0.5604
    # path = "./experiment/cor_gpt2_nod_gen_errbs8_fomi/result_test_ep5.jsonl"    # 0.5589
    # path = "./experiment/cor_gpt2_nod_gen_errbs8_foex/result_test_ep2.jsonl"    # 0.5519
    # path = "./experiment/cor_gpt2_nod_gen_errbs8_re/result_test_ep4.jsonl"      # 0.5578
    # path = "./experiment/cor_gpt2_nod_gen_errbs8_re_8010/result_test_ep5.jsonl" # 0.5557
    # path = "./experiment/cor_gpt2_nod_fresh_randerr/result_test_ep11.jsonl"     # 0.5625
    # # path = "./experiment/cor_gpt2_nod_fresh_randerr/result_test_ep11_iter1.jsonl"     # 0.5603
    # # # path = "./experiment/cor_gpt2_nod_fresh_randerr_skipgen/result_test.jsonl"  # 0.5546
    # # path = "./experiment/cor_gpt2_nod_fresh_disterr_valacc/result_test.jsonl"  # 0.5606
    # # path = "./experiment/cor_gpt2_nod_fresh_disterr_valacc/result_test_iter1.jsonl"  # 0.5608
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_valacc/result_test.jsonl"  # 0.5637
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_valacc_sd0/result_test_bs1.jsonl"     # 0.5590
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_valacc_sd1/result_test_bs1.jsonl"     # 0.5579
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_valacc_sd2/result_test_bs1.jsonl"     # 0.5444
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_valacc/result_test_iter1.jsonl"  # 0.5637
    # path = "./experiment/cor_gpt2_nod_fresh_disterr/result_test_ep16.jsonl"        # 0.5622
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_num3/result_test_ep9.jsonl"  # 0.5612
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_num3_wi10/result_test_ep16_bs1.jsonl"     #55.93
    # path = "./experiment/cor_gpt2_nod_fresh_randerr3003/result_test_ep10_bs1.jsonl"  # 0.5491
    # path = "./experiment/cor_gpt2_nod_fresh_randerr3030/result_test_ep15_bs1.jsonl"  # 0.5565
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_03030/result_test_ep9_bs1.jsonl"  # 0.5555
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_0301313/result_test_ep10.jsonl"     # 0.5423
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_valacc_sd0_sam/result_test.jsonl"     # 0.5590
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_valacc_sd0_dtv/result_test.jsonl"     # 0.5560
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_new/result_test.jsonl"     # 0.5617
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_new_sd0/result_test.jsonl"     # 0.5540
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_new_sd1/result_test.jsonl"     # 0.5595
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_new_sd2/result_test.jsonl"     # 0.5557
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_new/result_test.jsonl"     # 0.5617
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_new03344/result_test.jsonl"     # 0.5617
    # path = "./experiment/cor_gpt2_nod_fresh_randerr_new_comma/result_test.jsonl"     # 0.5548

    # path = "./experiment/cor_gpt2_nod_fresh_randerr_inf3133/result_test_ep11_bs1.jsonl"     # 0.5492
    # path = "./experiment/gen_gpt2_nod_val/result_test_ep5.jsonl"     # 0.7861
    # path = "./experiment/gen_gpt2_nod_type/result_test_ep5.jsonl"     # 0.5251
    # path = "./experiment/gen_gpt2_nod_val/result_test_ep5_type5.jsonl"     # 0.9529
    # path = "./experiment/gen_gpt2_nod_nldst/result_test.jsonl"     # 0.5316
    # path = "./experiment/gen_gpt2_nod_nldst_tem2/result_test.jsonl"     # 0.5469
    # path = "./experiment/gen_gpt2_nod_val_alltype_seq/result_test.jsonl"     # 0.3268
    # path = "./experiment/gen_gpt2_nod_val_alltype_sep/result_test.jsonl"     # 0.2185

    # path = "./experiment/gen_gpt2_nod_type_cor/result_test.jsonl"     # 0.5311
    # path = "./experiment/gen_gpt2_nod_type_valacc/result_test.jsonl"     # 0.5374

    # path = "./experiment/cor_gpt2_nod_fresh_kpmiss/result_test.jsonl"     # 0.5545
    # path = "./experiment/cor_gpt2_nod_fresh_kpextra/result_test.jsonl"     # 0.5574
    # path = "./experiment/cor_gpt2_nod_fresh_kperr/result_test.jsonl"     # 0.5565
    # path = "./experiment/cor_gpt2_nod_fresh_fom_kpe/result_test.jsonl"     # 0.5582
    # path = "./experiment/cor_gpt2_nod_fresh_rand22/result_test_bs1.jsonl"     # 0.3574
    # path = "./experiment/cor_gpt2_nod_fresh_rand1101/result_test.jsonl"     # 0.5394


    # path = "./experiment/gen_gpt2_nodict_aug_sd0/result_test.jsonl"                       # 
    # path = "./experiment/gen_gpt2_nodict_aug_sd1/result_test.jsonl"                       # 
    # path = "./experiment/gen_gpt2_nodict_aug_sd2/result_test.jsonl"                       # 
    # path = "./experiment/gen_gpt2_nodict_aug_all_sd2/result_test.jsonl"                       # 
    # path = "./experiment/gen_gpt2_nodict_aug_all_sd1/result_test.jsonl"                       # 
    # path = "./experiment/gen_gpt2_nodict_aug_all_sd0/result_test.jsonl"                       # 
    # # path = "./experiment/gen_gpt2_filtername/result_test.jsonl"                       # 0.4559 / 0.5524

    # path = "./experiment/gen_gpt2_nodict_scr_sd0/result_test.jsonl" 
    # path = "./experiment/gen_gpt2_nodict_scr_all_sd0/result_test.jsonl"                       # 
    
    # path = "./experiment/gen_gpt2_nodict/result_test.jsonl"                       # 0.5567 / 0.9707 / 0.5632 /0.9717
    # path = "./experiment/gen_gpt2_nodict_sd0/result_test.jsonl"                       # 0.5459 / 0.9693 / 0.5506 / 0.9701
    # path = "./experiment/gen_gpt2_nodict_sd1/result_test.jsonl"                       # 0.5366 / 0.9690 / 0.5412 / 0.9696
    # path = "./experiment/gen_gpt2_nodict_sd2/result_test.jsonl"                       # 0.5378 / 0.9680 / 0.5422 / 0.9688

    path = "./experiment/gen_gpt2_nodict_m22_sd0/result_test.jsonl"                       # 0.5530 / 0.9702 / 0.5649 / 0.9718
    path = "./experiment/gen_gpt2_nodict_m22_sd1/result_test.jsonl"                       # 0.5447 / 0.9702 / 0.5504 / 0.9692
    path = "./experiment/gen_gpt2_nodict_m22_sd2/result_test.jsonl"                       # 0.5103 / 0.9663 / 0.5162 / 0.9669

    # path = "./experiment/gen_gpt2_m23_sd0/result_test.jsonl"                       # 0.6207 0.6474
    # path = "./experiment/gen_gpt2_m23_sd3/result_test.jsonl"                       # 0.6224  /0.6521
    # path = "./experiment/gen_gpt2_m23_sd2/result_test.jsonl"                       # 0.6149 / 0.6448


    clean_analyze_result = Clean_Analyze_result(path_result=path)

    truth_path = "./data/multiwoz_dst/MULTIWOZ2.3/modify_data_reformat_test.json"
    clean_analyze_result.clean(save_clean_result=True, save_err=True, replace_truth=False, truth_path=truth_path)
    # clean_analyze_result.clean_err(save_clean_result=False)
    # clean_analyze_result.clean_all_type_seq(save_clean_result=False)
    # clean_analyze_result.clean_all_type_sep(save_clean_result=False)

    clean_analyze_result.compute_metrics()
    # clean_analyze_result.analyze()

    # mean, dev = mean_and_dev([56.17, 55.40, 55.95])
    # mean, dev = mean_and_dev([55.67, 51.62, 49.92])
    # print(f"mean:{mean}, dev:{dev}")
    # clean_analyze_result.analyze(clean_result_path="finetune_gpt2/results/best_accm_noend_len100_all_clean.json")

if __name__ == "__main__":
    main()