#!/usr/bin/env python3
#
import os, sys, json
import math
import pdb
from collections import defaultdict

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

# slot_types = ['book time', 'leaveat', 'name', 'internet', 'book stay', 
#              'pricerange', 'arriveby', 'area', 'destination', 'day', 
#              'food', 'departure', 'book day', 'book people', 'department', 
#              'stars', 'parking', 'type']

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

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
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
                    slots_list.append(domain+"--"+slot_type+"--"+slot_val)
        return slots_list

    def clean(self, raw_result=None, save_clean_result=False, save_clean_path=None, accm=0, replace_truth=0, order='dtv'):
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
            # data_path = '/checkpoint/kunqian/multiwoz/data/MultiWOZ_2.1/data_reformat_trade_slot_accm_hist_accm.json'
            with open(data_path) as df:
                ori_data = json.loads(df.read().lower())

        self.result_clean = {}
        for dial in raw_result:
            if "dial_id" not in dial["dialog"][0][0]:
                continue
            dial_id = dial["dialog"][0][0]['dial_id']
            turn_num = dial["dialog"][0][0]['turn_num']
            context = dial["dialog"][0][0]['text']
            gt_slots = dial["dialog"][0][0]['eval_labels'][0]
            gen_slots = dial["dialog"][0][1]['text']

            if dial_id not in self.result_clean:
                self.result_clean[dial_id] = {}
            if turn_num not in self.result_clean[dial_id]:
                self.result_clean[dial_id][turn_num] = {}

            # # # reformat ground truth
            slots_truth = []
                    
            oracle = gt_slots.split(",")
            if oracle[-1] == "":
                oracle = oracle[:-1]
            oracle = [slot.strip() for slot in oracle]

            for slot_ in oracle:
                slot = slot_.split()
                if order == 'dtv':
                    if len(slot) > 2 and slot[0] in domains:
                        domain = slot[0]
                        if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                            slot_type = slot[1]+" "+slot[2]
                            slot_val  = " ".join(slot[3:])
                        else:
                            slot_type = slot[1]
                            slot_val  = " ".join(slot[2:])
                        if not slot_val == 'dontcare':
                            slots_truth.append(domain+"--"+slot_type+"--"+slot_val)
                        # slots_truth.append(domain+"--"+slot_type+"--"+slot_val)
                    # else:
                    #     # pmul4204.json turn 5,6 slot: "15" from train set
                    #     # pdb.set_trace()

            if dial_id not in self.result_clean:
                pdb.set_trace()
            self.result_clean[dial_id][turn_num]["ground_truth"] = sorted(list(set(slots_truth)))

            # # # reformat generated slots
            slots_pred = []

            predic = gen_slots.split(",")
            if predic[-1] == "":
                predic = predic[:-1]
            predic = [slot.strip() for slot in predic]

            for slot_ in predic:
                slot = slot_.split()
                if order == 'dtv':
                    if len(slot) > 2 and slot[0] in domains:
                        domain = slot[0]
                        if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                            slot_type = slot[1]+" "+slot[2]
                            slot_val  = " ".join(slot[3:])
                        else:
                            slot_type = slot[1]
                            slot_val  = " ".join(slot[2:])
                        if not slot_val == 'dontcare':
                            slots_pred.append(domain+"--"+slot_type+"--"+slot_val)
                        # slots_pred.append(domain+"--"+slot_type+"--"+slot_val)
                    # else:
                    #     pdb.set_trace()

            self.result_clean[dial_id][turn_num]["generated_seq"] = sorted(list(set(slots_pred)))

            self.result_clean[dial_id][turn_num]["tru_seq"] = ", ".join(sorted(list(set(slots_truth))))
            self.result_clean[dial_id][turn_num]["gen_seq"] = ", ".join(sorted(list(set(slots_pred))))

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

    def compute_metrics(self, clean_result=None, save_err=False):
        """
        compute joint_goal_acc and slots_acc
        """
        # # # load cleaned results
        if clean_result is None:
            if self.result_clean is None:
                self.load_result_clean()
            clean_result = self.result_clean

        self.err = {}
        
        total_turn = 0
        correct_turn = 0
        slots_acc = 0
        for dial_id, dial in clean_result.items():
            for turn_num, turn in dial.items():
                total_turn += 1
                # # # slots acc
                slots_acc += self.compute_slot_acc_turn(turn["ground_truth"], turn["generated_seq"])
                # # # joint goal acc
                if set(turn["generated_seq"]) == set(turn["ground_truth"]):
                    correct_turn += 1
                elif save_err:
                    if (len(set(turn["generated_seq"]) - set(turn["ground_truth"])) > 1
                        or len(set(turn["ground_truth"]) - set(turn["generated_seq"])) > 1):
                        continue
                    if dial_id not in self.err:
                        self.err[dial_id] = {}
                    self.err[dial_id][turn_num] = turn
        if save_err:
            with open(self.err_result_path, "w") as ef:
                json.dump(self.err, ef, indent=2)
        joint_goal_acc = correct_turn / float(total_turn)
        slots_acc /= float(total_turn)
        return joint_goal_acc, slots_acc
        
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
                    "slot_val"  : {},
                    "slot_num"  : {},
                    },
                "extra" : {"slot_num":{}},
                "miss"  : {"slot_num":{}},
        }
        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                extra_predict = set(turn["generated_seq"]) - set(turn["ground_truth"])
                miss_predict = set(turn["ground_truth"]) - set(turn["generated_seq"])
                extra_predict_split = [slot.split("--") for slot in extra_predict]
                miss_predict_split = [slot.split("--") for slot in miss_predict]
                stats["overall"]["extra_slot_num"] += len(extra_predict)
                stats["overall"]["miss_slot_num"] += len(miss_predict)

                if len(turn["ground_truth"]) not in stats["extra"]["slot_num"]:
                    stats["extra"]["slot_num"][len(turn["ground_truth"])] = defaultdict(int)
                stats["extra"]["slot_num"][len(turn["ground_truth"])][len(extra_predict)] += 1
                if len(turn["ground_truth"]) not in stats["miss"]["slot_num"]:
                    stats["miss"]["slot_num"][len(turn["ground_truth"])] = defaultdict(int)
                stats["miss"]["slot_num"][len(turn["ground_truth"])][len(miss_predict)] += 1
                if len(turn["ground_truth"]) not in stats["match"]["slot_num"]:
                    stats["match"]["slot_num"][len(turn["ground_truth"])] = defaultdict(int)
                if len(extra_predict) == 0 or len(miss_predict) == 0:
                    stats["match"]["slot_num"][len(turn["ground_truth"])]["0"] += 1
                    continue

                matched_slot = []
                for extra_slot in extra_predict_split:
                    match_flag = 0
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
                        # # # for unmatched extra slots
                        if extra_slot[0] not in stats["extra"]:
                            stats["extra"][extra_slot[0]] = {"total" : 0, extra_slot[1] : 0}
                        elif extra_slot[1] not in stats["extra"][extra_slot[0]]:
                            stats["extra"][extra_slot[0]][extra_slot[1]] = 0
                        stats["extra"][extra_slot[0]][extra_slot[1]] += 1
                        stats["extra"][extra_slot[0]]["total"] += 1

                        # saving
                        extra_slot_turn["other"].append({
                                "dial_id": dial_id,
                                "turn_num": tn,
                                "extr_slot" : "--".join(extra_slot),
                                "gen_slot"  : ", ".join(sorted(turn["generated_seq"])),
                                "tru_slot"  : ", ".join(sorted(turn["ground_truth"]))
                        })

                # # # for unmatched miss slots
                for miss_slot in miss_predict_split:
                    if miss_slot not in matched_slot:
                        if miss_slot[0] not in stats["miss"]:
                            stats["miss"][miss_slot[0]] = {"total" : 0, miss_slot[1] : 0}
                        elif miss_slot[1] not in stats["miss"][miss_slot[0]]:
                            stats["miss"][miss_slot[0]][miss_slot[1]] = 0
                        stats["miss"][miss_slot[0]][miss_slot[1]] += 1
                        stats["miss"][miss_slot[0]]["total"] += 1

                stats["match"]["slot_num"][len(turn["ground_truth"])][len(matched_slot)] += 1

        # with open(extra_slot_path, 'w') as tf:
        #     json.dump(extra_slot_turn, tf, indent = 2)
                                
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
        with open(gen_data_path) as df:
            origin_data = json.loads(df.read().lower())
        
        no_err  = 0
        no_err_no_cor = 0
        no_err_fa_cor = 0
        err     = 0
        err_no_cor = 0
        err_fa_cor = 0
        err_su_cor = 0

        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                if dial_id+"-"+str(tn) not in origin_data:
                    continue
                err_slots_str = origin_data[dial_id+"-"+str(tn)]["slots_err"]
                err_slots_list = self._extract_slot_from_string(err_slots_str)
                
                tru_slots = turn["ground_truth"]
                gen_slots = turn["generated_seq"]

                if set(err_slots_list) == set(tru_slots):
                    no_err += 1
                    if set(tru_slots) == set(gen_slots):
                        no_err_no_cor += 1
                    else:
                        no_err_fa_cor += 1
                else:
                    err += 1
                    if set(gen_slots) == set(err_slots_list):
                        err_no_cor += 1
                    elif set(gen_slots) == set(tru_slots):
                        err_su_cor += 1
                    else:
                        err_fa_cor += 1

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

        print("no err turns:{} with {} no correction and {} wrong corrections".format(no_err, no_err_no_cor, no_err_fa_cor))
        print("err turns:{} with {} no correction and {} wrong corrections and {} success corrections".format(err, err_no_cor, err_fa_cor,err_su_cor))
                

        return stats

    def Count_Basic_Info(self, clean_result):
        total_gen_slot_num = 0
        total_tru_slot_num = 0
        total_turn_num = 0
        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                total_turn_num += 1
                total_gen_slot_num += len(set(set(turn["generated_seq"])))
                total_tru_slot_num += len(set(set(turn["ground_truth"])))
        
        stats = {
            "total_turn_num" : total_turn_num,
            "total_gen_slot_num" : total_gen_slot_num,
            "total_tru_slot_num" : total_tru_slot_num,
        }
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


def main():
    path = "./experiment/cor_gpt2/model.result.jsonl"
    path = "./experiment/gpt2_dst_nodict/result_test_bs1_2gpu.json"
    path = "./experiment/gpt2_dst_owndict/result.jsonl"
    path = "./experiment/gen_gpt2_nodict/result_decode_all_bs8.jsonl"
    path = "./experiment/gen_gpt2_nodict/result_decode_all.jsonl"
    path = "./experiment/gen_gpt2_nodict/result_test.jsonl"                       # 0.5567
    # path = "./experiment/cor_gpt2_nod_gen_errbs1_fomi/result_test_ep5.jsonl"    # 0.5570
    # path = "./experiment/cor_gpt2_nod_gen_errbs1_foex/result_test_ep4.jsonl"    # 0.5542
    # path = "./experiment/cor_gpt2_nod_gen_errbs1/result_test.jsonl"             # 0.5553
    # path = "./experiment/cor_gpt2_nod_fresh_errbs1/result_test.jsonl"           # 0.5557
    # path = "./experiment/cor_gpt2_nod_gen4_errbs1/result_test_ep5.jsonl"        # 0.5566
    # path = "./experiment/cor_gpt2_nod_gen_errbs8/result_test_ep9.jsonl"         # 0.5604
    # # # path = "./experiment/old_data/cor_gpt2_re/result_test.jsonl"
    # path = "./experiment/cor_gpt2_nod_gen_errbs8_fomi/result_test_ep5.jsonl"    # 0.5589
    # path = "./experiment/cor_gpt2_nod_gen_errbs8_foex/result_test_ep2.jsonl"    # 0.5519
    # path = "./experiment/cor_gpt2_nod_gen_errbs8_re/result_test_ep4.jsonl"      # 0.5578
    # path = "./experiment/cor_gpt2_nod_gen_errbs8_re_8010/result_test_ep5.jsonl" # 0.5557
    
    clean_analyze_result = Clean_Analyze_result(path_result=path)

    clean_analyze_result.clean(save_clean_result=False, accm=0, replace_truth=0, order='dtv')
    joint_goal_acc, slots_acc=clean_analyze_result.compute_metrics(save_err=False)
    print(joint_goal_acc, slots_acc)
    clean_analyze_result.analyze()

    # clean_analyze_result.analyze(clean_result_path="finetune_gpt2/results/best_accm_noend_len100_all_clean.json")

if __name__ == "__main__":
    main()