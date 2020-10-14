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
            self.load_raw_result()
            raw_result = self.result

        self.result_clean = {}
        for dial in raw_result:
            dial_id = dial['dial_id']
            turn_num = dial['turn_num']
            context = dial['context']
            gt_slots = dial['slots_inf']
            gen_slots = dial['slots_err']

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
                # # # joint goal acc
                if set(turn["generated_seq"]) == set(turn["ground_truth"]):
                    correct_turn += 1
                elif save_err:
                    if dial_id not in self.err:
                        self.err[dial_id] = {}
                    self.err[dial_id][turn_num] = turn
                # # # slots acc
                slots_acc += self.compute_slot_acc_turn(turn["ground_truth"], turn["generated_seq"])
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
        extra_tn, miss_tn, err_tn, total_tn = 0, 0, 0, 0
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
                if set(turn["generated_seq"]) != set(turn["ground_truth"]):
                    err_tn += 1
                total_tn += 1

        stats["total_turn"] = sum(list(stats["extra"].values()))
        stats["extra"]["total_num"] = sum([stats["extra"][i] * int(i) for i in stats["extra"]])
        stats["miss"]["total_num"] = sum([stats["miss"][i] * int(i) for i in stats["miss"]])
        stats["err"]["total_num"] = sum([stats["err"][i] * int(i) for i in stats["err"]])
        print(f"total turns {total_tn} err turns {err_tn}, miss:{miss_tn}, extra:{extra_tn}")
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
                    "domain":0, 
                    "slot_type":0, 
                    "slot_val":0, 
                    "extra_slot_num":0,
                    "match_slot_num":0,
                    "miss_slot_num":0,
                    },
                "match" : {
                    "domain":{},  # dom1-dom2-slot_type
                    "slot_type":{},  # dom-slot_type1-slot_typ2
                    "slot_val":{}
                    },
                "extra" : {},
                "miss"  : {}
        }
        for dial_id, dial in clean_result.items():
            for tn, turn in dial.items():
                extra_predict = set(turn["generated_seq"]) - set(turn["ground_truth"])
                miss_predict = set(turn["ground_truth"]) - set(turn["generated_seq"])
                extra_predict_split = [slot.split("--") for slot in extra_predict]
                miss_predict_split = [slot.split("--") for slot in miss_predict]
                stats["overall"]["extra_slot_num"] += len(extra_predict)
                stats["overall"]["miss_slot_num"] += len(miss_predict)
                if len(extra_predict) == 0 or len(miss_predict) == 0:
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
                        if extra_slot[0] not in stats["miss"]:
                            stats["miss"][extra_slot[0]] = {"total" : 0, extra_slot[1] : 0}
                        elif extra_slot[1] not in stats["miss"][extra_slot[0]]:
                            stats["miss"][extra_slot[0]][extra_slot[1]] = 0
                        stats["miss"][extra_slot[0]][extra_slot[1]] += 1
                        stats["miss"][extra_slot[0]]["total"] += 1


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
        

        # # # basic info
        count_basic_info = self.Count_Basic_Info(clean_result)
        # self.update_results(key_="count_basic_info", value_=count_basic_info)

        # # # turn num dist over err num per turn
        count_err_num = self.Count_Err_Num(clean_result)
        # self.update_results(key_="count_err_num", value_=count_err_num)

        # # # err of domain vs. slot_type vs. slot_val
        # count_err_type = self.Count_Err_Type(clean_result)
        # self.update_results(key_="count_err_type", value_=count_err_type)

        # # # result if ignore do_not_care
        # count_dontcare = self.Count_Dontcare(clean_result)
        # self.update_results(key_="count_dontcare", value_=count_dontcare)

        # # # # err correction ratio of succ & fail cases
        # count_cor_ratio = self.Count_Correct_Ratio(clean_result)
        # self.update_results(key_="count_cor_ratio", value_=count_cor_ratio)

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
    path = "./experiment/tmp/created_errs.json"
    clean_analyze_result = Clean_Analyze_result(path_result=path)

    clean_analyze_result.clean(save_clean_result=False)
    # joint_goal_acc, slots_acc=clean_analyze_result.compute_metrics(save_err=False)
    # print(joint_goal_acc, slots_acc)
    clean_analyze_result.analyze()


if __name__ == "__main__":
    main()