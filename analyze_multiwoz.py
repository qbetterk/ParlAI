#!/usr/bin/env python3
#
import os
import json, random, math, re
import pdb
from collections import defaultdict, OrderedDict
from fuzzywuzzy import fuzz
from tqdm import tqdm

DOMAINS = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]

#  # # # for trade data
SLOT_TYPES = [
    "day",
    "type",
    "area",
    "stars",
    "department",
    "food",
    "name",
    'internet',
    'parking',
    'book stay',
    'book people',
    'book time',
    'book day',
    'pricerange',
    'destination',
    'leaveat',
    'arriveby',
    'departure',
]

SLOT_TYPES_CAT = [
    'internet',
    'parking',
]

random.seed(0)


class Analyze_Multiwoz(object):
    def __init__(self, data_path=None):
        # self.data_path = (
        #     "./data/multiwoz_dst/MULTIWOZ2.1/data_reformat_trade_turn_sa_ha_test.json"
        # )
        self.data_path = (
            "./data/multiwoz_dst/MULTIWOZ2.2/data_reformat_test.json"
        )
        self.analyze_result_path = (
            ".".join(self.data_path.split(".")[:-1]) + "_analyze.json"
        )
        self.otgy_path = "./data/multiwoz_dst/MULTIWOZ2.2/otgy.json"

        self.analyze_bias_path = "./data/multiwoz_dst/MULTIWOZ2.2/bias_stats.json"

        self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + "_tmp.json"
        self.tmp_log = {}

    def _load_txt(self, file_path):
        with open(file_path) as df:
            data = df.read().lower().split("\n")
            data.remove('')
        return data

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def _load_data(self):
        # # # load cleaned results
        with open(self.data_path) as df:
            self.data = json.loads(df.read().lower())

        for _, turn in self.data.items():
            turn["slots"] = self._extract_slot_from_string(turn["slots_inf"])
            del turn["slots_err"], turn["slots_inf"]

    def _save_tmp_log(self):
        with open(self.tmp_log_path, "w") as tf:
            json.dump(self.tmp_log, tf, indent=2)

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format:
        "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
        and this function would reformat the string into list:
        ["dom--slot_type--slot_val", ... ]
        """
        slots_list = []

        # # # split according to ","
        str_split = slots_string.split(",")
        if str_split[-1] == "":
            str_split = str_split[:-1]
        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in DOMAINS:
                domain = slot[0]
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1] + " " + slot[2]
                    slot_val = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val = " ".join(slot[2:])
                if not slot_val == 'dontcare':
                    # slots_list.append(domain+"--"+slot_type+"--"+slot_val)
                    slots_list.append([domain, slot_type, slot_val])
        return slots_list

    def Count_Basic_Info(self):
        total_turn_num = 0
        total_slot_num = 0
        total_slot_num_dom = defaultdict(int)  # num of slot over domain
        total_slot_num_type = defaultdict(int)  # num of slot over domain_type == num of turn over domain_type
        total_turn_num_dom = defaultdict(int)  # num of slot over domain
        total_dial_num_dom = defaultdict(int) # num of dialog over domain
        total_dial_idx_dom = defaultdict(set) # idx set of dialog over domain
        for _, turn in self.data.items():
            total_turn_num += 1
            turn["domains"] = set()
            for slot in turn["slots"]:
                domain = slot[0]
                slot_type = slot[1]

                total_slot_num += 1
                total_slot_num_dom[domain] += 1
                total_slot_num_type[domain + '_' + slot_type] += 1
                turn["domains"].add(domain)
            turn["domains"] = list(turn["domains"])
            for domain in turn["domains"]:
                total_turn_num_dom[domain] += 1
                total_dial_idx_dom[domain].add(turn["dial_id"])
                total_dial_idx_dom["total"].add(turn["dial_id"])
        total_slot_num_dom = OrderedDict(
            sorted(total_slot_num_dom.items(), key=lambda t: t[0])
        )
        total_turn_num_dom = OrderedDict(
            sorted(total_turn_num_dom.items(), key=lambda t: t[0])
        )
        total_slot_num_type = OrderedDict(
            sorted(total_slot_num_type.items(), key=lambda t: t[0])
        )
        total_dial_idx_dom = OrderedDict(
            sorted(total_dial_idx_dom.items(), key=lambda t: t[0])
        )
        for dom in total_dial_idx_dom:
            total_dial_num_dom[dom] = len(total_dial_idx_dom[dom])

        stats = {
            "total_turn_num": total_turn_num,
            "total_slot_num": total_slot_num,
            "total_slot_num_dom": total_slot_num_dom,
            "total_turn_num_dom": total_turn_num_dom,
            "total_slot_num_type": total_slot_num_type,
            "total_dial_num_dom":total_dial_num_dom,
        }
        return stats

    def Count_Refer_Info(self):
        slot_source_num = {
            "usr": defaultdict(int),  # user mentioned first
            "sys": defaultdict(int),  # system mentioned first (refer)
            "other": defaultdict(int),  # not directly shown in utt (imply)
            "none": defaultdict(int),  # not directly shown in utt (imply)
        }
        for idx, turn in self.data.items():
            for slot in turn["slots"]:
                [domain, slot_type, slot_val] = slot
                if slot_val == "none":
                    slot_source_num["none"][f"{domain}_{slot_type}"] += 1

                # labeled
                last_turn = turn["context"].split("<system>")[-1]
                sys_utt, user_utt = last_turn.split("<user>")
                slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])

                if domain == "hotel" and slot_type == "type":
                    # pdb.set_trace()
                    if slot_val in user_utt: # and slot_val in sys_utt:
                        if idx not in self.tmp_log:
                            self.tmp_log[idx] = turn.copy()
                            del self.tmp_log[idx]["domains"]
                            self.tmp_log[idx]["context"] = last_turn

                            self.tmp_log[idx]["slots"] = ", ".join([" ".join(slot) for slot in self.tmp_log[idx]["slots"]])
                        
                            self.tmp_log[idx]["add_slot"] = f"{domain} {slot_type} {slot_val}"
                        elif value not in self.tmp_log[idx]["add_slot"]:
                            self.tmp_log[idx]["add_slot"] += f" | {slot_val}"

                # referring
                elif slot_val in turn["context"]:
                    utt_list = turn["context"].split("<")
                    for utt in utt_list:
                        if slot_val in utt:
                            if utt.startswith("system"):
                                slot_source_num["sys"][f"{domain}_{slot_type}"] += 1
                                # if domain == "attraction" \
                                #    and slot_type == "area" \
                                #    and "attraction name" in " ".join([" ".join(slot) for slot in turn["slots"]]) \
                                #    and len(self.tmp_log) < 1000:

                                #     self.tmp_log[idx] = turn.copy()
                                #     self.tmp_log[idx]["slots"] = f"{domain} {slot_type} {slot_val}"
                            else:
                                slot_source_num["usr"][f"{domain}_{slot_type}"] += 1
                            break
                else:
                    slot_source_num["other"][f"{domain}_{slot_type}"] += 1


        for key_ in slot_source_num:
            slot_source_num[key_] = OrderedDict(
                sorted(slot_source_num[key_].items(), key=lambda t: t[0])
            )

        # self.tmp_log = OrderedDict(
        #     sorted(self.tmp_log.items(), key=lambda t: t[0])
        # )

        stats = {
            "slot_source_num": slot_source_num,
        }

        return stats

    # manual
    def Count_Name_Info_old(self):

        wi_name_set = {
            "attraction": set(),
            "hotel": set(),
            "restaurant": set(),
            "train" : set()
        }
        wo_name_set = {
            "attraction": set(),
            "hotel": set(),
            "restaurant": set(),
            "train" : set()
        }
        tmp = {
            "attraction": set(),
            "hotel": set(),
            "restaurant": set(),
            "train" : set()
        }

        domain = "train"
        slot_type = "leaveat"
        for idx, turn in self.data.items():
            slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
            for dom in [domain]:
            # for domain in ["attraction", "hotel", "restaurant"]:
                if dom in turn["domains"]:
                    if dom+" "+slot_type in slots_str:
                        wi_name_set[dom].add(turn["dial_id"])
                    else:
                        wo_name_set[dom].add(turn["dial_id"])
        tmp_union = set()
        for dom in tmp:
            tmp[dom] = wo_name_set[dom] - wi_name_set[dom]
            # tmp[dom] = wi_name_set[dom]
            tmp_union = tmp_union.union(tmp[dom])
            # print(len())
        # pdb.set_trace()


        for dial_id in tmp_union:
            for turn_num in range(20, 0, -1):
                idx = dial_id + "-" + str(turn_num)
                # pdb.set_trace()
                if idx in self.data:
                    turn = self.data[idx]
                else:
                    continue
                if len(self.tmp_log) < 200:
                    self.tmp_log[idx] = turn.copy()
                    del self.tmp_log[idx]["domains"]
                    for slot in self.tmp_log[idx]["slots"]:
                        if slot[0] == domain and slot[1] == slot_type:
                            slot_val = slot[2]
                            break

                    self.tmp_log[idx]["slots"] = ", ".join([" ".join(slot) for slot in self.tmp_log[idx]["slots"]])
                    self.tmp_log[idx]["add_slot"] = f"{domain} {slot_type} "
                    # self.tmp_log[idx]["slots"] = f"train leave at {slot_val}"
                break

    def Create_OTGY_M22(self):
        self.otgy = {}
        for _, turn in self.data.items():
            for slot in turn["slots"]:
                [domain, slot_type, slot_val] = slot
                dom_type = f"{domain}--{slot_type}"
                if dom_type not in self.otgy:
                    self.otgy[dom_type] = defaultdict(int)
                self.otgy[dom_type][slot_val] += 1

        self.otgy = OrderedDict(
            sorted(self.otgy.items(), key=lambda t: t[0])
        )
        for dom_type in self.otgy:
            self.otgy[dom_type] = OrderedDict(
            sorted(self.otgy[dom_type].items(), key=lambda t: t[0], reverse=False)
        )

        with open(self.otgy_path, "w") as tf:
            json.dump(self.otgy, tf, indent=2)

    def Analyze_Bias(self):

        self._load_otgy()
        stats_dict = {
            "max/min": {},
            "first_portion" : {},
            "dev": {},
            "se": {},
            "entropy":{},
            "H1_H0":{},
            "Hinf_H0":{},
        }
        
        for dom_type, cn_dict in self.otgy.items():
            if "unknown" in cn_dict:
                del cn_dict["unknown"]
            cn_list = list(cn_dict.values())   # count number list over slot values
            max_num = max(cn_list)
            min_num = min(cn_list) if min(cn_list)!=0 else 1
            ratio = max_num / min_num

            stats_dict["max/min"][dom_type] = ratio
            stats_dict["first_portion"][dom_type] = max_num / sum(cn_list)

            # standard deviation and standard error
            avg = sum(cn_list) / len(cn_list)
            var = sum([(i-avg) ** 2 for i in cn_list]) / len(cn_list)
            sdv = math.sqrt(var)
            se = sdv / math.sqrt(len(cn_list))
            stats_dict["dev"][dom_type] = sdv
            stats_dict["se"][dom_type] = se

            # entropy
            freq_list = [cn/sum(cn_list) for cn in cn_list]
            # H1
            entropy = -1 * sum([freq * math.log(freq) for freq in freq_list])
            stats_dict["entropy"][dom_type] = entropy
            # H0
            H0 = math.log(len(freq_list))
            H1_H0 = entropy / H0
            stats_dict["H1_H0"][dom_type] = H1_H0
            # Hinf
            Hinf = -1 * math.log(max(freq_list))
            Hinf_H0 = Hinf / H0
            stats_dict["Hinf_H0"][dom_type] = Hinf_H0


        # print(ratio_dict)

        with open(self.analyze_bias_path, "w") as tf:
            json.dump(stats_dict, tf, indent=2)

    def Search_Name_Entity(self):
        ontology_path = './data/multiwoz_dst/MULTIWOZ2.2/otgy.json'
        with open(ontology_path) as df:
            self.otgy = json.loads(df.read().lower())
        target_domain = 'hotel'
        target_name_list = set(self.otgy[target_domain+"--name"].keys())

        for idx, turn in self.data.items():
            if target_domain in turn['domains']:
                for target_name in target_name_list:
                    if target_name in turn['context']:
                        if [target_domain, "name", target_name] not in turn["slots"]:
                            if len(self.tmp_log) < 100:
                                idx_ = f"{idx}-{target_name}"
                                self.tmp_log[idx_] = turn.copy()
                                self.tmp_log[idx_]["slots"] = f"{target_domain} name {target_name}"
                                del self.tmp_log[idx_]["domains"]


    def Count_Type_Info(self):
        wi_type_set = {
            "attraction": set(),
            "hotel": set(),
        }
        wo_type_set = {
            "attraction": set(),
            "hotel": set(),
        }
        tmp = {
            "attraction": set(),
            "hotel": set(),
        }
        for domain in tmp:
            slot_type = "type"
            self.tmp_log = {}
            self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + f"_{domain}_{slot_type}.json"
            for idx, turn in self.data.items():
                slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
                for dom in [domain]:
                # for domain in ["attraction", "hotel", "restaurant"]:
                    if dom in turn["domains"]:
                        if dom+" "+slot_type in slots_str:
                            wi_type_set[dom].add(turn["dial_id"])
                        else:
                            wo_type_set[dom].add(turn["dial_id"])
            tmp_union = set()
            for dom in tmp:
                tmp[dom] = wo_type_set[dom] - wi_type_set[dom]
                # tmp[dom] = wi_type_set[dom]
                tmp_union = tmp_union.union(tmp[dom])
            tmp_union = sorted(list(tmp_union))
            domain_related = sorted(list(wo_type_set[domain].union(wi_type_set[domain])))

            for dial_id in domain_related:
                for turn_num in range(30):
                    idx = dial_id + "-" + str(turn_num)
                    if idx in self.data:
                        turn = self.data[idx]
                    else:
                        continue
                    # already labeled
                    dom_type_list = [slot[0]+"--"+slot[1] for slot in turn["slots"]]
                    if domain+"--"+slot_type in dom_type_list:
                        break
                    # not labeled
                    last_turn = turn["context"].split("<system>")[-1]
                    sys_utt, user_utt = last_turn.split("<user>")
                    slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])

                    if domain == "hotel":
                        if "hotel" not in last_turn and ("guesthouse" in user_utt or "guest house" in user_utt):
                        # works 14 for user
                            pass
                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "guesthouse")

                        elif "hotel" in user_utt \
                            and not ("guesthouse" in last_turn or "guest house" in last_turn):
                            if not ("the hotel" in last_turn \
                                and "the hotel should" not in user_utt \
                                and "looking for the hotel" not in user_utt) \
                                and "car " not in last_turn \
                                and "taxi" not in last_turn \
                                and "type" not in last_turn :
                                # works for 134
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "hotel")

                        elif "hotel" in sys_utt \
                            and not ("guesthouse" in last_turn or "guest house" in last_turn):
                            if "the hotel" not in last_turn \
                                and "that hotel" not in last_turn \
                                and "taxi" not in last_turn:
                                # works 107 for sys
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "hotel")

                    if domain == "attraction":
                        # work for 146 turns; for user side 39; for sys side 107
                        poss_value = {"museum":"museum", 
                                    "college":"college",
                                    "nightclub":"nightclub",
                                    "architecture":"architecture",
                                    "entertainment":"entertainment",
                                    "theatre":"theatre",
                                    "park":"park",
                                    "swimmingpool":"swimmingpool",
                                    "boat":"boat",
                                    "cinema":"cinema",
                                    "multiple sports":"multiple sports",
                                    "concerthall":"concerthall",
                                    "hiking":"hiking",
                                    "night club":"nightclub",
                                    "theater":"theatre",
                                    "swimming pool":"swimmingpool",
                                    "architectural":"architecture",
                                    "entertain":"entertainment",
                                    "concert hall":"concerthall",
                                    }
                        flag = 0
                        for value in poss_value:
                            if value in user_utt \
                            and "taxi" not in last_turn \
                            and "parking" not in last_turn:
                                flag = 1
                                # pass
                                label_value = poss_value[value]
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                # break
                        if flag == 1:
                            continue

                        for value in poss_value:
                            if value in sys_utt \
                            and "parking" not in last_turn \
                            and not ("leisure park" in last_turn or "park street" in last_turn or "parkside" in last_turn):
                                label_value = poss_value[value]
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)

                        if idx in self.tmp_log and "|" in self.tmp_log[idx]["add_slot"]:
                            slot_values = self.tmp_log[idx]["add_slot"].split(" | ")
                            slot_values[0] = slot_values[0].split()[-1]
                            if "attraction name" in " ".join([" ".join(slot) for slot in turn["slots"]]):
                                if idx == "sng1105.json-2":
                                    # self.tmp_log[idx]["add_slot"] = f"attraction type entertainment"
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "entertainment")
                                if idx == "mul2466.json-3":
                                    # self.tmp_log[idx]["add_slot"] = f"attraction type nightclub"
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "nightclub")
                            else:
                                if idx == "pmul2272.json-5":
                                    # self.tmp_log[idx]["add_slot"] = "attraction type cinema"
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "cinema")
                                elif idx not in ["pmul1455.json-1"]:
                                    del self.tmp_log[idx]
                                    # self.tmp_log[idx]["add_slot"] = f"{domain} {slot_type} "

            self._save_tmp_log()
                          
    def Count_Dest_Depa_Info(self):
        wi_type_set = {
            "taxi": set(),
            "train": set(),
        }
        wo_type_set = {
            "taxi": set(),
            "train": set(),
        }
        tmp = {
            "taxi": set(),
            "train": set(),
        }
        for domain in tmp:
            for slot_type in ["departure", "destination"]:
                self.tmp_log = {}
                self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + f"_{domain}_{slot_type}.json"
                self._load_otgy()
                poss_value = self.otgy[f"{domain}--{slot_type}"]
                for idx, turn in self.data.items():
                    slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
                    for dom in [domain]:
                        if dom in turn["domains"]:
                            if dom+" "+slot_type in slots_str:
                                wi_type_set[dom].add(turn["dial_id"])
                            else:
                                wo_type_set[dom].add(turn["dial_id"])
                tmp_union = set()
                for dom in tmp:
                    tmp[dom] = wo_type_set[dom] - wi_type_set[dom]
                    # tmp[dom] = wi_type_set[dom]
                    tmp_union = tmp_union.union(tmp[dom])
                tmp_union = sorted(list(tmp_union))
                domain_related = sorted(list(wo_type_set[domain].union(wi_type_set[domain])))

                for dial_id in domain_related:
                    for turn_num in range(30):
                        idx = dial_id + "-" + str(turn_num)
                        if idx in self.data:
                            turn = self.data[idx]
                        else:
                            continue
                        # already labeled
                        dom_type_list = [slot[0]+"--"+slot[1] for slot in turn["slots"]]
                        if domain+"--"+slot_type in dom_type_list:
                            break
                        # not labeled
                        last_turn = turn["context"].split("<system>")[-1]
                        sys_utt, user_utt = last_turn.split("<user>")
                        slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])
                        flag = 0
                        if domain == "train" and slot_type == "departure":
                            # in total for 11
                            for value in poss_value:
                                label_value = value
                                
                                if " "+value in user_utt:
                                    if value not in slots_string \
                                        and re.search(r"((leave|depart)s?|from) "+value, user_utt):
                                        # # work for 2
                                        flag = 1
                                        pass
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                            if flag == 1:
                                continue

                            for value in poss_value:
                                label_value = value
                                if value in sys_utt:
                                    if value not in slots_string \
                                        and re.search(r"((leave|depart)s?|from) "+value, sys_utt):
                                        # # work for 9 
                                        pass
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                            
                        if domain == "train" and slot_type == "destination":
                            # in total for 2
                            for value in poss_value:
                                label_value = value
                                
                                if value in user_utt:
                                    if value not in slots_string \
                                        and "to "+value in user_utt:
                                        # # work for 0
                                        flag = 1
                                        pass
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                            if flag == 1:
                                continue
                            
                            for value in poss_value:
                                label_value = value
                                if value in sys_utt:
                                    if value not in slots_string \
                                        and re.search(r"(arrives?|arrives? (in|at)|to) "+value, sys_utt):
                                        # # work for 2
                                        pass
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                            
                        if domain == "taxi" and slot_type == "departure":
                            # in total for 
                            for value in poss_value:
                                label_value = value
                                
                                if " "+value in user_utt:
                                    if value not in slots_string \
                                        and re.search(r"((leave|depart)s?|from) "+value, user_utt):
                                        # # work for 2
                                        if value  == "the hotel" or value == "the restaurant":
                                            ref_dom = value.split()[-1]
                                            for slot in turn["slots"]:
                                                if slot[0] == ref_dom and slot[1] == "name":
                                                    label_value = slot[2]
                                                    flag = 1
                                                    pass
                                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                                    break
                                        else:
                                            flag = 1
                                            pass
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                            if flag == 1:
                                continue

                            for value in poss_value:
                                label_value = value
                                if value in sys_utt:
                                    if value not in slots_string \
                                        and 'taxi' in sys_utt \
                                        and re.search(r"((leave|depart)s?|from) "+value, sys_utt):
                                        # # work for 9 
                                        if value  == "the hotel" or value == "the restaurant":
                                            ref_dom = value.split()[-1]
                                            for slot in turn["slots"]:
                                                if slot[0] == ref_dom and slot[1] == "name":
                                                    label_value = slot[2]
                                                    flag = 1
                                                    pass
                                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                                    break
                                        else:
                                            pass
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                            
                        if domain == "taxi" and slot_type == "destination":
                            poss_value["the hotel"] = 5
                            # in total for 
                            weird_value_list = ["star", "ask", "cambridge"]
                            for value in poss_value:
                                label_value = value
                                
                                if value in user_utt:
                                    if value not in slots_string \
                                        and value not in weird_value_list \
                                        and poss_value[value] > 1 \
                                        and 'taxi' in user_utt \
                                        and "to "+value in user_utt:
                                        # # work for 
                                        if value  == "the hotel" or value == "the restaurant":
                                            ref_dom = value.split()[-1]
                                            for slot in turn["slots"]:
                                                if slot[0] == ref_dom and slot[1] == "name":
                                                    label_value = slot[2]
                                                    flag = 1
                                                    pass
                                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                                    break
                                        else:
                                            flag = 1
                                            pass
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                            if flag == 1:
                                continue
                            
                            for value in poss_value:
                                label_value = value
                                if value in sys_utt:
                                    if value not in slots_string \
                                        and value not in weird_value_list \
                                        and poss_value[value] > 1 \
                                        and 'taxi' in sys_utt \
                                        and re.search(r"(arrives?|arrives? (in|at)| to) "+value, sys_utt):
                                        # # work for 2
                                        if value  == "the hotel" or value == "the restaurant":
                                            ref_dom = value.split()[-1]
                                            for slot in turn["slots"]:
                                                if slot[0] == ref_dom and slot[1] == "name":
                                                    label_value = slot[2]
                                                    flag = 1
                                                    pass
                                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                                    break
                                        else:
                                            pass
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                            
                self._save_tmp_log()
                                        
    def Count_Area_Info(self):
        wi_type_set = {
            "attraction": set(),
            "hotel": set(),
            "restaurant": set(),
        }
        wo_type_set = {
            "attraction": set(),
            "hotel": set(),
            "restaurant": set(),
        }
        tmp = {
            "attraction": set(),
            "hotel": set(),
            "restaurant": set(),
        }
        for domain in tmp:
            slot_type = "area"
            self.tmp_log = {}
            self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + f"_{domain}_{slot_type}.json"
            for idx, turn in self.data.items():
                slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
                for dom in [domain]:
                    if dom in turn["domains"]:
                        if dom+" "+slot_type in slots_str:
                            wi_type_set[dom].add(turn["dial_id"])
                        else:
                            wo_type_set[dom].add(turn["dial_id"])
            tmp_union = set()
            for dom in tmp:
                tmp[dom] = wo_type_set[dom] - wi_type_set[dom]
                # tmp[dom] = wi_type_set[dom]
                tmp_union = tmp_union.union(tmp[dom])
            tmp_union = sorted(list(tmp_union))
            domain_related = sorted(list(wo_type_set[domain].union(wi_type_set[domain])))

            for dial_id in domain_related:
                for turn_num in range(30):
                    idx = dial_id + "-" + str(turn_num)
                    if idx in self.data:
                        turn = self.data[idx]
                    else:
                        continue
                    # already labeled
                    dom_type_list = [slot[0]+"--"+slot[1] for slot in turn["slots"]]
                    if domain+"--"+slot_type in dom_type_list:
                        break
                    # not labeled
                    last_turn = turn["context"].split("<system>")[-1]
                    sys_utt, user_utt = last_turn.split("<user>")
                    slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])
                    poss_value = {
                        "centre": "centre",
                        "center": "centre",
                        " west": "west",
                        " east": "east",
                        "south": "south",
                        "north": "north"
                                    }

                    flag = 0

                    if domain == "attraction":
                        for value in poss_value:
                            value = poss_value[value]
                            # none for attraction area value in user_utt
                            if value in sys_utt and value+" road" not in sys_utt:
                                name_val, other_name = "", ""
                                for slot in turn["slots"]:
                                    if slot[0] == domain and slot[1] == "name":
                                        name_val = slot[2]
                                    elif slot[1] == "name" and slot[0] != domain:
                                        other_name = slot[2]
                                # prev utt
                                prev_id = idx.split("-")[0]+"-"+str(turn["turn_num"]-1)
                                prev_utt = self.data[prev_id]["context"].split("<system>")[-1]

                            
                                # in total for 91 turns
                                if "attraction name" in slots_string \
                                    and name_val in sys_utt:
                                    # work for 46
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)
                                
                                if "attraction name" in slots_string \
                                    and name_val not in sys_utt \
                                    and ((other_name and other_name not in sys_utt) or not other_name) \
                                    and "restaurant" not in sys_utt \
                                    and "hotel" not in sys_utt \
                                    and "house" not in sys_utt \
                                    and "book" not in sys_utt:

                                    if "college" in sys_utt \
                                        or "museum" in sys_utt \
                                        or "night club" in sys_utt \
                                        or "nightclub" in sys_utt \
                                        or "swimming" in sys_utt \
                                        or "cinema" in sys_utt \
                                        or "theatre" in sys_utt \
                                        or "theater" in sys_utt \
                                        or "entertain" in sys_utt \
                                        or "architect" in sys_utt:
                                        # work for 24
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)

                                    elif name_val in prev_utt:
                                        # work for 11
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)

                                    elif name_val not in prev_utt \
                                            and "hotel" not in prev_utt \
                                            and (slots_string.count("name") == 1 \
                                            or  "museum" in prev_utt \
                                            or idx =="pmul3423.json-11"):
                                        # work for 9
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)
                
                    if domain == "hotel":
                        # in total for 100
                        for value in poss_value:
                            value = poss_value[value]
                            other_domain, name_val = "", ""
                            for slot in turn["slots"]:
                                if slot[1] == "area" and slot[2] == value:
                                    other_domain = slot[0]
                                if slot[0] == domain and slot[1] == "name":
                                    name_val = slot[2]
                            if value in user_utt and idx == "mul0088.json-1":
                                # for user side: mul0088.json-1
                                pass
                                flag = 1
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)
                            
                        if flag == 0:
                            for value in poss_value:
                                value = poss_value[value]
                                other_domain, name_val = "", ""
                                for slot in turn["slots"]:
                                    if slot[0] == domain and slot[1] == "name":
                                        name_val = slot[2]
                                if value in sys_utt:
                                    if name_val \
                                    and name_val in sys_utt \
                                    and value not in name_val: 
                                        # work for 48
                                        # pass
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)
                                    elif "restaurant" not in sys_utt \
                                        and ("guest" in sys_utt or "hotel" in sys_utt) \
                                        and value in last_turn.replace("centre north", ""):
                                        # work for 51
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)
                                
                    if domain == "restaurant":
                        # in total for 98
                        for value in poss_value:
                            value = poss_value[value]
                            if value in user_utt and idx in ["mul2365.json-6", "pmul3495.json-4"]:
                                # # for user side: mul2365.json-6, pmul3495.json-4
                                pass  
                                flag = 1
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)
                            
                        if flag == 0:
                            for value in poss_value:
                                value = poss_value[value]
                                other_domain, name_val = "", ""
                                for slot in turn["slots"]:
                                    if slot[0] == domain and slot[1] == "name":
                                        name_val = slot[2]
                                if value in sys_utt:
                                    # prev utt
                                    prev_id = idx.split("-")[0]+"-"+str(turn["turn_num"]-1)
                                    prev_utt = self.data[prev_id]["context"].split("<system>")[-1]
                                    if name_val \
                                    and name_val in sys_utt \
                                    and value not in name_val:
                                        if value + " road" not in sys_utt \
                                        and value+"ampton road" not in sys_utt \
                                        and "nandos city "+value not in sys_utt \
                                        and "city "+value+" on the west side" not in sys_utt: 
                                            # work for 47
                                            pass
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)
                                    elif "restaurant" in sys_utt:
                                        # work for 10
                                        pass
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)

                                    elif "hotel" not in sys_utt \
                                        and "guest" not in sys_utt \
                                        and "attraction" not in sys_utt \
                                        and "college" not in sys_utt \
                                        and "night" not in sys_utt \
                                        and "church" not in sys_utt \
                                        and "pool" not in sys_utt \
                                        and "museum" not in sys_utt:
                                        if name_val:
                                            # work for 23
                                            if "restaurant" not in user_utt \
                                                and idx not in ["mul0004.json-1", "mul0004.json-6"] \
                                                and "17 magdalene street city "+value not in sys_utt \
                                                and "183 east road city "+value not in sys_utt:
                                                pass
                                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)

                                        elif "bistro" in sys_utt \
                                            or "noodle" in sys_utt \
                                            or "noodle" in prev_utt \
                                            or "turkish restuarant" in sys_utt \
                                            or "dine in" in sys_utt \
                                            or "bedouin" in sys_utt \
                                            or "clowns cafe" in sys_utt \
                                            or "lan hong house" in sys_utt \
                                            or "cocum" in sys_utt \
                                            or "kohinoor" in sys_utt \
                                            or "charlie chan" in sys_utt \
                                            or "charlie chan" in prev_utt:
                                            # work for 13
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, value)

            self._save_tmp_log()

    def Count_Price_Info(self):
        wi_type_set = {
            "hotel": set(),
            "restaurant": set(),
        }
        wo_type_set = {
            "hotel": set(),
            "restaurant": set(),
        }
        tmp = {
            "hotel": set(),
            "restaurant": set(),
        }
        for domain in tmp:
            slot_type = "pricerange"
            self.tmp_log = {}
            self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + f"_{domain}_{slot_type}.json"
            for idx, turn in self.data.items():
                slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
                for dom in [domain]:
                    if dom in turn["domains"]:
                        if dom+" "+slot_type in slots_str:
                            wi_type_set[dom].add(turn["dial_id"])
                        else:
                            wo_type_set[dom].add(turn["dial_id"])
            tmp_union = set()
            for dom in tmp:
                tmp[dom] = wo_type_set[dom] - wi_type_set[dom]
                # tmp[dom] = wi_type_set[dom]
                tmp_union = tmp_union.union(tmp[dom])
            tmp_union = sorted(list(tmp_union))
            domain_related = sorted(list(wo_type_set[domain].union(wi_type_set[domain])))

            for dial_id in domain_related:
                # if dial_id in tmp_union:
                #     continue

                for turn_num in range(30):
                    idx = dial_id + "-" + str(turn_num)
                    if idx in self.data:
                        turn = self.data[idx]
                    else:
                        continue
                    # already labeled
                    dom_type_list = [slot[0]+"--"+slot[1] for slot in turn["slots"]]
                    if domain+"--"+slot_type in dom_type_list:
                        break
                    # not labeled
                    last_turn = turn["context"].split("<system>")[-1]
                    sys_utt, user_utt = last_turn.split("<user>")
                    slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])
                    poss_value = {
                        "cheap": "cheap",
                        "moderate": "moderate",
                        " expensive": "expensive",
                        "inexpensive": "moderate",
                                    }
                    flag = 0

                    if domain == "restaurant":
                        # in total for 84
                        for value in poss_value:
                            label_value = poss_value[value]
                            
                            if value in user_utt and value not in slots_string and "guest" not in last_turn:
                            # # work for 3
                                pass
                                flag = 1
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                        if flag == 0:
                            for value in poss_value:
                                label_value = poss_value[value]
                                other_domain, name_val, other_name_val = "", "", ""
                                for slot in turn["slots"]:
                                    
                                    if slot[0] == domain and slot[1] == "name":
                                        name_val = slot[2]
                                    if slot[0] != domain and slot[1] == "name":
                                        other_name_val = slot[2]
                                if value in sys_utt:
                                    # prev utt
                                    prev_id = idx.split("-")[0]+"-"+str(turn["turn_num"]-1)
                                    prev_utt = self.data[prev_id]["context"].split("<system>")[-1]
                                    if name_val:
                                        if  name_val in sys_utt:
                                            # work for 42
                                            pass
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                        elif name_val in prev_utt and "hotel" not in sys_utt:
                                            # work for 16
                                            pass
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                        elif name_val not in prev_utt \
                                            and "hotel" not in sys_utt \
                                            and "guest" not in sys_utt \
                                            and ("restaurant" in sys_utt or "hotel" not in turn["domains"]):
                                            # work for 7
                                            pass
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                    elif not (other_name_val and other_name_val in sys_utt) \
                                        and "accommodations" not in sys_utt \
                                        and "hotel" not in sys_utt:
                                        # work for 16
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)

                    if domain == "hotel":
                        # in total for 113
                        if idx == "mul1273.json-1":
                            continue
                        for value in poss_value:
                            label_value = poss_value[value]
                            if value in user_utt and value not in slots_string and "how expensive" not in user_utt and "cheapest" not in user_utt:
                                # # work for 3
                                pass
                                flag = 1
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                        if flag == 0:
                            for value in poss_value:
                                label_value = poss_value[value]
                                other_domain, name_val, other_name_val = "", "", ""
                                for slot in turn["slots"]:
                                    if slot[1] == slot_type and slot[2] == value:
                                        other_domain = slot[0]
                                    if slot[0] == domain and slot[1] == "name":
                                        name_val = slot[2]
                                    if slot[0] != domain and slot[1] == "name":
                                        other_name_val = slot[2]
                                if value in sys_utt:
                                    # prev utt
                                    prev_id = idx.split("-")[0]+"-"+str(turn["turn_num"]-1)
                                    prev_utt = self.data[prev_id]["context"].split("<system>")[-1]
                                    if name_val:
                                        if name_val in sys_utt:
                                            # work for 39
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                            pass
                                        elif "dine" not in sys_utt \
                                            and "cuisine" not in sys_utt \
                                            and "steakhouse" not in sys_utt \
                                            and "restaurant" not in sys_utt:
                                            # work for 32
                                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                            pass
                                    elif "restaurant" not in sys_utt \
                                        and "hotel" in slots_string:
                                        # work for 41
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)

            self._save_tmp_log()

    def Count_Food_Info(self):
        wi_type_set = {
            "restaurant": set(),
        }
        wo_type_set = {
            "restaurant": set(),
        }
        tmp = {
            "restaurant": set(),
        }
        domain = "restaurant"
        slot_type = "food"
        self.tmp_log = {}
        self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + f"_{domain}_{slot_type}.json"
        self._load_otgy()
        poss_value = self.otgy["restaurant--food"]
        for idx, turn in self.data.items():
            slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
            if domain in turn["domains"]:
                if domain+" "+slot_type in slots_str:
                    wi_type_set[domain].add(turn["dial_id"])
                else:
                    wo_type_set[domain].add(turn["dial_id"])
        tmp_union = wo_type_set[domain] - wi_type_set[domain]
        tmp_union = sorted(list(tmp_union))
        domain_related = sorted(list(wo_type_set[domain].union(wi_type_set[domain])))

        for dial_id in domain_related:
            # if dial_id in tmp_union:
            #     continue
            for turn_num in range(30):
                idx = dial_id + "-" + str(turn_num)
                if idx in self.data:
                    turn = self.data[idx]
                else:
                    continue
                # already labeled
                dom_type_list = [slot[0]+"--"+slot[1] for slot in turn["slots"]]
                if domain+"--"+slot_type in dom_type_list:
                    break
                # not labeled
                last_turn = turn["context"].split("<system>")[-1]
                sys_utt, user_utt = last_turn.split("<user>")
                slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])
                flag = 0
                other_domain, name_val, other_name_val = "", "", ""
                for slot in turn["slots"]:
                    if slot[0] == domain and slot[1] == "name":
                        name_val = slot[2]
                weird_word_list = ["local"]
                for value in poss_value:
                    label_value = value
                    # in total for 85
                    if re.search(r"[^a-z]"+value+r"[^a-z]", user_utt) \
                        and value not in name_val \
                        and value not in weird_word_list:
                    # # work for 2
                        pass
                        flag = 1
                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                
                if flag == 0:
                    if idx == "mul0831.json-4":
                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "unusual")
                    for value in poss_value:
                        label_value = value
                        
                        if re.search(r"[^a-z]"+value+r"[^a-z]", sys_utt):
                            # prev utt
                            prev_id = idx.split("-")[0]+"-"+str(turn["turn_num"]-1)
                            prev_utt = self.data[prev_id]["context"].split("<system>")[-1]
                            if name_val:
                                if name_val in sys_utt and value not in name_val:
                                    # work for 51
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                    pass
                                elif name_val not in last_turn \
                                    and value not in name_val \
                                    and "cinema" not in sys_utt:
                                    # work for 26
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                    pass
                            elif "restaurant" in slots_string:
                                # work for 6
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)

    def Count_Parking_Info(self):
        wi_type_set = {
            "hotel": set(),
        }
        wo_type_set = {
            "hotel": set(),
        }
        tmp = {
            "hotel": set(),
        }
        domain = "hotel"
        slot_type = "parking"
        self.tmp_log = {}
        self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + f"_{domain}_{slot_type}.json"
        self._load_otgy()
        poss_value = ["parking"]
        for idx, turn in self.data.items():
            slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
            if domain in turn["domains"]:
                if domain+" "+slot_type in slots_str:
                    wi_type_set[domain].add(turn["dial_id"])
                else:
                    wo_type_set[domain].add(turn["dial_id"])
        tmp_union = wo_type_set[domain] - wi_type_set[domain]
        tmp_union = sorted(list(tmp_union))
        domain_related = sorted(list(wo_type_set[domain].union(wi_type_set[domain])))

        for dial_id in domain_related:
            # if dial_id in tmp_union:
            #     continue
            for turn_num in range(30):
                idx = dial_id + "-" + str(turn_num)
                if idx in self.data:
                    turn = self.data[idx]
                else:
                    continue
                # already labeled
                dom_type_list = [slot[0]+"--"+slot[1] for slot in turn["slots"]]
                if domain+"--"+slot_type in dom_type_list:
                    break
                # not labeled
                last_turn = turn["context"].split("<system>")[-1]
                sys_utt, user_utt = last_turn.split("<user>")
                slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])

                for value in poss_value:
                    label_value = value
                    # in total for 83
                    other_domain, name_val, other_name_val = "", "", ""
                    for slot in turn["slots"]:
                        if slot[1] == slot_type and slot[2] == value:
                            other_domain = slot[0]
                        if slot[0] == domain and slot[1] == "name":
                            name_val = slot[2]
                        if slot[0] != domain and slot[1] == "name":
                            other_name_val = slot[2]
                    if value in user_utt:
                        if "parking or not" not in user_utt \
                            and "whether" not in user_utt \
                            and " if " not in user_utt \
                            and not re.search(r"do.*parking\?", user_utt):
                            # # work for 13
                            if ("not" in user_utt or "n't" in user_utt) \
                                and self.token_distance(user_utt, "not", "parking") < 5 \
                                or "no parking" in user_utt:
                                pass
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "no")
                            else:
                                pass
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "yes")
                        else:
                            # ask question about parking, work for 12
                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "yes")
                            pass
                    elif value in sys_utt:
                        # prev utt
                        prev_id = idx.split("-")[0]+"-"+str(turn["turn_num"]-1)
                        prev_utt = self.data[prev_id]["context"].split("<system>")[-1]
                        if name_val:
                            # work for 41
                            if ("not" in sys_utt or "n't" in sys_utt) \
                                and self.token_distance(sys_utt, "not", "parking") < 5 \
                                or "no parking" in sys_utt:
                                pass
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "no")
                            else:
                                pass
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "yes")
                        elif not re.search(r"parking[a-z ]*\?", sys_utt) \
                            and not re.search(r"\(.*parking\)", sys_utt) \
                            and not "if you need parking" in sys_utt \
                            and not "parking, or internet?" in sys_utt:
                            # work for 16
                            if ("not" in sys_utt or "n't" in sys_utt) \
                                and self.token_distance(sys_utt, "not", "parking") < 5 \
                                or "no parking" in sys_utt:
                                pass
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "no")
                            else:
                                pass
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "yes")

    def Count_Internet_Info(self):
        wi_type_set = {
            "hotel": set(),
        }
        wo_type_set = {
            "hotel": set(),
        }
        tmp = {
            "hotel": set(),
        }
        domain = "hotel"
        slot_type = "internet"
        self.tmp_log = {}
        self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + f"_{domain}_{slot_type}.json"
        self._load_otgy()
        poss_value = ["internet", "wifi"]
        for idx, turn in self.data.items():
            slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
            if domain in turn["domains"]:
                if domain+" "+slot_type in slots_str:
                    wi_type_set[domain].add(turn["dial_id"])
                else:
                    wo_type_set[domain].add(turn["dial_id"])
        tmp_union = wo_type_set[domain] - wi_type_set[domain]
        tmp_union = sorted(list(tmp_union))
        domain_related = sorted(list(wo_type_set[domain].union(wi_type_set[domain])))

        for dial_id in domain_related:
            # if dial_id in tmp_union:
            #     continue
            for turn_num in range(30):
                idx = dial_id + "-" + str(turn_num)
                if idx in self.data:
                    turn = self.data[idx]
                else:
                    continue
                # already labeled
                dom_type_list = [slot[0]+"--"+slot[1] for slot in turn["slots"]]
                if domain+"--"+slot_type in dom_type_list:
                    break
                # not labeled
                last_turn = turn["context"].split("<system>")[-1]
                sys_utt, user_utt = last_turn.split("<user>")
                slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])
                flag = 0
                other_domain, name_val, other_name_val = "", "", ""
                for slot in turn["slots"]:
                    if slot[1] == slot_type and slot[2] == value:
                        other_domain = slot[0]
                    if slot[0] == domain and slot[1] == "name":
                        name_val = slot[2]
                    if slot[0] != domain and slot[1] == "name":
                        other_name_val = slot[2]
                if idx == "mul1189.json-4":
                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "no")
                    continue
                for value in poss_value:
                    label_value = value
                    # in total for 79
                    if value in user_utt:
                        # work for 19
                        if value+" or not" not in user_utt \
                            and "whether" not in user_utt \
                            and " if " not in user_utt \
                            and not re.search(r"do[a-z ]*(internet|wifi)[a-z', ]*\?", user_utt):
                            # # work for 10
                            if ("not" in user_utt or "n't" in user_utt) \
                                and self.token_distance(user_utt, "not", value) < 5 \
                                or "no "+value in user_utt:
                                pass
                                flag = 1
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "no")
                            else:
                                pass
                                flag = 1
                                self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "yes")
                        else:
                            # ask question about internet, work for 9
                            flag = 1
                            self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "yes")
                            pass
                if flag == 0:
                    for value in poss_value:
                        label_value = value
                        if value in sys_utt:
                            # work for 60
                            # prev utt
                            prev_id = idx.split("-")[0]+"-"+str(turn["turn_num"]-1)
                            prev_utt = self.data[prev_id]["context"].split("<system>")[-1]
                            if name_val:
                                # work for 44
                                if ("not" in sys_utt or "n't" in sys_utt) \
                                    and self.token_distance(sys_utt, "not", value) < 2 \
                                    or "no "+value in sys_utt:
                                    pass
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "no")
                                else:
                                    pass
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "yes")
                            elif not re.search(value+r"[a-z ]*\?", sys_utt) \
                                and not re.search(r"\(.*"+value+r"\)", sys_utt) \
                                and not f"if you need parking" in sys_utt \
                                and not f"parking, or {value}?" in sys_utt:
                                # work for 16
                                if ("not" in sys_utt or "n't" in sys_utt) \
                                    and self.token_distance(sys_utt, "not", value) < 5 \
                                    or "no "+value in sys_utt:
                                    pass
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "no")
                                else:
                                    pass
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, "yes")

    def Count_Star_Info(self):
        wi_type_set = {
            "hotel": set(),
        }
        wo_type_set = {
            "hotel": set(),
        }
        tmp = {
            "hotel": set(),
        }
        domain = "hotel"
        slot_type = "stars"
        self.tmp_log = {}
        self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + f"_{domain}_{slot_type}.json"

        num_map = {
            " one":" 1",
            " two" : " 2",
            " three" : " 3",
            " four" : " 4",
            " five" : " 5",
            " six" : " 6",
            " seven" : " 7",
            " eight" : " 8",
            " nine" : " 9",
            " zero" : " 0"
        }

        for idx, turn in self.data.items():
            slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
            if domain in turn["domains"]:
                if domain+" "+slot_type in slots_str:
                    wi_type_set[domain].add(turn["dial_id"])
                else:
                    wo_type_set[domain].add(turn["dial_id"])
        tmp_union = wo_type_set[domain] - wi_type_set[domain]
        tmp_union = sorted(list(tmp_union))
        domain_related = sorted(list(wo_type_set[domain].union(wi_type_set[domain])))

        for dial_id in domain_related:
            for turn_num in range(30):
                idx = dial_id + "-" + str(turn_num)
                if idx in self.data:
                    turn = self.data[idx]
                else:
                    continue
                # already labeled
                dom_type_list = [slot[0]+"--"+slot[1] for slot in turn["slots"]]
                if domain+"--"+slot_type in dom_type_list:
                    break
                # not labeled
                last_turn = turn["context"].split("<system>")[-1]
                sys_utt, user_utt = last_turn.split("<user>")
                slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])

                # in total for 100
                other_domain, name_val, other_name_val = "", "", ""
                for slot in turn["slots"]:
                    if slot[1] == slot_type and slot[2] == value:
                        other_domain = slot[0]
                    if slot[0] == domain and slot[1] == "name":
                        name_val = slot[2]
                    if slot[0] != domain and slot[1] == "name":
                        other_name_val = slot[2]
                if re.search(r"star[s ]", sys_utt):
                    # work for 0
                    for num_word, num in num_map.items():
                        for suff in [" ", "-"]:
                            sys_utt = re.sub(num_word+suff, num+suff, sys_utt)
                    if re.search(r"[0-9][- ]star[s ]", sys_utt):
                        tokens = re.findall(r"\d+[- ]star[s ]", sys_utt)
                        label_value = []
                        for token in tokens:
                            if "-" in token:
                                label_value.append(token.split("-")[0])
                            else:
                                label_value.append(token.split()[0])
                        label_value = list(set(label_value))
                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, " | ".join(label_value))
                        pass
                    elif re.search(r"star rating of [0-9]", sys_utt):
                        tokens = re.findall(r"star rating of \d+", sys_utt)
                        label_value = []
                        for token in tokens:
                            label_value.append(token.split()[3])
                        label_value = list(set(label_value))
                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, " | ".join(label_value))
                        pass

    def Count_Name_Info(self):
        wi_type_set = {
            "attraction": set(),
            "hotel": set(),
            "restaurant": set(),
        }
        wo_type_set = {
            "attraction": set(),
            "hotel": set(),
            "restaurant": set(),
        }
        tmp = {
            "attraction": set(),
            "hotel": set(),
            "restaurant": set(),
        }
        for domain in ["restaurant"]:#tmp:
            slot_type = "name"
            self.tmp_log = {}
            # self.tmp_log_path = ".".join(self.data_path.split(".")[:-1]) + f"_{domain}_{slot_type}.json"
            self._load_otgy()
            poss_value = self.otgy[f"{domain}--{slot_type}"]
            for idx, turn in self.data.items():
                slots_str = ", ".join([" ".join(slot) for slot in turn["slots"]])
                if domain in turn["domains"]:
                    if domain+" "+slot_type in slots_str:
                        wi_type_set[domain].add(turn["dial_id"])
                    else:
                        wo_type_set[domain].add(turn["dial_id"])
            tmp_union = wo_type_set[domain] - wi_type_set[domain]
            tmp_union = sorted(list(tmp_union))
            domain_related = sorted(list(wo_type_set[domain].union(wi_type_set[domain])))
            
            for dial_id in tqdm(domain_related):
                for turn_num in range(30):
                    idx = dial_id + "-" + str(turn_num)
                    if idx in self.data:
                        turn = self.data[idx]
                    else:
                        continue
                    # already labeled
                    dom_type_list = [slot[0]+"--"+slot[1] for slot in turn["slots"]]
                    if domain+"--"+slot_type in dom_type_list:
                        continue
                    # not labeled
                    last_turn = turn["context"].split("<system>")[-1]
                    sys_utt, user_utt = last_turn.split("<user>")
                    slots_string = ", ".join([" ".join(slot) for slot in turn["slots"]])
                    flag = 0
                    other_domain, name_val, other_name_val = "", "", ""
                    for slot in turn["slots"]:
                        if slot[0] == domain and slot[1] == "name":
                            name_val = slot[2]
                        if slot[0] != domain and slot[1] == "name":
                            other_name_val = slot[2]

                    if domain == "restaurant":
                        # in total for 137
                        # adding
                        poss_value.update({
                            "caffe uno":0,
                            "travelers rest":0,
                        })
                        # removing
                        weird_rest_name = ["one", "ali", "bridge", "ask", "indian", 
                                            "south", "city", "italian restaurant", 
                                            "other restaurant", "ashley hotel", 
                                            "pizza", "funky","scudamores punt","scudamores punting",
                                            "molecular gastronomy", "broughton house gallery",
                                            "cambridge punter", "el shaddai",  "el shaddia guesthouse",
                                            "indian", "indiana restaurants", "south", ]
                            
                        for value in poss_value:
                            label_value = value
                            # if " "+value in user_utt:
                            if re.search(r"[^a-z]"+value+r"[^a-z]", user_utt):
                                if value not in weird_rest_name\
                                    and value not in other_name_val \
                                    and "taxi" not in user_utt:
                                    # # work for 18
                                    flag = 1
                                    # self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                    pass
                        if flag == 0:
                            for value in poss_value:
                                label_value = value
                                if value not in weird_rest_name \
                                    and value not in other_name_val:
                                    # used to use weird list ["one", "ali", "bridge", "ask", "indian", "south", "city",  "italian restaurant", "other restaurant"]
                                    if re.search(r"[^a-z]"+value+r"[^a-z]", sys_utt) and value not in user_utt:
                                        # work for 125
                                        # self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                        pass
                                    elif value != "j restaurant" and fuzz.partial_ratio(value, sys_utt) >= 90:

                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                        pass


                    if domain == "hotel":
                        # in total for 150
                        # adding
                        poss_value.update({
                            "rosa's bed and breakfast": 0,
                            "alexander b & b": 0,
                            "marriott": 0,
                        })
                        # removing
                        weird_rest_name = ["yes", "hotel", "sou", "north", "bridge","doubletree by hilton cambridge"]
                        
                        for value in poss_value:
                            label_value = value
                            
                            if re.search(r"[^a-z]"+value+r"[^a-z]", user_utt):
                                if value not in weird_rest_name \
                                    and value not in other_name_val \
                                    and "taxi" not in user_utt:
                                    # # work for 11
                                    pass
                                    flag = 1
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                        if flag == 0:
                            for value in poss_value:
                                label_value = value

                                if re.search(r"[^a-z]"+value+r"[^a-z]", sys_utt) and value not in user_utt \
                                    and value not in weird_rest_name \
                                    and "taxi" not in sys_utt \
                                    and value not in other_name_val:
                                    # work for 146
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                    pass

                    if domain == "attraction":
                        # in total for 147
                        # adding
                        poss_value.update({
                            "all saints":0,
                        })
                        # removing
                        weird_val_list = ["pizza", "country park", "university arms hotel", 
                                        "art", "cambridge", "street", "places", "bridge", 
                                        "place", "museums", "museum", "fun", "college", "trinity",
                                        "free", "boat", "gallery", "aylesbray lodge guest house", 
                                        "church's", "milton", "funky", "nusha"]
                            
                        for value in poss_value:
                            label_value = value
                            
                            if re.search(r"[^a-z]"+value+r"[^a-z]", user_utt):
                                if value not in weird_val_list \
                                    and value not in other_name_val \
                                    and "taxi" not in user_utt:
                                    # # work for 8
                                    pass
                                    flag = 1
                                    self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                        if flag == 0:
                            for value in poss_value:
                                label_value = value

                                if re.search(r"[^a-z]"+value+r"[^a-z]", sys_utt):
                                    if value not in user_utt \
                                        and value not in weird_val_list \
                                        and "taxi" not in sys_utt \
                                        and value not in other_name_val:
                                        # work for 141
                                        self.update_tmp_log(idx, turn, last_turn, domain, slot_type, label_value)
                                        pass

            self._save_tmp_log()

    def token_distance(self, s, w1, w2):  
      
        if w1 == w2 : 
            return 0
        # get individual words in a list 
        s = re.sub("n't ", " not ", s)
        s = re.sub(r"\.|,|\?", " ", s)
        words = s.split(" ") 
        # assume total length of the string as 
        # minimum distance 
        min_dist = len(words)+1 
    
        # traverse through the entire string 
        for index in range(len(words)): 
    
            if words[index] == w1: 
                for search in range(len(words)): 
    
                    if words[search] == w2:  
    
                        # the distance between the words is 
                        # the index of the first word - the  
                        # current word index  
                        curr = abs(index - search) - 1; 

                        # comparing current distance with  
                        # the previously assumed distance 
                        if curr < min_dist: 
                            min_dist = curr 
  
        # w1 and w2 are same and adjacent 
        return min_dist 
    
    def _load_otgy(self):
        self.otgy = self._load_json(self.otgy_path)

    def update_tmp_log(self, idx, turn, last_turn, domain, slot_type, label_value):
        
        if idx not in self.tmp_log:
            self.tmp_log[idx] = turn.copy()
            del self.tmp_log[idx]["domains"]
            self.tmp_log[idx]["context"] = last_turn

            self.tmp_log[idx]["slots"] = ", ".join([" ".join(slot) for slot in self.tmp_log[idx]["slots"]])
        
            self.tmp_log[idx]["add_slot"] = f"{domain} {slot_type} {label_value}"
        else:
            exist_val_list = self.tmp_log[idx]["add_slot"].split(" | ")
            exist_val_list[0] = " ".join(exist_val_list[0].split(" ")[2:])
            remove_set = set()

            label_value_tmp = label_value
            if label_value.startswith("the "):
                label_value_tmp = "".join(label_value[4:])

            for exist_val in exist_val_list:
                exist_val_tmp = exist_val
                if exist_val.startswith("the "):
                    exist_val_tmp = "".join(exist_val[4:])
                if exist_val_tmp in label_value \
                    and exist_val_tmp != label_value \
                    and exist_val_tmp != label_value_tmp:
                    remove_set.add(exist_val)

            exist_val_list = list(set(exist_val_list) - remove_set)
            if label_value_tmp not in self.tmp_log[idx]["add_slot"]:
                exist_val_list.append(label_value)
                
            # order as in context
            if len(exist_val_list) > 1:
                val_idx_dict = {}
                for value in exist_val_list:
                    val_idx_dict[value] = last_turn.find(value)
                
                val_idx_dict = OrderedDict(
                    sorted(val_idx_dict.items(), key=lambda t: t[1])
                )

                exist_val_list = list(val_idx_dict.keys())
                # pdb.set_trace()


            self.tmp_log[idx]["add_slot"] = f"{domain} {slot_type} "+ " | ".join(exist_val_list)

    def analyze(self):
        """
        analyze results
        """
        self._load_data()

        # basic info
        count_basic_info = self.Count_Basic_Info()
        self.update_results(key_="count_basic_info", value_=count_basic_info)

        # refer info
        count_refer_info = self.Count_Refer_Info()
        self.update_results(key_="count_refer_info", value_=count_refer_info)

        # # name info
        # count_name_info = self.Count_Name_Info_old()
        # self.update_results(key_="count_name_info", value_=count_name_info)

        # self.Create_OTGY_M22()

        # self.Analyze_Bias()

        # self.Search_Name_Entity()

        # # save tmp log
        # self._save_tmp_log()

        # self.Count_Type_Info()

        # self.Count_Dest_Depa_Info()

        # self.Count_Area_Info()

        # self.Count_Price_Info()

        # self.Count_Food_Info()

        # self.Count_Parking_Info()

        # self.Count_Internet_Info()
        
        # self.Count_Star_Info()

        self.Count_Name_Info()



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
    analyze_multiwoz = Analyze_Multiwoz()
    analyze_multiwoz.analyze()


if __name__ == "__main__":
    main()
