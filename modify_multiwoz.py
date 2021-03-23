#!/usr/bin/env python3
#
import os
import json, random, math, re
import pdb
from tqdm import tqdm
from collections import defaultdict, OrderedDict

DOMAINS = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train", "bus"]

#  # # # for trade data
SLOT_TYPES_v21 = [
    "day",
    "type",
    "area",
    "stars",
    "department",
    "food",
    "name",
    'internet',
    'parking',
    'pricerange',
    'destination',
    'leaveat',
    'arriveby',
    'departure',
    'book stay',
    'book people',
    'book time',
    'book day',
]
SLOT_TYPES_v22 = [
    "day",
    "type",
    "area",
    "stars",
    "department",
    "food",
    "name",
    'internet',
    'parking',
    'pricerange',
    'destination',
    'leaveat',
    'arriveby',
    'departure',
    'stay',
    'people',
    'time',
    'day',
]

BOOK_TYPE = [
    'stay',
    'people',
    'time',
    'day',
]


random.seed(0)


class Modify_Multiwoz(object):
    def __init__(self, data_path=None, mode="train"):
        self.mode = mode
        self.old_data_dir = "./data/multiwoz_dst/MULTIWOZ2.2"
        self.data_path = (
            f"{self.old_data_dir}/data_reformat_{self.mode}.json"
        )
        # accumulated slots and dialog history
        self.new_data_dir = "./data/multiwoz_dst/MULTIWOZ2.2+"
        if not os.path.exists(self.new_data_dir):
            os.mkdir(self.new_data_dir)

        self.new_data_path = (
            f"{self.new_data_dir}/modify_data_reformat_{self.mode}.json"
        )

        # non-accumulated slots
        self.new_data_path2 = (
            f"{self.new_data_dir}/modify_sig_data_reformat_{self.mode}.json"
        )

        self.otgy_path = f"{self.old_data_dir}/otgy.json"



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

    def modify_multiwoz(self):
        total_state = {}
        for mode in ["train", "valid", "test"]:
            self.mode = mode
            self.data_path = (
                f"{self.old_data_dir}/data_reformat_{self.mode}.json"
            )
            self.new_data_path = (
                f"{self.old_data_dir}/modify_data_reformat_{self.mode}.json"
            )
            # load data from 2.2
            self._load_data()

            # load modifcation
            modify_dialog_path_list = []
            for domain in DOMAINS:
                for slot_type in SLOT_TYPES_v22:
                    modify_dialog_path = f"{self.old_data_dir}/data_reformat_{self.mode}_{domain}_{slot_type}.json"
                    if os.path.exists(modify_dialog_path):
                        modify_dialog_path_list.append(modify_dialog_path)

            for modify_dialog_path in modify_dialog_path_list:
                modify_dialog = self._load_json(modify_dialog_path)
                for idx, turn in modify_dialog.items():
                    new_slot = turn["add_slot"]
                    if len(new_slot.split()) >= 3:
                        new_slot_list = new_slot.split()
                        domain, slot_type, slot_val = new_slot_list[0], new_slot_list[1], " ".join(new_slot_list[2:])
                        if f"{domain}_{slot_type}" in self.data[idx]["slots_inf"]:
                            pdb.set_trace()
                        if slot_val:
                            self.data[idx]["slots_inf"] = self.data[idx]["slots_inf"] + f" {new_slot},"
                            self.data[idx]["slots_err"] = self.data[idx]["slots_err"] + f" {new_slot},"

            # # # count maximum turn number for each dialog
            dial_max_turn_dict = {}
            for idx, turn in self.data.items():
                if turn["dial_id"] not in dial_max_turn_dict:
                    dial_max_turn_dict[turn["dial_id"]] = turn["turn_num"]
                if turn["turn_num"] > dial_max_turn_dict[turn["dial_id"]]:
                    dial_max_turn_dict[turn["dial_id"]] = turn["turn_num"]

            # # # copy added slots to the future turns
            for dial_id, max_turn_num in dial_max_turn_dict.items():
                bspan = {}
                for turn_num in range(max_turn_num+1):
                    idx = f"{dial_id}-{turn_num}"
                    # if idx == "mul0237.json-4":
                    #     pdb.set_trace()
                    if idx not in self.data:
                        continue
                    # slot_list = [slot.strip() for slot in self.data[idx]["slots_inf"].split(", ") if len(slot.split()) > 2]
                    for slot in self.data[idx]["slots_inf"].split(","):
                        slot = slot.strip()
                        if len(slot.split()) < 3:
                            continue
                        domain, slot_type, slot_val = slot.split()[0], slot.split()[1], " ".join(slot.split()[2:])
                        if domain not in DOMAINS or slot_type not in SLOT_TYPES_v22:
                            pdb.set_trace()
                            continue
                        dom_type = f"{domain}_{slot_type}"
                        # update slot value
                        if dom_type not in bspan:
                            bspan[dom_type] = slot_val
                        if bspan[dom_type] != slot_val and slot_val != "dontcare":
                            del bspan[dom_type]
                            bspan[dom_type] = slot_val
                    # if idx == "mul0237.json-4":
                    #     pdb.set_trace()
                    #     continue
                    # generate slots
                    new_slot_list = []
                    for dom_type, slot_val in bspan.items():
                        domain, slot_type = dom_type.split("_")
                        new_slot_list.append(f"{domain} {slot_type} {slot_val}")
                    self.data[idx]["slots_inf"] = ", ".join(new_slot_list)

            # # # copy added slots to the future turns
            for dial_id, max_turn_num in dial_max_turn_dict.items():
                bspan = {}
                for turn_num in range(max_turn_num+1):
                    idx = f"{dial_id}-{turn_num}"
                    # if idx == "mul0237.json-4":
                    #     pdb.set_trace()
                    if idx not in self.data:
                        continue
                    # slot_list = [slot.strip() for slot in self.data[idx]["slots_inf"].split(", ") if len(slot.split()) > 2]
                    for slot in self.data[idx]["slots_err"].split(","):
                        slot = slot.strip()
                        if len(slot.split()) < 3:
                            continue
                        domain, slot_type, slot_val = slot.split()[0], slot.split()[1], " ".join(slot.split()[2:])
                        if domain not in DOMAINS or slot_type not in SLOT_TYPES_v22:
                            pdb.set_trace()
                            continue
                        dom_type = f"{domain}_{slot_type}"
                        # update slot value
                        if dom_type not in bspan:
                            bspan[dom_type] = slot_val
                        if bspan[dom_type] != slot_val and slot_val != "dontcare":
                            del bspan[dom_type]
                            bspan[dom_type] = slot_val
                    # if idx == "mul0237.json-4":
                    #     pdb.set_trace()
                    #     continue
                    # generate slots
                    new_slot_list = []
                    for dom_type, slot_val in bspan.items():
                        domain, slot_type = dom_type.split("_")
                        new_slot_list.append(f"{domain} {slot_type} {slot_val}")
                    self.data[idx]["slots_addup"] = ", ".join(new_slot_list)
            # # # revert it back to the non-accumulated version
            # # # revert it back to multiwoz version, like data.json

            with open(self.new_data_path, "w+") as tf:
                json.dump(self.data, tf, indent=2)

    def compare_before_after_modify(self):
        state = {}
        for mode in ["train", "valid", "test"]:
            self.mode = mode
            self.data_path = (
                f"{self.old_data_dir}/data_reformat_{self.mode}.json"
            )

            self.new_data_path = (
                f"{self.new_data_dir}/modify_data_reformat_{self.mode}.json"
            )
            # load data from 2.2
            self._load_data()
            # load modified data
            self.new_data = self._load_json(self.new_data_path)

            # reformat original data
            # # # count maximum turn number for each dialog
            dial_max_turn_dict = {}
            for idx, turn in self.data.items():
                if turn["dial_id"] not in dial_max_turn_dict:
                    dial_max_turn_dict[turn["dial_id"]] = turn["turn_num"]
                if turn["turn_num"] > dial_max_turn_dict[turn["dial_id"]]:
                    dial_max_turn_dict[turn["dial_id"]] = turn["turn_num"]

            # # # copy added slots to the future turns
            for dial_id, max_turn_num in dial_max_turn_dict.items():
                bspan = {}
                for turn_num in range(max_turn_num+1):
                    idx = f"{dial_id}-{turn_num}"
                    if idx not in self.data:
                        continue
                    # slot_list = [slot.strip() for slot in self.data[idx]["slots_inf"].split(", ") if len(slot.split()) > 2]
                    for slot in self.data[idx]["slots_inf"].split(","):
                        slot = slot.strip()
                        if len(slot.split()) < 3:
                            continue
                        domain, slot_type, slot_val = slot.split()[0], slot.split()[1], " ".join(slot.split()[2:])
                        if domain not in DOMAINS or slot_type not in SLOT_TYPES_v22:
                            pdb.set_trace()
                            continue
                        dom_type = f"{domain}_{slot_type}"
                        # update slot value
                        if dom_type not in bspan:
                            bspan[dom_type] = slot_val
                        if bspan[dom_type] != slot_val:
                            del bspan[dom_type]
                            bspan[dom_type] = slot_val
                    # generate slots
                    new_slot_list = []
                    for dom_type, slot_val in bspan.items():
                        domain, slot_type = dom_type.split("_")
                        new_slot_list.append(f"{domain} {slot_type} {slot_val}")
                    self.data[idx]["slots_inf"] = ", ".join(new_slot_list)


            # for count turn related:
            num_idx_type = defaultdict(int)
            idx_dom      = defaultdict(set)
            idx_total    = set()

            # for count dialog related
            dial_id_type  = defaultdict(set)
            dial_id_dom   = defaultdict(set)
            dial_id_total = set()
            num_dial_type = {}

            # for counting total number
            total_idx_dom      = defaultdict(set)
            total_dial_id_dom  = defaultdict(set)
            total_num_idx_dom  = {}
            total_num_dial_dom = {}

            for idx, turn in self.new_data.items():
                if idx not in self.data:
                    pdb.set_trace()
                    continue
                old_slot_str = self.data[idx]["slots_inf"]
                old_slot_set = set([slot.strip() for slot in old_slot_str.split(",") if len(slot.split()) > 2])
                new_slot_str = turn["slots_inf"]
                new_slot_set = set([slot.strip() for slot in new_slot_str.split(",") if len(slot.split()) > 2])

                for slot in new_slot_set-old_slot_set:
                    domain, slot_type = slot.split()[0], slot.split()[1]
                    if slot_type in ['stay', 'people', 'time', 'day', 'leaveat', 'arriveby']:
                        continue
                    dom_type = f"{domain}_{slot_type}"
                    num_idx_type[dom_type] += 1
                    idx_dom[domain].add(idx)
                    dial_id_type[dom_type].add(turn["dial_id"])

                for domain in set([slot.split()[0] for slot in new_slot_set]):
                    if domain not in DOMAINS:
                        pdb.set_trace()
                        continue
                    total_idx_dom[domain].add(idx)
                    total_idx_dom["total"].add(idx)
                    total_dial_id_dom[domain].add(turn["dial_id"])
                    total_dial_id_dom["total"].add(turn["dial_id"])


            # for count dialog-related
            for dom_type in dial_id_type:
                domain, slot_type = dom_type.split("_")
                num_dial_type[dom_type] = len(dial_id_type[dom_type])
                dial_id_dom[domain] = dial_id_dom[domain].union(dial_id_type[dom_type])
                dial_id_total = dial_id_total.union(dial_id_type[dom_type])
            for domain in dial_id_dom:
                num_dial_type[f"{domain}_ztotal"] = len(dial_id_dom[domain])
                dial_id_total = dial_id_total.union(dial_id_dom[domain])
            num_dial_type = OrderedDict(sorted(num_dial_type.items(), key=lambda t: t[0]))  
            num_dial_type["total"] = len(dial_id_total)

            # for counting turn-related
            for domain in idx_dom:
                num_idx_type[f"{domain}_ztotal"] = len(idx_dom[domain])
                idx_total = idx_total.union(idx_dom[domain])
            num_idx_type = OrderedDict(sorted(num_idx_type.items(), key=lambda t: t[0]))  
            num_idx_type["total"] = len(idx_total)

            # for total
            for domain in total_dial_id_dom:
                total_num_dial_dom[domain] = len(total_dial_id_dom[domain])
            for domain in total_idx_dom:
                total_num_idx_dom[domain] = len(total_idx_dom[domain])

            # compute percentage
            num_dial_with_per = {}
            for dom_type, num in num_dial_type.items():
                domain = dom_type.split("_")[0]
                percentage = num / total_num_dial_dom[domain] * 100
                num_dial_with_per[dom_type] = "{:d} ({:.1f}%)".format(num, percentage)

            num_idx_with_per = {}
            for dom_type, num in num_idx_type.items():
                domain = dom_type.split("_")[0]
                percentage = num / total_num_idx_dom[domain] * 100
                num_idx_with_per[dom_type] = "{:d} ({:.2f}%)".format(num, percentage)


            state[self.mode] = {
                "num_dial_type"      : num_dial_type,
                "total_num_dial_dom" : total_num_dial_dom, 
                "num_dial_with_per"  : num_dial_with_per,
                "num_idx_type"       : num_idx_type,
                "total_num_idx_dom"  : total_num_idx_dom,
                "num_idx_with_per"   : num_idx_with_per,
            }
            
        with open(f"{self.new_data_dir}/modify_state.json", "w") as tf:
            json.dump(state, tf, indent=2)

    def convert_back_multiwoz(self):
        # modify based on data.json in 2.2, output data.json in 2.2+
        data_path = (
            f"{self.old_data_dir}/data.json"
        )
        new_data_path = (
            f"{self.new_data_dir}/data.json"
        )

        with open(data_path) as df:
            self.data = json.loads(df.read().lower())

        for mode in ["train", "valid", "test"]:
            self.mode = mode

            modified_data_path = (
                f"{self.old_data_dir}/modify_data_reformat_{self.mode}.json"
            )
            modified_data = self._load_json(modified_data_path)
            # load modifcation
            for idx, turn in tqdm(modified_data.items()):
                for slot in turn["slots_inf"].split(","):
                    if not slot:
                        continue
                    new_slot_list = slot.strip().split()
                    domain, slot_type, slot_val = new_slot_list[0], new_slot_list[1], " ".join(new_slot_list[2:])
                    turn_num = turn["turn_num"]
                    dial_id = turn["dial_id"]
                    idx = 2 * turn_num + 1
                    if not slot_val:
                        continue
                    if slot_type not in BOOK_TYPE:
                        if slot_type not in self.data[dial_id]["log"][idx]["metadata"][domain]["semi"]:
                            pdb.set_trace()
                        self.data[dial_id]["log"][idx]["metadata"][domain]["semi"][slot_type] = slot_val.split("|")
        
        self.new_data = {}
        for dial_id in tqdm(self.data):
            dial_id_cap = dial_id.split(".")[0].upper() + ".json"
            self.new_data[dial_id_cap] = self.data[dial_id].copy()

        with open(new_data_path, "w") as tf:
            json.dump(self.new_data, tf, indent=2)

    def convert_back_sgd(self):
        # convert modified result back to sgd format, still 2.2 version

        for mode in ["train", "dev", "test"]:
            self.mode = mode
            old_sub_dir = f"{self.old_data_dir}/{self.mode}"
            new_sub_dir = f"{self.new_data_dir}/{self.mode}"
            if not os.path.exists(new_sub_dir):
                os.makedirs(new_sub_dir)

            modified_data_path = (
                f"{self.old_data_dir}/modify_data_reformat_{self.mode}.json"
            )
            # valid subdir is named of "dev"
            if self.mode == "dev":
                modified_data_path = (
                    f"{self.old_data_dir}/modify_data_reformat_valid.json"
                )
            # load modified dialog
            modified_data = self._load_json(modified_data_path)

            # open each file
            for file_name in tqdm(os.listdir(old_sub_dir)):
                old_file_path = os.path.join(old_sub_dir, file_name)
                new_file_path = os.path.join(new_sub_dir, file_name)

                self.data = self._load_json(old_file_path)
                self.data_dict = {dial["dialogue_id"]: dial for dial in self.data}

                # load modifcation
                for idx, turn in modified_data.items():
                    # nothing to modify
                    if turn["slots_addup"] == "":
                        continue

                    turn_num = turn["turn_num"]
                    dial_id = turn["dial_id"]

                    if dial_id in self.data_dict:
                        old_turn = self.data_dict[dial_id]["turns"][2 * turn_num]
                        # newly added slot in this turn
                        for slot in turn["slots_err"].split(","):
                            if not slot:
                                continue
                            new_slot_list = slot.strip().split()
                            domain, slot_type, slot_val = new_slot_list[0], new_slot_list[1], " ".join(new_slot_list[2:])
                    
                            for frame in old_turn["frames"]:
                                if domain == frame["service"] and slot_val in old_turn["utterance"]:
                                    start = old_turn["utterance"].find(slot_val)
                                    end   = start + len(slot_val)
                                    dom_type = f"{domain}-{slot_type}"

                                    frame["slots"].append({
                                        "exclusive_end": end,
                                        "slot": dom_type,
                                        "start" : start,
                                        "value" : slot_val
                                        })

                        # accumulatively added slots
                        for slot in turn["slots_addup"].split(","):
                            if not slot:
                                continue
                            new_slot_list = slot.strip().split()
                            domain, slot_type, slot_val = new_slot_list[0], new_slot_list[1], " ".join(new_slot_list[2:])
                            
                            for frame in old_turn["frames"]:
                                if domain == frame["service"]:
                                    frame["state"]["slot_values"][f"{domain}-{slot_type}"] = slot_val.split("|")

                with open(new_file_path, "w") as tf:
                    json.dump(list(self.data_dict.values()), tf, indent=2)







    def sample_verification_dialog(self):
        self.verify_num = 100
        self.verify_data_path = f"{self.new_data_dir}/verification_{self.verify_num}.json"
        self.verify_data_unc_path = f"{self.new_data_dir}/verification_{self.verify_num}_unc.json"
        self.total_corrected = []
        self.total_uncorrected = []
        for mode in ["train", "valid"]:
            self.mode = mode
            # load modifcation
            modify_dialog_path_list = []
            for domain in DOMAINS:
                for slot_type in SLOT_TYPES_v22:
                    modify_dialog_path = f"{self.new_data_dir}/data_reformat_{self.mode}_{domain}_{slot_type}.json"
                    if os.path.exists(modify_dialog_path):
                        modify_dialog_path_list.append(modify_dialog_path)

            for modify_dialog_path in modify_dialog_path_list:
                modify_dialog = self._load_json(modify_dialog_path)
                self.total_corrected += list(modify_dialog.values())
            # get uncorrected ones
            self.data_path = (
                f"{self.new_data_dir}/data_reformat_{self.mode}.json"
            )
            self._load_data()
            for idx, turn in tqdm(self.data.items()):
                flag = 0
                for turn_cor in self.total_corrected:
                    if turn["dial_id"] == turn_cor["dial_id"] and turn["turn_num"] == turn_cor["turn_num"]:
                        flag = 1
                        break
                if flag == 0:
                    turn["context"] = turn["context"].split("<system>")[-1]
                    self.total_uncorrected.append(turn)

        
        print(len(self.total_corrected))
        print(len(self.total_uncorrected))
        
        # sample corrected one
        sample = random.choices(self.total_corrected, k=self.verify_num)
        # sample uncorrected one
        sample2 = random.choices(self.total_uncorrected, k=self.verify_num)

        with open(self.verify_data_path, "w") as tf:
            json.dump(sample2, tf, indent=2)






def main():
    modify = Modify_Multiwoz(mode="test")
    modify.modify_multiwoz()
    # modify.compare_before_after_modify()
    # modify.convert_back_multiwoz()
    modify.convert_back_sgd()
    # modify.sample_verification_dialog()

if __name__ == "__main__":
    main()