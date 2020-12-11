#!/usr/bin/env python3
#
import os
import json
import re
import pdb
from collections import defaultdict, OrderedDict

DOMAINS = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]


class Compare_Multiwoz_Simpletod():
    def __init__(self):
        self.multiwoz_path = (
            "./data/multiwoz_dst/MULTIWOZ2.1/data_reformat_trade_turn_sa_ha_test.json"
        )
        self.simpletod = "../simpletod/resources/gpt2/test.history_belief"
        self.multiwoz_data = self._load_json(self.multiwoz_path)
        self.simpletod_data_raw = self._load_txt(self.simpletod)

    def _load_txt(self, file_path):
        with open(file_path) as df:
            data = df.read().lower().split("\n")
            data.remove('')
        return data

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def transfer_to_json(self):
        self.simpletod_data_json = {}
        self.rest = []
        for line in self.simpletod_data_raw:
            # reformat context
            context = line.split("<|endofcontext|>")[0].split("<|context|>")[1].strip()
            context = re.sub(r"<\|user\|>", "<user>", context)
            context = re.sub(r"<\|system\|>", "<system>", context)
            context = re.sub("are214star", "are 214 star", context) # for mul2206.json
            context = re.sub("are164star", "are 164 star", context) # for sng01534.json

            # reformat slots
            slots = line.split("<|endofbelief|>")[0].split("<|belief|>")[1].strip()

            # flag for match or not
            flag = 0
            for idx, turn in self.multiwoz_data.items():
                if turn["dial_id"] == "pmul2848.json":
                    turn["context"] = turn["context"].replace("cb5 8bs", "cb58bs")
                if turn["context"] == context:
                    self.simpletod_data_json[idx] = {
                        "turn_num": turn["turn_num"],
                        "dial_id": turn["dial_id"],
                        "slots": slots,
                        "context": context
                    }
                    flag = 1
                    del self.multiwoz_data[idx]
                    break
            if flag == 0:
                self.rest.append({
                    "context": context,
                    "slots"  : slots,
                })
                """
                mul1799.json
                """

        with open("../simpletod/resources/gpt2/test.history_belief.json", "w") as tf:
            json.dump(self.simpletod_data_json, tf, indent=2)
        with open("../simpletod/resources/gpt2/test.history_belief_rest.json", "w") as tf:
            json.dump(self.rest, tf, indent=2)

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
                    slots_list.append(domain+"--"+slot_type+"--"+slot_val)
                    # slots_list.append([domain, slot_type, slot_val])
        return slots_list

    def compare(self):
        self.simpletod_data = self._load_json("../simpletod/resources/gpt2/test.history_belief.json")
        print(len(self.multiwoz_data), len(self.simpletod_data))
        count_equal = 0
        for idx, turn in self.simpletod_data.items():
            simpletod_slots = self._extract_slot_from_string(turn["slots"])
            multiwoz_slots = self._extract_slot_from_string(self.multiwoz_data[idx]["slots_inf"])
            simpletod_slots_dense = []
            for slot in simpletod_slots:
                if slot.split("--")[2] != "not mentioned":
                    simpletod_slots_dense.append(slot)
            if set(simpletod_slots_dense) == set(multiwoz_slots):
                count_equal += 1
        print(count_equal)


def main():
    compare_multiwoz_simpletod = Compare_Multiwoz_Simpletod()
    # compare_multiwoz_simpletod.transfer_to_json()
    compare_multiwoz_simpletod.compare()


if __name__ == "__main__":
    main()