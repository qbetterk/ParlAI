#!/usr/bin/env python3
#
import os, sys, json
import pdb
from collections import defaultdict, OrderedDict

DOMAINS    = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]
SLOT_TYPES = ["stay", "price", "addr",  "type", "arrive", "day", "depart", "dest",
            "area", "leave", "stars", "department", "people", "time", "food", 
            "post", "phone", "name", 'internet', 'parking',
            'book stay', 'book people','book time', 'book day',
            'pricerange', 'destination', 'leaveat', 'arriveby', 'departure']

def _load_json(file_path):
    with open(file_path) as df:
        data = json.loads(df.read().lower())
    return data

def _extract_slot_from_string(slots_string):
    """
    Either ground truth or generated result should be in the format:
    "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
    and this function would reformat the string into list:
    [[dom, slot_type, slot_val], ... ]
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
                slot_type = slot[1]+" "+slot[2]
                slot_val  = " ".join(slot[3:])
            else:
                slot_type = slot[1]
                slot_val  = " ".join(slot[2:])
            slots_list.append([domain, slot_type, slot_val])
        else:
            pass
            # pmul4204.json-5
            # in slots "taxi arriveby 16:15" --> "taxi arriveby 16,15"
    return slots_list

def main():
    # train_data_path = "./data/multiwoz_dst/MULTIWOZ2.1/data_reformat_trade_turn_sa_ha_train.json"
    # valid_data_path = "./data/multiwoz_dst/MULTIWOZ2.1/data_reformat_trade_turn_sa_ha_valid.json"
    otgy = {}
    # for dt in ["train", "valid"]:
    for dt in ["test"]:
        data_path = "./data/multiwoz_dst/MULTIWOZ2.1/data_reformat_trade_turn_sa_ha_"+dt+".json"
        data = _load_json(data_path)

        for dial_id, turn in data.items():
            slots_list = _extract_slot_from_string(turn["slots_inf"])
            for slot in slots_list:
                [domain, slot_type, slot_val] = slot
                dom_type = domain+"--"+slot_type
                if dom_type not in otgy:
                    otgy[dom_type] = defaultdict(int)
                otgy[dom_type][slot_val] += 1

    for dom_type in otgy:
        otgy[dom_type] = OrderedDict(sorted(otgy[dom_type].items(), key=lambda t: t[1], reverse=True))

    otgy_path = "./data/multiwoz_dst/MULTIWOZ2.1/new_otgy.json"
    with open(otgy_path, "w") as tf:
        json.dump(otgy, tf, indent=2)





if __name__ == "__main__":
    main()