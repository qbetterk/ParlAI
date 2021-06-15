#!/usr/bin/env python3
#
import os, sys, json
import pdb
import re, argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import random

"""
This script if for analyze and extract the ambiguation existing in the current 
dialog dataset: MultiWOZ, SGD, and SIMMC.
Stage 1: care only for the ambiguation in the following format:
    ---
    Sstem: there are three available results, A, B and C (listing multiple results)
    User: I would like B / the second one/ the cheaper one/ any one/ none of them
    ---
    Target: DST for the user response --> disambiguation

"""
class AnalyzeMultiWOZ(object):
    """docstring for AnalyzeMultiWOZ"""
    def __init__(self, args):
        super(AnalyzeMultiWOZ, self).__init__()
        self.args = args
        self.data_dir = "data/multiwoz_dst/MULTIWOZ2.2"
        
    def _load_txt(self, file_path):
        with open(file_path) as df:
            data = df.read().lower().split("\n")
            data.remove('')
        return data


    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data


    def _load_data(self, dir_path):
        """
        SGD-format dialog are separated into files by services (doamins).
        We need to load them all in one dir
        """
        data = []
        for file_name in tqdm(os.listdir(dir_path)):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                data += self._load_json(file_path)
        return data


    def analyze(self, mode="test"):
        # load data
        data_folder = os.path.join(self.data_dir, mode)
        self.data = self._load_data(data_folder)

        random.shuffle(self.data)
        # look into data
        for dial in self.data:
            for turn in dial["turns"]:
                for frame in turn["frames"]:
                    pdb.set_trace()
                    # if frame.get("service_results") is not None:
                    #     print(turn["speaker"])
                    #     print(turn["utterance"])
                    #     print("*"*10)
                    #     pdb.set_trace()

                # if len(turn["frames"]) > 1:
                #     pdb.set_trace()


class AnalyzeSGD(object):
    """docstring for AnalyzeSGD"""
    def __init__(self, args):
        super(AnalyzeSGD, self).__init__()
        self.args = args
        self.data_dir = self.args.data_dir
        self.stats = {}


    def _load_txt(self, file_path):
        with open(file_path) as df:
            data = df.read().lower().split("\n")
            data.remove('')
        return data


    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data


    def _load_data(self, dir_path):
        """
        SGD dialog are separated into files by services (doamins).
        We need to load them all in one dir
        """
        data = []
        for file_name in tqdm(os.listdir(dir_path)):
            if not file_name.startswith("dialog"):
                continue
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                data += self._load_json(file_path)
        return data


    def _update_stats(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.data_dir, "stats.json")
        with open(save_path, "w+") as tf:
            json.dump(self.stats, tf, indent=2)

    def _pd_csv(self, data, path=None):
        if path is None:
            path = os.path.join(self.data_dir, "stats.csv")
        df = pd.DataFrame.from_dict(data, orient='index', columns=list(data.values())[0].keys())
        df.to_csv(path, index=True)


    def _save_csv(self, data, path=None):
        with open(path, "w") as csvfile:
            wirter = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writeheader()
            wirter.wirterows(data)

    def analyze(self, mode="dev"):
        # load data
        data_folder = os.path.join(self.data_dir, mode)
        self.data = self._load_data(data_folder)

        # random.shuffle(self.data)
        # look into data
        service_count = {
            "dial":{},
            "turn":{},
        }
        # for each dialog
        for dial in self.data:
            rel_dom_multi, rel_dom_total = set(), set()
            # for each turn
            for turn in dial["turns"]:
                # for each service
                for frame in turn["frames"]:
                    if frame["service"] not in service_count["turn"]:
                        service_count["turn"][frame["service"]] = {
                            "multi" : 0,
                            "total" : 0,
                        }

                    if frame.get("service_results") is not None and len(frame["service_results"]) > 1:
                        service_count["turn"][frame["service"]]["multi"] += 1
                        rel_dom_multi.add(frame["service"])

                    rel_dom_total.add(frame["service"])
                service_count["turn"][frame["service"]]["total"] += 1
            # count total dial num
            if rel_dom_total:                              
                for dom in rel_dom_total:
                    if dom not in service_count["dial"]:
                        service_count["dial"][dom] = {
                            "multi" : 0,
                            "total" : 0,
                        }
                    service_count["dial"][dom]["total"] += 1
            # count dial num with multi results
            if rel_dom_multi:
                for dom in rel_dom_multi:
                    service_count["dial"][dom]["multi"] += 1

        service_count["turn"] = OrderedDict(
            sorted(service_count["turn"].items(), key=lambda t: t[0])
        )
        service_count["dial"] = OrderedDict(
            sorted(service_count["dial"].items(), key=lambda t: t[0])
        )

        csv_path = os.path.join(self.data_dir, f"stats_{mode}_turn.csv")
        self._pd_csv(service_count["turn"], path=csv_path)
        csv_path = os.path.join(self.data_dir, f"stats_{mode}_dial.csv")
        self._pd_csv(service_count["dial"], path=csv_path)

        self.stats.update(service_count)
        self._update_stats()


class AnalyzeSIMMC(AnalyzeSGD):
    """docstring for AnalyzeSIMMC"""
    def __init__(self, args):
        super(AnalyzeSIMMC, self).__init__(args)
        self.args = args
        self.data_dir = "./data/simmc/"
        self.stats = {}

    def analyze(self, mode="merged"):
        file_path = os.path.join(self.data_dir, f"simmc2_dials_{mode}.json")
        self.data = self._load_json(file_path)
        service_count = {
            "dial":{},
            "turn":{},
        }
        question_templates = set()
        answer_templates = set()
        for dial in self.data.get("dialogue_data"):
            # init
            dial_id = dial["dialogue_idx"]
            domain = dial["domain"]
            if service_count["turn"].get(domain) is None:
                service_count["turn"][domain] = {
                    "sys" : 0,
                    "user" : 0,
                    "match" : 0,
                    "total" : 0,
                }
            if service_count["dial"].get(domain) is None:
                service_count["dial"][domain] = {
                    "sys" : 0,
                    "user" : 0,
                    "match" : 0,
                    "total" : 0,
                }
            flag_sys, flag_user, flag_match = 0, 0, 0

            # count for turn
            prev_turn = None
            for turn in dial["dialogue"]:
                turn_num = turn["turn_idx"]
                sys_utt = turn["system_transcript"]
                usr_utt = turn["transcript"]
                service_count["turn"][domain]["total"] += 1
                if turn["system_transcript_annotated"].get("act") == "REQUEST:DISAMBIGUATE".lower():
                    service_count["turn"][domain]["sys"] += 1
                    flag_sys = turn_num
                    question_templates.add(sys_utt)
                if turn["transcript_annotated"].get("act") == "INFORM:DISAMBIGUATE".lower():
                    service_count["turn"][domain]["user"] += 1
                    flag_user = 1
                    answer_templates.add(usr_utt)
                    if prev_turn["system_transcript_annotated"].get("act") == "REQUEST:DISAMBIGUATE".lower():
                        service_count["turn"][domain]["match"] += 1
                        flag_match = 1

                prev_turn = turn

            # count for dialog
            if flag_sys:
                service_count["dial"][domain]["sys"] += 1
            if flag_user:
                service_count["dial"][domain]["user"] += 1
            if flag_match:
                service_count["dial"][domain]["match"] += 1
            service_count["dial"][domain]["total"] += 1

        print(service_count["dial"])
        print(len(question_templates))

        service_count["turn"] = OrderedDict(
            sorted(service_count["turn"].items(), key=lambda t: t[0])
        )
        service_count["dial"] = OrderedDict(
            sorted(service_count["dial"].items(), key=lambda t: t[0])
        )

        # save count numbers
        csv_path = os.path.join(self.data_dir, f"stats_{mode}_turn.csv")
        self._pd_csv(service_count["turn"], path=csv_path)
        csv_path = os.path.join(self.data_dir, f"stats_{mode}_dial.csv")
        self._pd_csv(service_count["dial"], path=csv_path)
        # save count details
        self.stats.update(service_count)
        self._update_stats()
        # save templates
        with open(os.path.join(self.data_dir, f"question_templates_{mode}.json"), "w+") as tf:
            json.dump(sorted(list(question_templates)), tf, indent=2)
        with open(os.path.join(self.data_dir, f"answer_templates_{mode}.json"), "w+") as tf:
            json.dump(sorted(list(answer_templates)), tf, indent=2)

def Parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",   default="sgd")
    parser.add_argument(      "--data_dir", default="./data/google_sgd_dst/")
    args = parser.parse_args()
    return args


def main():
    args = Parse_args()

    # SGD = AnalyzeSGD(args)
    # SGD = AnalyzeMultiWOZ(args)
    SGD = AnalyzeSIMMC(args)
    SGD.analyze()
    # pass


if __name__ == "__main__":
    main()