#!/usr/bin/env python3
#
import sys, os
import json
sys.path.append("../")
import numpy as np
import pdb
from tqdm import tqdm

from Klickitat.package import KlickitatGenerator, KlickitatGrammarCollection
from templates import QBasics, ABasics, ABasics_Level1, ABasics_Level2, ABasics_Level3

np.random.seed(0)
DOMAINS = ["hotel", "restaurant", "attraction"]

class Generate(object):
    """docstring for Generate"""
    def __init__(self, data_dir=None):
        super(Generate, self).__init__()
        self.db_dir = "data/multiwoz_dst/MULTIWOZ2.1/"
        if data_dir is None:
            self.target_dir = "data/disambiguation/"
        else:
            self.target_dir = data_dir

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def extract_entity(self, path=None):
        self.db_data, self.db_name = {}, {}
        for domain in DOMAINS:
            path = os.path.join(self.db_dir, f"{domain}_db.json")
            self.db_data[domain] = self._load_json(path)
            self.db_name[domain] = [item["name"] for item in self.db_data[domain]]

    # # # build generators
    def build_generator(self, domain, level="1", cands=None, index=None):
        # # # build binging 
        self.bind = {
            "DOMAIN": [domain], 
            "A": [cands[0]],
            "B": [cands[1]],
            "C": [cands[2]],
            }
        positions = ["first", "second", "third"]
        # # # generator for questions
        self.gen_q = KlickitatGenerator(QBasics.combined_grammar, 
                            binding=self.bind, linter="not-strict")
        # # # generator for answers
        if level == "1":
            self.gen_a = KlickitatGenerator(ABasics.combined_grammar, 
                                binding={"OBJECT":[f"the {cands[index]}"]}, linter="not-strict")
        elif level == "2":
            self.gen_a = KlickitatGenerator(ABasics.combined_grammar, 
                                binding={"OBJECT":[f"the {positions[index]} one"]}, linter="not-strict")
        elif level == "3":
            self.gen_a = KlickitatGenerator(ABasics_Level3.combined_grammar, 
                                binding=self.bind, linter="not-strict")

    def replace_punc(self, sent):
        """
        Some denotions:
            CM: COMMA ,
            QM: QUESTION MARK ?
            PD: PERIOD .
            _: '
        """
        mapping = {
            " CM" : " ,",
            " QM" : " ?",
            " PD" : " .",
            "_"  : "'"
        }
        for rep, punc in mapping.items():
            sent = sent.replace(rep, punc)
        return sent

    # # # generate data
    def generate_data(self, filename=None, level="1", data_size=100):
        # check path
        if filename is None:
            filename = f"data_level{level}.json"
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
        self.target_file_path = os.path.join(self.target_dir, filename)

        # initialization
        data = []
        # load entities from database
        self.extract_entity()

        for idx in tqdm(range(data_size)):
            # randomize
            domain = np.random.choice(DOMAINS)
            cands = np.random.choice(self.db_name[domain], size=3, replace=False)
            index = np.random.choice(len(cands))
            level_ = np.random.choice(list(level))

            # generate
            self.build_generator(domain=domain, level=level_, cands=cands, index=index)

            for i in range(12): # train:dev:test=10:1:1
                sys_utt = self.replace_punc(self.gen_q.generate_utterance(root="ROOT"))
                usr_utt = self.replace_punc(self.gen_a.generate_utterance(root="ROOT"))
                data.append({
                    "system" : sys_utt,
                    "user"   : usr_utt,
                    "output" : cands[index],
                    "domain" : domain
                    })

        # save
        np.random.shuffle(data)
        with open(self.target_file_path, "w+") as tf:
            json.dump(data, tf, indent=2)
        last_train_idx, last_valid_idx = len(data) * 10 // 12, len(data) * 11 // 12

        train, valid, test = data[:last_train_idx], data[last_train_idx : last_valid_idx], data[last_valid_idx:]
        with open(self.target_file_path.replace(".json", "_train.json"), "w+") as tf:
            json.dump(train, tf, indent=2)
        with open(self.target_file_path.replace(".json", "_valid.json"), "w+") as tf:
            json.dump(valid, tf, indent=2)
        with open(self.target_file_path.replace(".json", "_test.json"), "w+") as tf:
            json.dump(test, tf, indent=2)




    # # # print
    def test(self, count=10):
        self.build_generator()
        for utterance in [self.gen_q.generate_utterance(root="ROOT") for __i in range(count)]:
            print(self.replace_punc(utterance))
        print("%"*30)
        for utterance in [self.gen_a.generate_utterance(root="ROOT") for __i in range(count)]:
            print(utterance)

def main():
    gen = Generate()
    # gen.generate_data(filename="data_test.json", data_size=100)
    gen.generate_data(level="12", data_size=100)
    # gen.generate_data(level="2", data_size=10000)

if __name__ == "__main__":
    main()


