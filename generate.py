#!/usr/bin/env python3
#
import sys, os
import json
sys.path.append("../")

from Klickitat.package import KlickitatGenerator, KlickitatGrammarCollection
from templates import QBasics, ABasics_Level1, ABasics_Level2, ABasics_Level3
import numpy as np
import pdb
np.random.seed(0)
DOMAINS = ["hotel"]

class Generate(object):
    """docstring for Generate"""
    def __init__(self, arg=None):
        super(Generate, self).__init__()
        self.db_dir = "data/multiwoz_dst/MULTIWOZ2.1/"
        self.domain = "hotel"
        self.db_path = os.path.join(self.db_dir, f"{self.domain}_db.json")

        self.target_dir = "data/disambiguation/"
    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def extract_entity(self, path=None):
        if path is None:
            path = self.db_path
        self.db_data = self._load_json(path)
        self.db_name = [item["name"] for item in self.db_data]

    # # # build generators
    def build_generator(self, level=1):
        # load entities from database
        self.extract_entity()
        # sample candidates
        cands = np.random.choice(self.db_name, size=5)
        # # # build binging 
        self.bind = {
            "DOMAIN": [self.domain], 
            "A": [cands[0]],
            "B": [cands[1]],
            "C": [cands[2]],
            "D": [cands[3]],
            "E": [cands[4]]}
        # self.bind = {"DOMAIN": bind_domain, "NAME": bind_db_name}
        # # # generator for questions
        self.gen_q = KlickitatGenerator(QBasics.combined_grammar, 
                            binding=self.bind, linter="not-strict")
        # # # generator for answers
        if level == 1:
            self.gen_a = KlickitatGenerator(ABasics_Level1.combined_grammar, 
                                binding=self.bind, linter="not-strict")
        elif level == 2:
            self.gen_a = KlickitatGenerator(ABasics_Level2.combined_grammar, 
                                binding=self.bind, linter="not-strict")
        elif level == 3:
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
    def generate_data(self, path=None, data_size=100):
        # check path
        if dir_path is None:
            dir_path = self.target_dir
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.target_file_path = os.path.join(dir_path, "data.json")

        data = []
        for idx in range(data_size):
            self.domain = np.random.choice(DOMAINS)
            self.build_generator()
            sys_utt = self.replace_punc(self.gen_q.generate_utterance(root="ROOT"))
            usr_utt = self.replace_punc(self.gen_a.generate_utterance(root="ROOT"))
            pdb.set_trace()
            data.append({
                "system" : sys_utt,
                "user" : usr_utt,
                "output" : ""
                })





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
    gen.test(5)

if __name__ == "__main__":
    main()


