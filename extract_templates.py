#!/usr/bin/env python3
#
import sys, os
import json, re
import pdb

OBJECT = ["shirts", "sofas", "sofa chairs", "rug", "sweater", "blouses", "jacket", "shelf"]

class ExtractTemplate(object):
    """docstring for ExtractTemplate"""
    def __init__(self, arg=None):
        super(ExtractTemplate, self).__init__()
        self.arg = arg
        self.data_dir = "data/simmc/"
        
    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def extract_q(self, path=None):
        if path is None:
            path = os.path.join(self.data_dir, "question_templates_merged.json")
        q_temps = self._load_json(path)

        left_q_list, object_list = set(), set()
        right_q_list = []
        for question in q_temps:
            # complete missing puncturation
            if question[-1] not in [".","?"]: # missing puncturation
                if question.startswith("which") or question.startswith("what"):
                    question += "?"
                else:
                    question += "."
            q_list = question.split()
            # pdb.set_trace()
            # delexilize
            if "which" in q_list:
                w_idx = q_list.index("which")
                if len(q_list) == w_idx + 2: # ... which item?
                    object_list.add(q_list[w_idx+1][:-1])
                    q_list[w_idx+1] = "[value]?"
                    left_q_list.add(" ".join(q_list))
                elif q_list[w_idx+2] in ["are", "were", "would", "did", "do", "you", "you're", "you'd"]: # which item are you ... /which item you ...
                    object_list.add(q_list[w_idx+1])
                    q_list[w_idx+1] = "[value]"
                    left_q_list.add(" ".join(q_list))
                elif len(q_list) > w_idx + 3 and q_list[w_idx+3] in ["are", "would", "do", "you", "you're", "you'd"]:
                    right_q_list.append(question)

        print(len(q_temps))
        print(sum(["which" in temp or "what" in temp for temp in q_temps]))
        print(object_list)



        with open(os.path.join(self.data_dir, f"question_templates_left.json"), "w+") as tf:
            json.dump(sorted(list(left_q_list)), tf, indent=2)

def main():
    extracttemplate = ExtractTemplate()
    extracttemplate.extract_q()

if __name__=="__main__":
    main()