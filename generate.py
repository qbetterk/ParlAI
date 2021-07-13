#!/usr/bin/env python3
#
import sys, os
import json
sys.path.append("../")
import numpy as np
import pdb
import random, string
from tqdm import tqdm

from Klickitat.package import KlickitatGenerator, KlickitatGrammarCollection
from templates import QBasics, ABasics, ABasics_Level1, ABasics_Level2, ABasics_Level3

random.seed(0)
np.random.seed(0)
DOMAINS = ["hotel", "restaurant", "attraction"]
TPYE_DICT = {
    "hotel" : ["area", "internet", "parking", "type", "stars", "pricerange"],
    "restaurant" : ["area", "pricerange", "food"],
    "attraction" : ["area", "type"]
}
TPYE_LIST = ["area", "internet", "parking", "type", "stars", "pricerange", "food"]
class Generate(object):
    """docstring for Generate"""
    def __init__(self, data_dir=None):
        super(Generate, self).__init__()
        self.db_dir = "data/multiwoz_dst/MULTIWOZ2.1/"
        if data_dir is None:
            self.target_dir = "data/disambiguation/"
        else:
            self.target_dir = data_dir
        self.alarm = False
        self.err = False

    def _load_json(self, file_path):
        with open(file_path) as df:
            data = json.loads(df.read().lower())
        return data

    def extract_entity(self, path=None):
        self.db_data, self.db_name = {}, {}
        for domain in DOMAINS:
            path = os.path.join(self.db_dir, f"{domain}_db.json")
            self.db_data[domain] = self._load_json(path)
            for item in self.db_data[domain]:
                if item["name"].startswith("the "):
                    item["name"] = item["name"][4:]
            self.db_name[domain] = [item["name"] for item in self.db_data[domain]]

    # # # build generators
    def build_generator(self, domain, level, cands=None, index=None):
        # # # generator for questions
        # build binging 
        self.bind = {
            "DOMAIN": [domain], 
            }
        tmp_list = ["A","B","C","D","E"]
        for idx, cand in enumerate(cands):
            cand_str = cand["name"] + " ["
            for att in cand:
                if att in TPYE_DICT[domain]:
                    cand_str += f" {att} : {cand[att]} , "
            cand_str = cand_str[:-2]
            cand_str += " ]"
            self.bind[tmp_list[idx]] = [cand_str]
        list_str = " CM ".join(tmp_list[:len(cands)-1])
        list_str += " or " + tmp_list[len(cands)-1]
        list_grammar = f"""
            LIST -> {list_str}
        """
        # if "acorn" in cands[2]["name"] and 2 in index:
        #     pdb.set_trace()
        self.gen_q = KlickitatGenerator(QBasics.combined_grammar + list_grammar, 
                            binding=self.bind, linter="not-strict")

        # pdb.set_trace()

        # # # generator for answers
        cands_num = len(cands)
        chose_num = len(index)
        if chose_num == 0:
            grammar_level0 = f"""
                ROOT -> none of them
            """
            self.gen_a = KlickitatGenerator(grammar_level0, linter="not-strict")

        elif level == "1":
            if chose_num == 1:
                name = cands[index[0]]["name"]
                if self.err:
                    name = self.import_error(domain, name)
                grammar_str = f" the {name}"
            elif cands_num == chose_num:
                grammar_str = " any one [of them]"
            elif chose_num == 2:
                name1, name2 = cands[index[0]]["name"], cands[index[1]]["name"]
                if self.err:
                    name1, name2 = self.import_error(domain, name1), self.import_error(domain, name2)
                grammar_str = f" the {name1} or the {name2}"
            else:
                raise ValueError("Should not choose so many options (>2)")

            grammar_level1 = f"""
                OBJECT -> {grammar_str}
            """
            self.gen_a = KlickitatGenerator(ABasics.combined_grammar + grammar_level1, linter="not-strict")
        
        elif level == "2":
            index.sort()
            positions = [["first"],
                         ["second"],
                         ["third"],
                         ["fourth"],
                         ["fifth"]]
            positions[cands_num-1].append("last")
            if cands_num / 2 != cands_num // 2:
                positions[(cands_num - 1)//2].append("middle")
            else:
                positions[cands_num-2].append("last but one")
            if chose_num == 1:
                posi_token = np.random.choice(positions[index[0]])
                grammar_str = f"the {posi_token} [one]"
            elif cands_num == chose_num:
                grammar_str = " any one [of them]"
            elif chose_num == 2:
                pos1, pos2 = np.random.choice(positions[index[0]]), np.random.choice(positions[index[1]])
                grammar_str = f"the {pos1} [one] or the {pos2} one"
                if pos1 == "last but one":
                    grammar_str = f"the {pos1} or the {pos2} (one|{domain})"
                elif pos2 == "last but one":
                    grammar_str = f"the {pos1} [one] or the last but one {domain}"
                if sum(index) == 1:
                    grammar_str += "| the first two"
                if sum(index) == cands_num *2 - 3:
                    grammar_str += "| the last two"
            else:
                raise ValueError("Should not choose so many options (>2)")

            grammar_level2 = f"""
                OBJECT -> {grammar_str}
            """
            # print(grammar_str)
            self.gen_a = KlickitatGenerator(ABasics.combined_grammar + grammar_level2, linter="not-strict")

        elif level == "3":
            """
            Hotel:
                area, internet, parking, 
                type, stars, pricerange
            restaurant:
                area, 
                pricerange, food, 
            attraction:
                area,
                type
            """
            slot_type, slot_value = self.find_value(domain=domain, cands=cands, index= index)
            if slot_value is None:
                level_new = np.random.choice(["1","2"])
                self.build_generator(domain=domain, level=level_new, cands=cands, index=index)
            elif chose_num == 2:
                if slot_value[0] == slot_value[1]:
                    # grammar_str = f"the {slot_value[0]} twos"
                    slot_value = slot_value[0]
                    grammar_str = {}
                    common_str = {
                        "area" : f"(these|those) (two | {domain}s | two {domain}s) [which are] in the {slot_value} [area | part of the city]",
                        "pricerange" : f"(these|those) ({slot_value} | {slot_value}ly-priced) (two | {domain}s | two {domain}s) | (these|those) (two | {domain}s | two {domain}s) with {slot_value} price",
                    }
                    if slot_type in ["internet", "parking"]:
                        slot_type = f"{slot_type}-{slot_value}"

                    hotel_type = "" if domain != "hotel" else cands[index[0]]["type"]
                    if domain != "hotel":
                        hotel_type = ""
                    elif cands[index[0]]["type"] == cands[index[1]]["type"]:
                        hotel_type = cands[index[0]]["type"]
                    else:
                        hotel_type = "hotel"

                    grammar_str["hotel"] = {
                        "area" : common_str["area"],
                        "internet-yes" : f"(these|those) (two | {hotel_type}s | two {hotel_type}s) [equiped] with (internet | wifi)",
                        "internet-no" : f"(these|those) (two | {hotel_type}s | two {hotel_type}s) (without | with no) (internet | wifi) ",
                        "parking-yes" : f"(these|those) (two | {hotel_type}s | two {hotel_type}s) (with a parking lot | that has a place to park)",
                        "parking-no" : f"(these|those) (two | {hotel_type}s | two {hotel_type}s) ((without a| with no) parking lot | that (has no | doesn't have a) place to park)",
                        "type" : f"(these|those) (two | {slot_value}s | two {slot_value}s)",
                        "stars" : f"(these|those) [two] {slot_value} star {hotel_type}s | (these|those) (two | {hotel_type}s | two {hotel_type}s) ((with | (that | which) have) a {slot_value} star rating | those that have {slot_value} star)",
                        "pricerange" : common_str["pricerange"],
                    }
                    grammar_str["restaurant"] = {
                        "area" : common_str["area"],
                        "pricerange" : common_str["pricerange"],
                        "food" : f"(these|those) {slot_value} (two | {domain}s | two {domain}s) | (these|those) (two | {domain}s | two {domain}s) (with | serving | which serve) the {slot_value} food"
                    }
                    grammar_str["attraction"] = {
                        "area" : common_str["area"],
                        "type" : f"(these|those) {slot_value} [two]"
                    }
                    grammar_level3 = f"""
                        OBJECT -> {grammar_str[domain][slot_type]}
                    """

                elif slot_value[0] != slot_value[1]:
                    # grammar_str = f"the {slot_value[0]} [one] (and | or) the {slot_value[1]} one"

                    def grammar_for_one(slot_value, slot_type, domain):
                        grammar_str = {}
                        common_str = {
                            "area" : f"the (one | {domain}) [which is] in the {slot_value} [area | part of the city]",
                            "pricerange" : f"the ({slot_value} | {slot_value}ly-priced) one | the {domain} with {slot_value} price",
                        }
                        if slot_type in ["internet", "parking"]:
                            slot_type = f"{slot_type}-{slot_value}"

                        hotel_type = "" if domain != "hotel" else cands[index[0]]["type"]

                        grammar_str["hotel"] = {
                            "area" : common_str["area"],
                            "type" : f"the {slot_value}",
                            "stars" : f"the {slot_value} star (one | {hotel_type}) | the (one| {hotel_type}) ((with | (that | which) has) a {slot_value} star rating | that has {slot_value} star)",
                            "pricerange" : common_str["pricerange"],
                        }
                        grammar_str["restaurant"] = {
                            "area" : common_str["area"],
                            "pricerange" : common_str["pricerange"],
                            "food" : f"(the | that) {slot_value} one | the restaurant (with | serving | which serves) {slot_value} food"
                        }
                        grammar_str["attraction"] = {
                            "area" : common_str["area"],
                            "type" : f"the {slot_value} [one]"
                        }
                        return grammar_str[domain][slot_type]

                    first_str = grammar_for_one(slot_value[0], slot_type, domain)
                    second_str = grammar_for_one(slot_value[1], slot_type, domain)
                    grammar_level3 = f"""
                        OBJECT -> {first_str} (and | or) {second_str}
                    """

                self.gen_a = KlickitatGenerator(ABasics.combined_grammar + grammar_level3, linter="not-strict")
            else:
                grammar_str = {}
                common_str = {
                    "area" : f"the (one | {domain}) [which is] in the {slot_value} [area | part of the city]",
                    "pricerange" : f"the ({slot_value} | {slot_value}ly-priced) one | the {domain} with {slot_value} price",
                }
                if slot_type in ["internet", "parking"]:
                    slot_type = f"{slot_type}-{slot_value}"

                hotel_type = "" if domain != "hotel" else cands[index[0]]["type"]

                grammar_str["hotel"] = {
                    "area" : common_str["area"],
                    "internet-yes" : f"the one [equiped] with (internet | wifi)",
                    "internet-no" : f"the one (without | with no) (internet | wifi) ",
                    "parking-yes" : f"the (one | {hotel_type}) (with a parking lot | that has a place to park)",
                    "parking-no" : f"the (one | {hotel_type}) ((without a| with no) parking lot | that (has no | doesn't have a) place to park)",
                    "type" : f"the {slot_value}",
                    "stars" : f"the {slot_value} star (one | {hotel_type}) | the (one| {hotel_type}) ((with | (that | which) has) a {slot_value} star rating | that has {slot_value} star)",
                    "pricerange" : common_str["pricerange"],
                }
                grammar_str["restaurant"] = {
                    "area" : common_str["area"],
                    "pricerange" : common_str["pricerange"],
                    "food" : f"(the | that) {slot_value} one | the restaurant (with | serving | which serves) {slot_value} food"
                }
                grammar_str["attraction"] = {
                    "area" : common_str["area"],
                    "type" : f"the {slot_value} [one]"
                }
                grammar_level3 = f"""
                    OBJECT -> {grammar_str[domain][slot_type]}
                """
                self.gen_a = KlickitatGenerator(ABasics.combined_grammar + grammar_level3, linter="not-strict")

    def find_value(self, domain, cands, index):
        """
        find distinct attributes to describe the target object(s)
        """
        cands_num = len(cands)
        chose_num = len(index)
        random.shuffle(TPYE_DICT[domain])

        if chose_num == 1:
            # identify with one attribute
            for slot_type in TPYE_DICT[domain]:
                value_set = set()
                for cand in cands:
                    if cand != cands[index[0]]:
                        value_set.add(cand[slot_type])
                if cands[index[0]][slot_type] not in value_set:
                    return slot_type, cands[index[0]][slot_type]

            # pdb.set_trace()

            return None, None
            # TODO: identify with two attributes

        elif chose_num == 2:
            # identify with one attribute:
            for slot_type in TPYE_DICT[domain]:
                value_set = set()
                for cand in cands:
                    if cand != cands[index[0]] and cand != cands[index[1]]:
                        value_set.add(cand[slot_type])
                if cands[index[0]][slot_type] not in value_set and cands[index[1]][slot_type] not in value_set:
                    if cands[index[0]][slot_type] != cands[index[1]][slot_type]:
                        # different values
                        return slot_type, [cands[index[0]][slot_type], cands[index[1]][slot_type]]
                    else:
                        # same value
                        return slot_type, [cands[index[0]][slot_type], cands[index[1]][slot_type]]

            return None, None

             # identify with two attribute:
        else:
            # TODO: choose multiple results
            return None, None

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

    def import_error(self, domain, name):
        if np.random.choice([0,1]):
            return name

        error_type = np.random.choice([0,1,2])
        if error_type == 0:
            # cut off suffix or prefix

            if name.startswith(domain):
                return name[len(domain):]
            elif name.endswith(domain):
                return name[:len(name)-len(domain)]
            elif name.endswith(" guesthouse"):
                return name[:len(name)-len("guesthouse")]
            elif name.endswith(" guest house"):
                return name[:len(name)-len("guest house")]
            elif name.endswith(" church"):
                return name[:len(name)-len("church")]
            elif name.endswith(" theatre"):
                return name[:len(name)-len("theatre")]
            elif name.endswith(" museum"):
                return name[:len(name)-len("museum")]
            elif name.endswith(" cinema"):
                return name[:len(name)-len("cinema")]
            else:
                return name

        elif error_type == 1:
            # replace guest house with hotel, and vice versa

            if name.endswith(" guest house"):
                return name[:len(name)-len("guest house")] + "hotel"
            elif name.endswith(" guesthouse"):
                return name[:len(name)-len("guesthouse")] + "hotel"
            elif name.endswith(" hotel"):
                return name[:len(name)-len("hotel")] + "guesthouse"
            else:
                return name

        elif error_type == 2:
            # spelling error 
            err_num = int(len(name) * random.uniform(0.01,0.2))
            replace_idx = random.sample(range(len(name)), k=err_num)
            name = list(name)
            for idx in replace_idx:
                if name[idx] == " ":
                    continue
                name[idx] = random.choice(string.ascii_lowercase)
            return "".join(name)




    # # # generate data
    def generate_data(self, filename=None, level="1", data_size=10000, multi=False, err=False):
        # check path
        if filename is None:
            filename = f"data_level{level}.json"
        if multi and "test" not in filename:
            filename = filename.replace(".json", "_multi.json")
        if err and "test" not in filename:
            filename = filename.replace(".json", "_err.json")
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
        self.target_file_path = os.path.join(self.target_dir, filename)

        if err:
            self.err = True
        # initialization
        data = []
        # load entities from database
        self.extract_entity()

        for idx in tqdm(range(data_size)):
            # randomize
            domain = np.random.choice(DOMAINS)
            # number of candidates can be 3,4 or 5
            cands_num = np.random.choice(range(4,6))
            cands = np.random.choice(self.db_data[domain], size=cands_num, replace=False)
            # number of options that the user choose, set to be under half of the options, 1 2
            chose_num = np.random.choice(range(2, cands_num // 2 + 1)) if multi else 1
            index = np.random.choice(cands_num, size=chose_num, replace=False)
            level_ = np.random.choice(list(level))
            # pdb.set_trace()
            # generate
            self.build_generator(domain=domain, level=level_, cands=cands, index=index)

            for i in range(12): # train:dev:test=10:1:1
                sys_utt = self.replace_punc(self.gen_q.generate_utterance(root="ROOT"))
                usr_utt = self.replace_punc(self.gen_a.generate_utterance(root="ROOT"))
                data.append({
                    "system" : sys_utt,
                    "user"   : usr_utt,
                    "output" : " , ".join([cands[idx]["name"] for idx in index]),
                    "domain" : domain,
                    # "cands_num" : int(cands_num),
                    # "index" : " ".join([str(idx) for idx in index])
                    })
                # if "last but one" in usr_utt:
                #     pdb.set_trace()
        # save
        np.random.shuffle(data)
        with open(self.target_file_path, "w+") as tf:
            json.dump(data, tf, indent=2)
        last_train_idx, last_valid_idx = len(data) * 10 // 12, len(data) * 11 // 12

        if "test" not in filename:
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
    gen.generate_data(filename="data_test.json", multi=True, err=False, level="3", data_size=10)
    # gen.generate_data(level="3")
    # gen.generate_data(level="123")
    # gen.generate_data(level="123", multi=True)
    # gen.generate_data(level="123", multi=False, err=True)
    # gen.generate_data(level="123", multi=True, err=True)

if __name__ == "__main__":
    main()


