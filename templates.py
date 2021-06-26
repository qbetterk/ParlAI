#!/usr/bin/env python3
#
import sys
sys.path.append("../")
from Klickitat.package import KlickitatGenerator, KlickitatGrammarCollection

import numpy as np


def print_examples(grammar, count=10):
    g = KlickitatGenerator(grammar, linter="not-strict")
    for utterance in [g.generate_utterance(root="ROOT") for __i in range(count)]:
        print(utterance)


@KlickitatGrammarCollection
class QBasics:
    root = """
        ROOT -> [PREFIX] (SIMPLE | COMPLEX) CM A CM B CM or C QM
    """
    """
    Some denotions:
        CM: COMMA ,
        QM: QUESTION MARK ?
        PD: PERIOD .
        _: '
    """
    
    prefix = """
        PREFIX -> APOLOGY| ACKNOWLEDGE | HOLD
        APOLOGY -> (((i am | i_m) sorry | i apologize) | sorry) CM
        ACKNOWLEDGE -> ((i would | i_d) (love | be happy) to [help] | i have several | of course | certainly | sure) CM [but]
        HOLD -> (once again [please] | oops | pardon ) CM
    """

    simple = """
        SIMPLE -> which OBJECT ((do | did) you VERB-2 | would you VERB-2-WOULD | (would you be | (are | were) you) (VERB-2-ING | ADJ)) 
        VERB-2 -> want [to (know | learn) [about]] | wish to (know | learn) [more] about | have in mind | mean [by that| exactly | precisely] | need [information for |that info for] | refer to
        VERB-2-WOULD -> want [to (know | learn) [about]] | wish to (know | learn) about | care about | like VERB-2-WOULD-LIKE
        VERB-2-WOULD-LIKE -> (further | more) information about | me to check | to (hear about | know [more] about)
        VERB-2-ING -> asking [about | for] | inquiring about | looking at | referring to [exactly] | talking about | thinking [about | of] | (requesting | seeking) (further | more) information about
        ADJ -> curious about | interested in [exactly | learning more about] 
    """

    complex = """
        COMPLEX -> MAIN-YOU CLAUSE | MAIN-I CLAUSE CM MAIN-YOU-SIMPLE 

        MAIN-YOU -> (can | could | would) you [please] VERB-1 | (do | would) you mind VERB-ING-1

        VERB-1 -> (BE-PHRASE-1 | ME-PHRASE-1 | VERBS-1)
        VERBS-1 -> (clarify | specify | identify | explain | indicate | describe | elaborate) [exactly | more (precisely | concretely) | for me] | restate | say again | (provide | give [me]) [a bit] more details [ 0.1 as to] | point out [again | specifically]
        ME-PHRASE-1 -> tell me [again | exactly]| let me know | help me (understand | identify | out by identifying)
        BE-PHRASE-1 -> be [[a bit] more] (specific | precise | clear) (about | 0.1 as to)

        VERB-ING-1 -> (BE-PHRASE-ING-1 | ME-PHRASE-ING-1 | VERB-ING-1) 
        VERBS-ING-1 -> (clarifying | specifying | identifying | explaining | indicating | describing | elaborating) [exactly | more precisely | for me] | (providing | giving) [a bit] more details [ 0.1 as to] | pointing out [again | specifically]
        ME-PHRASE-ING-1 -> telling me | letting me know | helping me understand
        BE-PHRASE-ING-1 -> being [[a bit] more] (specific | precise | clear) (about | 0.1 as to)

        MAIN-I -> MAIN-I-BE | MAIN-I-DO | MAIN-I-IT
        MAIN-I-BE -> (i am | i_m) (not [exactly | quite | very] (sure | certain | clear on) | having [some] (trouble | difficulty) (determining | figuring out | identifying)) 
        MAIN-I-DO -> i don_t (know | understand) | i can't figure out
        MAIN-I-IT -> (it is | it_s) (not clear | unclear) [to me] | it is hard for me to pinpoint

        MAIN-YOU-SIMPLE -> (can | could | would) you [please] VERB-1-SIMPLE | (do | would) you mind VERB-ING-1-SIMPLE
        VERB-1-SIMPLE -> (BE-PHRASE-1-SIMPLE | VERBS-1-SIMPLE)
        VERBS-1-SIMPLE -> (clarify | specify | identify | explain | indicate | describe | elaborate) it [more (precisely | concretely) | for me] | restate it | say it again
        BE-PHRASE-1-SIMPLE -> be [a bit] more (specific | precise | clear) (about | 0.1 as to) it

        VERB-ING-1-SIMPLE -> (BE-PHRASE-ING-1-SIMPLE | VERB-ING-1-SIMPLE) 
        VERBS-ING-1-SIMPLE -> (clarifying | specifying | identifying | explaining | indicating | describing | elaborating) it [more (precisely | concretely) | for me] | pointing it out [again | specifically]
        BE-PHRASE-ING-1-SIMPLE -> being [[a bit] more] (specific | precise | clear) about it
    """

    clause = """
        CLAUSE -> which OBJECT (you are | you_re) (VERB-2-ING | ADJ)
        CLAUSE -> which OBJECT (you would | you_d) VERB-2-WOULD
    """

    obejct = """
        OBJECT -> one
        OBJECT -> DOMAIN
    """

# print(QBasics.combined_grammar)
# print_examples(QBasics.combined_grammar, 10)

@KlickitatGrammarCollection
class ABasics:
    root = """
        ROOT -> [PREFIX] SENT
    """
    prefix = """
        PREFIX -> (APOLOGY | ACKNOWLEDGE | OTHER) CM
        APOLOGY -> [(oh | ah) CM] ([yeah] sorry | [yeah] my (bad | fault) | whoops | oops | right) | i_m sorry | oh
        ACKNOWLEDGE -> [oh] ([yes CM] sure | [yes CM] of course | right | yeah | allright ) | [yeah] no (prob | problem | worries) | ok | yeah | yes
        OTHER -> let me clear that up | let_s see
    """
    sent = """
        SENT -> STATE-I OBJECT PD | OBJECT CM STATE-I-SIMPLE PD
    """
    state_i = """
            STATE-I -> i STATE-I-DO | (i am | i_m) STATE-I-BE | (i would | i_d) like to STATE-I-WOULD
            STATE-I-DO -> [actually] (mean [for] | meant | think i_d like to know about | want to know about)
            STATE-I-BE -> interested in | talking about | referring to | asking about | looking at
            STATE-I-WOULD -> (talk | know | ask) about | know for
        """
    state_i_simple = """
        STATE-I-SIMPLE -> i mean
    """

# print_examples(ABasics.combined_grammar, 1)

@KlickitatGrammarCollection
class ABasics_Level1:
    # directly gives entity
    
    # import the whole collection:
    _imports = [ABasics]

    object = """
        OBJECT -> the (A | B | C)
    """

@KlickitatGrammarCollection
class ABasics_Level2:
    # position order

    # import the whole collection:
    _imports = [ABasics]


    object = """
        OBJECT -> the (first | second | third) one
    """

@KlickitatGrammarCollection
class ABasics_Level3:
    # with more attributes

    # import the whole collection:
    _imports = [ABasics]


    object = """
        OBJECT -> (the | that ) SHORT-ATT one  | the [SHORT-ATT] one that LONG-ATT
    """


