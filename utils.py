import numpy as np
import pickle as pkl
import os
from collections import OrderedDict
import spacy
import string
import csv

# Data path and redundancy input file name
data_path = ''
outp_notes_file = ''
select_mrns = ''
bn_redundancy_file = ''

# Needleman-Wunsch alignment weights
match = 2
mismatch = -10
gap_open = -0.5
gap_extend = -0.1

# Overlap threshold for template detection
template_threshold = 0.8

nlp = spacy.load("en_core_web_sm", disable=["ner"])

# INPATIENT
# # BETWEEN-NOTE PARAMETERS
# # With redundancy
# w2v_param_bn_r = {"epochs": [50],
#                   "start_alpha": [0.2],
#                   "end_alpha": [0.0001, 0.00001],
#                   "vector_size": [100, 500, 600, 800],
#                   "min_count": [1],
#                   "window": [5, 10]}
# # W/o redundancy
# w2v_param_bn_nr = {"epochs": [50],
#                    "start_alpha": [0.2],
#                    "end_alpha": [0.0001, 0.00001],
#                    "vector_size": [500, 600, 800],
#                    "min_count": [1],
#                    "window": [5, 10]}
#
# # WITHIN-NOTE PARAMETERS
# # W/ redundancy
# w2v_param_wn_r = {"epochs": [10],
#                   "start_alpha": [0.2],
#                   "end_alpha": [0.00001],
#                   "vector_size": [100],
#                   "min_count": [2],
#                   "window": [2]}
# # W/o redundancy
# w2v_param_wn_nr = {"epochs": [10],
#                    "start_alpha": [0.2],
#                    "end_alpha": [0.0001],
#                    "vector_size": [200],
#                    "min_count": [2],
#                    "window": [2]}


# OUTPATIENT
# BETWEEN-NOTE PARAMETERS
w2v_param_bn_r = {"epochs": [50],
                  "start_alpha": [0.2],
                  "end_alpha": [0.0001, 0.00001],
                  "vector_size": [100, 500, 600, 800],
                  "min_count": [1],
                  "window": [5, 10]}
# W/o redundancy
w2v_param_bn_nr = {"epochs": [50],
                   "start_alpha": [0.2],
                   "end_alpha": [0.0001, 0.00001],
                   "vector_size": [500, 600, 800],
                   "min_count": [1],
                   "window": [5, 10]}
# WITHIN-NOTE PARAMETERS
# W/ redundancy
w2v_param_wn_r = {"epochs": [5],
                  "start_alpha": [0.2],
                  "end_alpha": [0.0001],
                  "vector_size": [100],
                  "min_count": [2],
                  "window": [2]}
# W/o redundancy
w2v_param_wn_nr = {"epochs": [5],
                   "start_alpha": [0.2],
                   "end_alpha": [0.0001],
                   "vector_size": [100],
                   "min_count": [2],
                   "window": [2]}


class Memoize:
    """
    Memoization class
    """

    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]


# @Memoize
def read_notes(pkl_file):
    """
    Read medium-long notes
    :param pkl_file: file name
    :return: list of two elements. Index of column names and list of tuples with corresponding values.
    """
    notes = pkl.load(open(os.path.join(data_path, pkl_file), 'rb'))
    return notes


"""
Functions
"""


def nr_score(counter):
    """
    Function that computes within note redundancy score
    :param counter: collections counter dictionary
    :return: score
    """
    s = sum(counter.values())
    score = (s - len(counter)) / s
    return score


def create_pairs(ids_notes):
    """
    Create pairs of notes to align.
    :param ids_notes: dictionary {note id: tokenized note}
    :return: OrderedDict with ordered noteid as keys and as values OrderedDict with tuple of all possible combinations
        of note ids within the same patient as keys and tuple of corresponding tokenized notes as values
    """
    note_ids = sorted(np.array(list(ids_notes.keys()), dtype=int))
    noteid_pairs = OrderedDict()
    for i, note_id in enumerate(note_ids[:-1]):
        noteid_pairs[int(note_id)] = OrderedDict(
            {(note_id, idx): (ids_notes[note_id], ids_notes[idx]) for idx in note_ids[i + 1:]})
    return noteid_pairs


def tokenize_sen(sent):
    """
    Sentence parser

    :param sent: str (sentence)
    :return: tokenized sentence
    """
    tkn_sen = []
    prev, head = None, None
    s = nlp(sent)
    for w in s:
        if w.is_punct or w.is_space:
            continue

        if w.like_url or w.like_email:
            continue

        if w.text in string.punctuation:
            continue

        if not (w.is_alpha):
            continue

        if w.dep_ == 'neg' or w.text == 'no':
            if w.text != w.head.text:
                if w.head.pos_ == 'VERB':
                    head = w.head.lemma_
                    tkn_sen.append(
                        ' '.join(sorted([head, w.norm_], key=len)))
                else:
                    head = w.head.norm_
                    tkn_sen.append(
                        ' '.join(sorted([head, w.norm_], key=len)))
                prev = w.norm_

        if w.pos_ == 'VERB':
            pw = w.lemma_

        else:
            pw = w.norm_

        if pw == prev or len(pw) == 1 or pw == head:
            continue

        if pw == '-PRON-':
            continue

        tkn_sen.append(pw)
        head = None
        prev = None

    return tkn_sen


def dump_notes(note_dict, file_name):
    """
    Save notes for language modeling
    :param note_dict:
    :param file_name:
    :return:
    """
    with open(os.path.join(data_path, file_name), 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for mrn, notes in note_dict.items():
            for _, n in notes.items():
                if len(n) == 0:
                    continue
                else:
                    wr.writerow(n)
                # if len(n) == 1:
                #     wr.writerow(n[0])
                # elif len(n) > 1:
                #     for sen in n:
                #         wr.writerow(sen)
