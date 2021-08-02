import utils
import os
import pickle as pkl
import random
import numpy as np
import itertools
import time
from nltk.tokenize import sent_tokenize
from collections import OrderedDict
import argparse
import sys


def create_wn_datasets(mrn_to_id_sen):
    """
    Function that returns two dictionaries {mrn: idx: [tokenized sentences]} one for
    the original set of notes with within-note redundancy dropped and the other with the raw notes

    :param mrn_to_id_sen: dictionary {mrn: idx: [sentences]}
    :return: dict, dict
    """
    idx_to_sen, sen_to_idx = {}, {}
    sen_set = set()
    for m in mrn_to_id_sen.keys():
        sen_set.update(
            list(dict.fromkeys(list(itertools.chain.from_iterable(mrn_to_id_sen[m].values())))))
    N = len(sen_set)
    i = 0
    tloop = time.time()
    start = time.time()
    for idx, s in enumerate(sen_set):
        sen_to_idx[s] = idx
        idx_to_sen[idx] = utils.tokenize_sen(s)
        i += 1
        if i % 100000 == 0:
            print(f"Processed {i}/{N} sentences in {round(time.time() - start, 2)}s")
            start = time.time()
    print(f'Finished tokenizing sentences in {round(time.time() - tloop, 2)}s')
    raw, wn = {}, {}
    for mrn in mrn_to_id_sen.keys():
        raw[mrn] = {}
        wn[mrn] = {}
        for nid, sen in mrn_to_id_sen[mrn].items():
            raw[mrn][nid] = [idx_to_sen[sen_to_idx[s]] for s in sen]
            wn[mrn][nid] = [idx_to_sen[sen_to_idx[s]] for s in list(dict.fromkeys(sen))]
    print('Within-note redundancy dataset ready')
    return wn, raw


def create_bn_datasets(bn_out, idx_to_tknote):
    """
    Function that builds two dictionaries {mrn: idx: [tokenized sentences]} one with
    between-note redundancy dropped and the other one with raw notes for selected mrns.

    :param bn_out: list with [(mrn, (idx_pair, seqA, seqB, percentage overlap))]
    :param idx_to_tknote: dict {mrn: idx: [sentences]}
    :return: dict dict
    """
    bn, raw = {}, {}
    N = len(bn_out)
    i = 0
    tloop = time.time()
    for val in bn_out:
        start = time.time()
        mrn = val[0]
        el = val[1]
        ctrl_ids = set()
        bn[mrn] = OrderedDict()
        raw[mrn] = OrderedDict()
        for tup in _order(el):
            pair = tup[0]
            if pair[0] not in ctrl_ids:
                ctrl_ids.add(pair[0])
                # note = [utils.tokenize_sen(s.strip('.')) for s in sent_tokenize(idx_to_tknote[mrn][pair[0]])]
                note = utils.tokenize_sen(' '.join(idx_to_tknote[mrn][pair[0]]))
                bn[mrn][pair[0]] = note
                raw[mrn][pair[0]] = note
                # note = _tokenize_sen(idx_to_tknote[mrn][pair[0]])
                # if note is not None:
                #     bn[mrn][pair[0]] = note
                #     raw[mrn][pair[0]] = note
            if pair[1] not in ctrl_ids:
                ctrl_ids.add(pair[1])
                # note_redu = [utils.tokenize_sen(s.strip('.')) for s in
                #              sent_tokenize(remove_bn_redundancy(tup[1], tup[2]))]
                note_redu = utils.tokenize_sen(remove_bn_redundancy(tup[1], tup[2]))
                note = utils.tokenize_sen(' '.join(idx_to_tknote[mrn][pair[1]]))
                bn[mrn][pair[1]] = note_redu
                raw[mrn][pair[1]] = note
                # note_redu = _tokenize_sen(remove_bn_redundancy(tup[1], tup[2]))
                # note = _tokenize_sen(idx_to_tknote[mrn][pair[1]])
                # if note_redu is not None:
                #     bn[mrn][pair[1]] = note_redu
                # if note is not None:
                #     raw[mrn][pair[1]] = note
        i += 1
        if i % 10 == 0:
            print(f"Processed {i}/{N} MRNs in {round(time.time() - start, 2)}s")
    print(f"Finished tokenizing sentences in {round(time.time() - tloop, 2)}s")
    print(f"Between-note datasets ready.")
    return bn, raw


def remove_bn_redundancy(align1, align2):
    """
    Takes as input two alignments for note 1 and note 2 and returns non redundant note 2.

    :param align1: list
    :param align2: list
    :return: str with sentence
    """
    nr_note = []
    for a, b in zip(align1, align2):
        if a == b:
            continue
        else:
            if b != '-':
                nr_note.append(b)
            else:
                continue
    return ' '.join(nr_note)


def create_train_test(notes_dict, ratio=0.8):
    """
    Takes as input a dictionary as returned from the create dataset functions and returns
    two dictionaries for training and testing

    :param notes_dict: dict
    :param ratio: percentage to include in training
    :return: dict, dict
    """
    train_mrns, test_mrns = _train_test_split(list(notes_dict.keys()), ratio=ratio)
    train_dict = {mrn: notes_dict[mrn] for mrn in train_mrns}
    test_dict = {mrn: notes_dict[mrn] for mrn in test_mrns}
    return train_dict, test_dict


""""
Private functions
"""


def _train_test_split(mrns, ratio=0.8):
    """
    Takes as input a list of MRNs
    and returns MRNs for training and test sets, with training size equal ratio.
    :param mrns: list of mrns
    :param ratio: float
    :return: tuple of lists
    """
    random.seed(42)
    n_mrn = int(len(mrns) * ratio)
    train_mrns = random.sample(sorted(mrns), n_mrn)
    test_mrns = [m for m in mrns if m not in train_mrns]
    return train_mrns, test_mrns


def _overlap_percentage(align1, align2):
    """
    Function that computes the percentage of overlap between note1 and note2
    :param align1: list (tokenized note)
    :param align2: list (tokenized note)
    :return: float
    """
    perc = 0
    for a, b in zip(align1, align2):
        if a == b and b != '-':
            perc += 1
    return (perc / len([b for b in align2 if b != '-'])) * 100


def _order(alignments):
    """
    Function that returns a list of (idx pair, seqA, seqB, percentage overlap) ordered decreasingly according to the
    percentage of overlap.

    :param alignments: tuple
    :return: list
    """
    vect = [(p[0], p[1], p[2], _overlap_percentage(p[1], p[2])) for p in alignments]
    ordered = sorted(vect, key=lambda x: x[0][1], reverse=False)
    return ordered


# def _tokenize_sen(text):
#     """
#     Sentence tokenizer.
#
#     :param text: list of sentences
#     :return: list of tokenized sentences
#     """
#     tkn_sen = sent_tokenize(text)
#     if len(tkn_sen) > 1:
#         note = [utils.tokenize_sen(s.strip('.')) for s in tkn_sen if s.strip('.') != '']
#         note = [nt for nt in note if len(nt) > 1]
#     elif len(tkn_sen) == 1 and tkn_sen[0].strip('.') != '':
#         note = [utils.tokenize_sen(tkn_sen[0].strip('.'))]
#         if len(note[0]) <= 1:
#             note = None
#     else:
#         note = None
#     return note


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create datasets for language modeling")
    parser.add_argument('-dt', '--dataset',
                        type=str,
                        dest='create_dataset',
                        help='Select dataset to create')
    parser.add_argument('-co', '--csvout',
                        type=str,
                        dest='csv_output',
                        help='Name of the csv output file with tokenized sentences')
    parser.add_argument('-o', '--out',
                        type=str,
                        dest='output_file',
                        help='Name of the pkl file with the datasets, both raw and w/o redundancy')
    config = parser.parse_args(sys.argv[1:])

    # Read raw notes
    notes = utils.read_notes(utils.outp_notes_file)

    # Create raw datasets
    colnames = list(notes[0])
    values = notes[1]
    noteidpos = colnames.index('NOTE_ID')
    senpos = colnames.index('SENTENCES')
    mrnpos = colnames.index('MRN')
    tknpos = colnames.index('TOKENS')

    if config.create_dataset == 'bn':
        # Read between-note redundancy output
        bn_redu = pkl.load(open(os.path.join(utils.data_path, utils.bn_redundancy_file), 'rb'))
        select_mrn = np.array(bn_redu, dtype='object')[:, 0]
        mrn_to_notes = {mrn: {} for mrn in select_mrn}
        for row in values:
            if row[mrnpos] in select_mrn:
                # mrn_to_notes[row[mrnpos]][row[noteidpos]] = '. '.join(row[senpos])
                mrn_to_notes[row[mrnpos]][row[noteidpos]] = row[tknpos]

        bn_mrn_to_notes, raw_bn_mrn_to_notes = create_bn_datasets(bn_redu, mrn_to_notes)
        bn_train, bn_test = create_train_test(bn_mrn_to_notes, ratio=0.8)
        raw_bn_train, raw_bn_test = create_train_test(raw_bn_mrn_to_notes, ratio=0.8)

        # Save sentences to csv file
        utils.dump_notes(bn_train, f'{config.csv_output}_train.csv')
        utils.dump_notes(bn_test, f'{config.csv_output}_test.csv')
        utils.dump_notes(raw_bn_train, f'raw_{config.csv_output}_train.csv')
        utils.dump_notes(raw_bn_test, f'raw_{config.csv_output}_test.csv')
        # Save output to pkl object
        pkl.dump((bn_train, raw_bn_train), open(os.path.join(utils.data_path, f'{config.output_file}_train.pkl'), 'wb'))
        pkl.dump((bn_test, raw_bn_test), open(os.path.join(utils.data_path, f'{config.output_file}_test.pkl'), 'wb'))
    elif config.create_dataset == 'wn':
        mrn_to_sen = {}
        for row in values:
            if row[mrnpos] in mrn_to_sen:
                mrn_to_sen[row[mrnpos]][row[noteidpos]] = row[senpos]
            else:
                mrn_to_sen[row[mrnpos]] = {row[noteidpos]: row[senpos]}
        wn_mrn_to_sen, raw_wn_mrn_to_sen = create_wn_datasets(mrn_to_sen)

        wn_train, wn_test = create_train_test(wn_mrn_to_sen, ratio=0.8)
        raw_wn_train, raw_wn_test = create_train_test(raw_wn_mrn_to_sen, ratio=0.8)

        pkl.dump((wn_train, raw_wn_train), open(os.path.join(utils.data_path, f'{config.output_file}_train.pkl'), 'wb'))
        pkl.dump((wn_test, raw_wn_test), open(os.path.join(utils.data_path, f'{config.output_file}_test.pkl'), 'wb'))

        utils.dump_notes(wn_train, f'{config.csv_output}_train.csv')
        utils.dump_notes(wn_test, f'{config.csv_output}_test.csv')
        utils.dump_notes(raw_wn_train, f'raw_{config.csv_output}_train.csv')
        utils.dump_notes(raw_wn_test, f'raw_{config.csv_output}_test.csv')
    else:
        raise ModuleNotFoundError(
            f"Could not find the dataset requested: {config.create_dataset}. "
            f"Please specify one of the available methods: "
            f"'wn' within note; 'bn' between note.")
