import pickle as pkl
import utils
import time
from collections import Counter, namedtuple
import numpy as np
import itertools
import multiprocessing
from Bio import pairwise2
import csv
from nltk.tokenize import word_tokenize
import os
import sys
import argparse
from scipy.special import binom
import random

# Named tuples for within-note and between-patient redundancy output
wn_redundancy = namedtuple('wn_redundancy', ['note_id', 'nr_score', 'counts'])
bp_redundancy = namedtuple('bp_redundancy', ['sen_count', 'sen_vocab', 'tkn_notes', 'red_matrix'])


class RedundancyDetection:
    """
    Redundancy detection class.

    Attributes:
        id_to_note: dict with note id as key and list of
            sentences as valies
        mrn_to_notes: dict with mrn as key and list of tuples
            with note id and list of tokenized notes as values
    """

    def __init__(self, data):
        self.data = data
        self.id_to_note = {}
        self.mrn_to_notes = {}
        self.mrn_to_sent = {}
        colnames = list(data[0])
        values = data[1]
        noteidpos = colnames.index('NOTE_ID')
        senpos = colnames.index('SENTENCES')
        mrnpos = colnames.index('MRN')
        tknpos = colnames.index('TOKENS')
        random_mrn = set([val[mrnpos] for val in values])
        # random_mrn = random.sample([val[mrnpos] for val in values], 100)
        for row in values:
            if row[mrnpos] in random_mrn:
                self.id_to_note[row[noteidpos]] = row[senpos]
                self.mrn_to_notes.setdefault(row[mrnpos], list()).append((row[noteidpos], row[tknpos]))
                # self.mrn_to_notes.setdefault(row[mrnpos], list()).append((row[noteidpos], row[senpos]))
                self.mrn_to_sent.setdefault(row[mrnpos], list()).append((row[noteidpos], row[senpos]))
        print(f'Selected {len(self.id_to_note)} out of {values.shape[0]} notes')

    def within_note(self):
        """
        Aim: detect errors in notes investigating within note redundancy
        :return: list of named tuple (note_id, nr_score, counts)
            int, float, dict {sentence: counts>1}. nr_score = (n sentences - n unique sentences)/n sentences
        """
        # within note redundancy
        id_note_list = list(filter(lambda x: len(x[1]) > 1, self.id_to_note.items()))
        wnr_list = []
        for el in id_note_list:
            out = self._wn_redundancy(el)
            if out.nr_score > 0:
                wnr_list.append(out)
            else:
                continue
        return wnr_list

    def between_note(self):
        """
        Computes within patient redundancy to detect the copy-paste practice.
        :return: list of tuples with MRN, list of tuples with best aligned note pairs, alignments, and alignment score
        """
        with open(os.path.join(utils.data_path, utils.select_mrns), 'r') as f:
            rd = csv.reader(f)
            mrns = [str(r[0]) for r in rd]
        mrn_note_list = list(filter(lambda x: len(x[1]) > 1 and str(x[0]) in mrns, self.mrn_to_notes.items()))

        # mrn_note_list = list(filter(lambda x: len(x[1]) > 1 and str(x[0]) in mrns[:100], self.mrn_to_notes.items()))
        # print(f'Number of MRNs: {len(mrn_note_list)}/{len(mrn_note)}')

        print(f'Number of MRNs: {len(mrn_note_list)}')
        print(
            f"Number of notes per patient: {len(mrn_note_list[0][1])} -- "
            f"Total number of notes: {len(mrn_note_list) * len(mrn_note_list[0][1])}")
        # begin = time.time()
        out = [(el[0], self._align_notes(el[1])) for el in mrn_note_list]
        # print(
        #     f'Task ended in {round(time.time() - begin, 2)}s -- '
        #     f'estimated {round((time.time() - begin), 2) * len(mrn_note) / len(mrn_note_list)}')
        return out

    def between_patient(self):
        """
        Investigate between patient redundancy to detect templates
        :return: named tuple (sen_count, sen_vocab, tkn_notes, red_matrix)
            sen_count = dictionary of sentences with counts for the number of notes they
                exactly appear in
            sen_vocab = sentence vocabulary, dictionary idx to tokenized unique sentences
            tkn_notes = dictionary idx to (note id, tokenized note)
            red_matrix = csr matrix of dimension len(sen_vocab), len(tkn_notes) storing the
                percentage of unique words shared between each sentence and note (all comparisons)
                divided by the number of unique words in the sentence
        """
        # Drop within note redundancy
        mrn_to_notes_notred = {
            mrn: list(dict.fromkeys(list(itertools.chain.from_iterable([s[1] for s in self.mrn_to_sent[mrn]])))) for
            mrn in self.mrn_to_sent.keys()}
        # Concatenate sentences
        sentences = list(itertools.chain.from_iterable([s for s in mrn_to_notes_notred.values()]))
        # Consider sentences that appear more than once
        sen_dict = Counter(sentences)
        q3 = np.percentile([v for v in sen_dict.values() if v > 1], q=75)
        sen_dict_rid = {sen: count for sen, count in sen_dict.items() if count > q3}
        print(f"N = {len(sen_dict_rid)} sentences that are repeated above 75th percentile {q3}.")

        align_sen = self._align_sentences(sen_dict_rid)
        #
        # idx_to_sen = {int(idx): word_tokenize(s) for idx, s in enumerate(set(sentences)) if s != ''}
        # # Tokenize non-redundant notes
        # idx_to_id_tknnote_notred = {int(idx): (el[0], word_tokenize(' '.join(el[1]))) for idx, el in
        #                             enumerate(id_to_note_notred.items())}
        # # Count exact redundancy
        # sen_dict = Counter(sentences)
        # print(f"Number of sentences considered: {len(sen_dict)}")
        # # Estimate template redundancy
        # align_mat = self._align_sentences(idx_to_sen, idx_to_id_tknnote_notred)
        # return bp_redundancy(sen_count=sen_dict, sen_vocab=idx_to_sen, tkn_notes=idx_to_id_tknnote_notred,
        #                      red_matrix=align_mat)
        return align_sen

    def _align_notes(self, note_list):
        """
        Method that aligns notes for between note comparison.
        :param note_list: list of tuples with (note id, tokenized note)
        :return: list of tuples with note id pair, aligned note 1, aligned note 2, maximum alignment score for all pair
            comparisons
        """
        # ids_notes = utils.create_pairs({int(el[0]): word_tokenize('. '.join(el[1])) for el in note_list})
        ids_notes = utils.create_pairs({int(el[0]): el[1] for el in note_list})
        with multiprocessing.Pool(processes=8) as pool:
            out = pool.map(self._nw, ids_notes.items())
        return out

    @staticmethod
    def _align_sentences(sen_dict_rid):
        """
        Method to intersect sentences with notes for between-patient redundancy detection.
        :param idx_to_sen: dictionary idx to tokenized sentence with no repeated sentences
        :param idx_to_id_tknnote_notred: dictionary idx to tuple with note id and tokenized note from non redundant
            sentences
        :return: csr matrix with overlapping percentages of sentences and notes > than threshold defined in utils
        """
        align_sen = {}
        idx_to_sen = {idx: (sen, word_tokenize(sen)) for idx, sen in enumerate(sen_dict_rid.keys())}
        n_comp = binom(len(idx_to_sen), 2)
        for n, p in enumerate(itertools.combinations(list(idx_to_sen.keys()), 2)):
            tmp_align = pairwise2.align.globalms(idx_to_sen[p[0]][1],
                                                 idx_to_sen[p[1]][1],
                                                 utils.match,
                                                 utils.mismatch,
                                                 utils.gap_open,
                                                 utils.gap_extend,
                                                 gap_char=['-']
                                                 )[0]
            align_sen.setdefault(idx_to_sen[p[0]][0], list()).append(
                (idx_to_sen[p[1]][0], [tmp_align.seqA, tmp_align.seqB, tmp_align.score]))
            if n % 1000000 == 0:
                print(f"Completed {n}/{n_comp} comparisons")
                # row, col, data = [], [], []
                # for p in itertools.product(list(idx_to_sen.keys()), list(idx_to_id_tknnote_notred.keys())):
                #     template_score = self._overlap_percentage(set(idx_to_sen[p[0]]),
                #                                               set(idx_to_id_tknnote_notred[p[1]][1]))
                #     if template_score > utils.template_threshold:
                #         row.append(p[0])
                #         col.append(p[1])
                #         data.append(template_score)
                # mat = csr_matrix((data, (row, col)), shape=(len(idx_to_sen), len(idx_to_id_tknnote_notred)), dtype=np.float16)
        return align_sen

    @staticmethod
    def _overlap_percentage(sen1, sen2):
        """
        Method used for between-patient redundancy
        :param sen1: tokenized sentence set
        :param sen2: tokenized note set
        :return: percentage of unique words in set sentence 1 that occur in set sentence 2
        """
        score = len(sen1.intersection(sen2)) / len(sen1)
        return score

    @staticmethod
    def _wn_redundancy(idx_note_tuple):
        """
        Method for within-note redundancy.
        :param idx_note_tuple: tuple (note id, list of sentences)
        :return: named tuple with note id, redundancy score, and sentences that occure more than once with
            corresponding count
        """
        counts = Counter(idx_note_tuple[1])
        nr_score = utils.nr_score(counts)
        redundant_counts = {sen: n for sen, n in counts.items() if n > 1}
        return wn_redundancy(note_id=idx_note_tuple[0], nr_score=nr_score, counts=redundant_counts)

    @staticmethod
    def _nw(pair_notes):
        """
        Method that aligns pairs of notes and returns a tuple with note pair ids for best alignment, aligned note 1,
        aligned note 2, maximum alignment score for all pairs with same note 1.
        :param pair_notes: list of note ids and ordered dict with all possible comparisons for note id
        :return: best aligned pair, alignment note 1, alignment note 2, maximum alignment score
        """
        max_val = -np.inf
        final_align = None
        final_pair = None
        for pair, notes in pair_notes[1].items():
            alignments = pairwise2.align.globalms(notes[0],
                                                  notes[1],
                                                  utils.match,
                                                  utils.mismatch,
                                                  utils.gap_open,
                                                  utils.gap_extend,
                                                  gap_char=['-'])
            if alignments[0].score > max_val:
                max_val = alignments[0].score
                final_align = alignments[0]
                final_pair = pair
            else:
                continue
        return final_pair, final_align.seqA, final_align.seqB, max_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Investigate psychiatric notes redundancy")
    parser.add_argument('-r', '--redundancy',
                        type=str,
                        dest='redundancy_method',
                        help='Select redundancy type to investigate')
    parser.add_argument('-o', '--out',
                        type=str,
                        dest='output_file',
                        help='Name of the output file')
    config = parser.parse_args(sys.argv[1:])

    start_file = time.time()
    outp_notes = utils.read_notes(utils.outp_notes_file)
    outp_notes_all = (outp_notes[0], outp_notes[1])

    print(f'Finished reading the note file: {round(time.time() - start_file, 2)}s')

    redundancy = RedundancyDetection(outp_notes_all)

    start = time.time()
    if config.redundancy_method == 'wn':
        red = redundancy.within_note()
    elif config.redundancy_method == 'bn':
        red = redundancy.between_note()
    elif config.redundancy_method == 'bp':
        red = redundancy.between_patient()
    else:
        raise ModuleNotFoundError(
            f"Could not find redundancy method {config.redundancy_method}. "
            f"Please specify one of the available methods: "
            f"'wn' within note redundancy; 'bn' between note redundancy; 'bp' between patient redundancy.")
    stop = time.time() - start
    print(f'Task ended in {round(stop, 5)}s')
    # print(f'Task ended in {round(stop, 5)}s -- Estimated time {round(stop, 5) * (outp_notes[1].shape[0] / 100)}s')

    pkl.dump(red, open(os.path.join(utils.data_path, config.output_file), 'wb'))
    print(f'Tasked ended: {round(time.time() - start_file, 2)}s')
