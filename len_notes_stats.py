import pandas as pd
import re
import pickle as pkl
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from nltk.tokenize import sent_tokenize, word_tokenize
import utils
import time

note_file = 'outpatient_notes.pkl'
note_label = 'opud.pkl'


def preprocess_note(note):
    """
    Function that replaces punctuation marks (except for '.') and special
    characters with a single space character. Text is also transformed to lower case.
    """
    regexp = r'[!?\\\-,;\'\*\(\)\[\]:`\"_=(\/\/){}\|]'
    note_replace = re.sub(regexp, ' ', note).lower()
    note_replace = re.sub(r"\S*@\S*\s?", " ", note_replace)
    note_replace = re.sub(r"http\S+", " ", note_replace)
    note_replace = re.sub(r'(\+[0-9]*[ ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})([ ]?x[0-9]{3})?', ' ',
                          note_replace)
    note_space = re.sub(' +', ' ', note_replace)
    note_out = re.sub(r'pt\.', 'pt', note_space)
    return note_out


start = time.time()
notes = utils.read_notes(os.path.join(utils.data_path, note_file))
labels = pkl.load(open(os.path.join(utils.data_path, note_label), 'rb'))

outp_notes = pd.DataFrame(notes[1])
outp_notes.columns = notes[0]
outp_notes = outp_notes.loc[outp_notes.NOTE_ID.isin(list(labels.keys()))].copy()

# We first get rid of special characters and transform notes to lower case
outp_notes['PR_NOTE_TEXT'] = [preprocess_note(n) for n in outp_notes.NOTE_TEXT]
outp_notes.drop('NOTE_TEXT', axis=1, inplace=True)

print(f'**N patients**: {outp_notes.MRN.nunique()}')
print(f'**N notes**: {outp_notes.shape[0]}')

note_length = []
for _, row in outp_notes.iterrows():
    note_length.append(len(row.PR_NOTE_TEXT))
outp_notes['LENGTH'] = note_length

ml_notes_rid = outp_notes.copy()
# loglen = np.log(outp_notes.LENGTH)
# gm = GaussianMixture(n_components=3, random_state=42).fit_predict(np.array(loglen).reshape(-1, 1))
#
# max_lengths = np.array([max(outp_notes.loc[gm == i].LENGTH) for i in range(len(set(gm)))])
# print(max_lengths)
#
# short_pos = int(np.where(max_lengths == min(max_lengths))[0])
# long_pos = int(np.where(max_lengths == max(max_lengths))[0])
# medium_pos = int(np.where(((max_lengths < max_lengths[long_pos]) & (max_lengths > max_lengths[short_pos])))[0])
#
# m_notes = outp_notes.loc[gm == medium_pos].copy()
# l_notes = outp_notes.loc[gm == long_pos].copy()
# ml_notes = pd.concat([m_notes, l_notes])
# ml_notes_count = ml_notes[['MRN', 'NOTE_ID']].groupby(['MRN']).nunique()
#
# # median number of notes per patient
# min_nnotes = ml_notes_count.NOTE_ID.describe()[5]
# # 75th percentile
# max_nnotes = ml_notes_count.NOTE_ID.describe()[6]
# select_mrn = list(
#     ml_notes_count.loc[((ml_notes_count.NOTE_ID >= min_nnotes) & (ml_notes_count.NOTE_ID <= max_nnotes))].index)
# ml_notes_rid = ml_notes.loc[ml_notes.MRN.isin(select_mrn)].copy()
#
# print(f'**N patients**: {ml_notes_rid.MRN.nunique()}')
# print(f'**N notes**: {ml_notes_rid.shape[0]}')

ml_text = list(ml_notes_rid.PR_NOTE_TEXT)

# Report number of sequence stats
ml_sents = [sent_tokenize(sn) for sn in ml_text]
medlong_sents = [[re.sub(r'([ ]?\.)', '', sn) for sn in sen if
                  re.search('([a-z]* [a-z]*)+',
                            sn)] for sen in
                 ml_sents]
ml_notes_rid['SENTENCES'] = [[sn for sn in sen if sn != ''] for sen in
                             medlong_sents]
ml_notes_tkn = [word_tokenize(' '.join(sen)) for sen in ml_notes_rid['SENTENCES']]
ml_notes_rid['TOKENS'] = ml_notes_tkn

ml_notes_rid.drop('PR_NOTE_TEXT', axis=1, inplace=True)
ml_notes_rid = ml_notes_rid.sort_values(['MRN', 'CONTACT_DATE', 'NOTE_ID'])
pkl.dump((ml_notes_rid.columns, ml_notes_rid.to_numpy()),
         open(os.path.join(utils.data_path, 'filtered_outpatient_notes.pkl'), 'wb'))
print(f"Task ended in {round(time.time() - start, 2)}s")
