import numpy as np
import pandas as pd
import time
import os


def concat_notes(ar_notes):
    idx = 0
    notes_text = []
    while idx < ar_notes.shape[0]:
        jdx = idx + 1
        note = [ar_notes[idx][-1]]
        while jdx < ar_notes.shape[0] and ar_notes[jdx][-2] > 1:
            note.append(ar_notes[jdx][-1])
            jdx += 1
        notes_text.append(np.append(ar_notes[idx][:-1], [''.join(note)]))
        idx = jdx
    return notes_text


data_folder = './data'
notes_file = 'PSYCH_OUTPATIENT_NOTES_2021-06-28.csv'

notes = pd.read_csv(os.path.join(data_folder, notes_file), low_memory=False)

time_cnotes = time.time()
notes_rid = notes.loc[((notes.NOTE_TEXT.notna()) & (notes.LINE > 0))]
snotes = notes_rid.sort_values(by=['NOTE_ID', 'LINE'], axis=0)
note_text = concat_notes(snotes.to_numpy())
new_notes = pd.DataFrame(note_text, columns=snotes.columns)
new_notes.drop('LINE', axis=1, inplace=True)
print(f'Created dataset with concatenated notes {round(time.time() - time_cnotes, 2)}s')

print(f'Saving outpatient concatenated notes (N={new_notes.shape[0]})...\n')
new_notes.to_csv(os.path.join(data_folder, 'outpatient_notes.csv'), index=False, index_label=None)
