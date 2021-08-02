import pandas as pd
import os
import re
from datetime import datetime
import time

msdw_folder = '/sharepoint/msdw-id/msdw-2020'
data_folder = './data'


def modify_notes(note_vect):
    """
    Function that creates the LINE column, which keeps track of the note lines
    that need to be concatenated to obtain the final note.

    :param note_vect: VALUE column with notes.

    :return: list of tuples with line number extracted from the beginning of the note and the text w/o the number
        in the format [n].
    """
    text = []
    for nv in note_vect:
        mynote = nv.strip('"')
        m = re.match(r' *\[[0-9]+\]', mynote)
        if m:
            n = int(re.sub(r'\]', '', re.sub(r' *\[', '', m.group(0))))
            s = re.sub(r'^ *\[[0-9]+\]', '', mynote)
        else:
            if re.match('Additional Progress Note Text is available in Epic or upon request '
                        'from the Data Warehouse team',
                        mynote):
                n = 21
            else:
                n = 0
            s = mynote
        text.append((n, s))
    return text


# Read filtered data and drop entries with empty person keys
time_rfact = time.time()
fact_data = pd.read_csv(os.path.join(data_folder, 'psych-notes-outpatient-579.0Gb.csv'), low_memory=False)
print(f'Finished loading fact data: {round(time.time() - time_rfact, 2)}s')
fact_data = fact_data.loc[fact_data.PERSON_KEY != '""']
fact_data = fact_data.astype({'PERSON_KEY': int})
print(f'Outpatient notes: {fact_data.shape[0]}\n')

# Select columns from metadata table related to status (e.g., signed) and note type (e.g., progress note)
time_rmtd = time.time()
mtd = pd.read_csv(os.path.join(msdw_folder, 'D_METADATA_DATA.csv'), low_memory=False)[['META_DATA_KEY',
                                                                                       'LEVEL1_CONTEXT_NAME',
                                                                                       'LEVEL2_EVENT_NAME',
                                                                                       'LEVEL3_ACTION_NAME']]
print(f'Loaded metadata: {round(time.time() - time_rmtd, 2)}s\n')
out_notes = fact_data.merge(mtd, on='META_DATA_KEY')

# Add LINE column
note_text = modify_notes(list(out_notes.VALUE))
note_text_df = pd.DataFrame(note_text, columns=["LINE", "NOTE_TEXT"])

out_notes_df = pd.concat([out_notes.reset_index(drop=True), note_text_df.reset_index(drop=True)], axis=1)
out_notes_df.columns = list(out_notes.columns) + list(note_text_df.columns)
# Rename columns
out_notes_df.rename(columns={'AUDIT_KEY': 'NOTE_ID',
                             'LEVEL1_CONTEXT_NAME': 'SYSTEM_NAME',
                             'LEVEL3_ACTION_NAME': 'STATUS',
                             'LEVEL2_EVENT_NAME': 'NOTE_TYPE',
                             'METAD_DATA_KEY': 'NOTE_TYPE_ID'}, inplace=True)
out_notes_df.drop(["VALUE"], inplace=True, axis=1)

print(f'Unique number of notes: {out_notes_df.NOTE_ID.nunique()}')
print(f'Note type counts:\n{out_notes_df.NOTE_TYPE.value_counts()}\n')

# Add dates and times
time_rcld = time.time()
cld = pd.read_csv(os.path.join(msdw_folder, 'D_CALENDAR_DATA.csv'), low_memory=False)[['CALENDAR_KEY',
                                                                                       'CALENDAR_DATE']]
print(f'Loaded calendar: {round(time.time() - time_rcld, 2)}s')
time_rtod = time.time()
tod = pd.read_csv('/sharepoint/msdw-id/msdw-2020/D_TIME_OF_DAY_DATA.csv', low_memory=True)[['TIME_OF_DAY_KEY', 'HOURS',
                                                                                            'MINUTES']]
print(f'Loaded time of day data: {round(time.time() - time_rtod, 2)}s\n')

out_notes_df = out_notes_df.merge(cld, on='CALENDAR_KEY').drop(['CALENDAR_KEY'], axis=1)
out_notes_df = out_notes_df.merge(tod, on='TIME_OF_DAY_KEY').drop(['TIME_OF_DAY_KEY'], axis=1)
dt_list = []
for _, row in out_notes_df.iterrows():
    try:
        dt = datetime.strptime(row.CALENDAR_DATE, "%Y-%m-%d 00:00:00")
        newdt = dt.replace(hour=row.HOURS, minute=row.MINUTES)
    except TypeError:
        newdt = None
    dt_list.append(newdt)
out_notes_df['SPECIFIED_DATETIME'] = dt_list
out_notes_df.rename(columns={'CALENDAR_DATE': 'CONTACT_DATE'}, inplace=True)
out_notes_df.drop(['HOURS', 'MINUTES'], inplace=True, axis=1)

# Add medical record number
time_rprs = time.time()
prs = pd.read_csv(os.path.join(msdw_folder, 'D_PERSON_DATA.csv'), low_memory=False)[['PERSON_KEY',
                                                                                     'MEDICAL_RECORD_NUMBER']]
print(f'Loaded person data: {round(time.time() - time_rprs)}s')

out_df = out_notes_df.merge(prs, on='PERSON_KEY')
out_df.rename(columns={'MEDICAL_RECORD_NUMBER': 'MRN', 'PERSON_KEY': 'PAT_ID'}, inplace=True)

col_n = ['PAT_ID', 'MRN', 'SYSTEM_NAME', 'CONTACT_DATE',
         'ENCOUNTER_KEY', 'FACILITY_KEY', 'NOTE_ID',
         'SPECIFIED_DATETIME', 'META_DATA_KEY',
         'NOTE_TYPE', 'STATUS']
notes = out_df[
    col_n + list(set(out_df.columns).difference(set(col_n + ["LINE", "NOTE_TEXT"]))) + ["LINE", "NOTE_TEXT"]].copy()
print('Saving preprocessed outpatient notes to csv...\n')
notes.to_csv(os.path.join(data_folder, f"PSYCH_OUTPATIENT_NOTES_{datetime.today().date()}.csv"),
             index=False, index_label=None)

print(f'Task ended: {round(time.time() - time_rfact, 2)}s')
