import time
import csv
import os
import pandas as pd

# Set minimum number of characters per note (minimum length [0n] at the beginning of the note)
min_length = 4


def read_factd_outpatient(data_file, size_in_bytes):
    """
    Read D_FACT_DATA from MSDW and select only outpatient
    notes based on list provided by Lauren A. Lepow, MD.

    Outfile is saved to csv

    :param data_file: fact data from MSDW
    :param size_in_bytes: chunk size

    """
    msdw_folder = "/sharepoint/msdw-id/msdw-2020"

    # Read fact_data and filter by facility key
    start = time.time()

    with open(os.path.join(msdw_folder, data_file)) as f:
        text = []
        last = ''
        i = 0
        try:
            for chunk in iter(lambda: f.read(size_in_bytes), ''):
                partial = time.time()
                ctext = chunk.split('\n')
                if i == 0:
                    header = ctext[0]
                    ll = ctext[1:-1]
                    fout, line = process_chunks(ll)
                else:
                    ll = [line] + [last + ctext[0]] + ctext[1:-1]
                    fout, line = process_chunks(ll)
                if len(fout) > 0:
                    text.extend(fout)
                last = ctext[-1]
                i += 1
                print(f"Processed {(size_in_bytes / 10 ** 9) * i} Gb in {round(time.time() - start, 2)}s "
                      f"(partial {round(time.time() - partial, 2)})")
            chk_last = last.split(',', 21)
            if len(chk_last) == 22:
                last_row = line.split(',', 21)
                last_filter = filter_data([last_row, chk_last])
            else:
                last_row = line + last
                last_filter = filter_data([last_row])
            if len(last_filter) > 0:
                text.extend(last_filter)
        except KeyboardInterrupt:
            pass
    if len(text) > 0:
        notes = pd.DataFrame(text)
        notes.columns = [nc.strip('"') for nc in header.split(',')]
        notes.to_csv(f'./data/psych-notes-outpatient-{(size_in_bytes / 10 ** 9) * i}Gb.csv',
                     index=False, index_label=None)
    else:
        print("No notes found for psychiatric outpatients.")
    return


"""
Private functions
"""


def read_facility_keys():
    """
    Read facility keys to select outpatient notes based on
    facilities.

    :return: facility keys
    """
    # Facility keys for outpatient psychiatric departments
    with open('./data/facilities.csv', 'r') as f:
        rd = csv.reader(f)
        next(rd)
        fkeys = [str(r[1]) for r in rd]
    return fkeys


def filter_data(lines):
    """
    Filter notes based on facility keys and minimum character length.

    :param lines: list of lines from fact data
    :return: filtered list
    """
    keys = read_facility_keys()
    ftext = list(
        filter(lambda x: str(x[5]) in keys and len(x[-1].strip('"')) > min_length, lines))
    return ftext


def process_chunks(text):
    """
    Function that stores notes and concatenates text that leaks in subsequent rows.

    :param text: fact data chunk
    :return: filtered chunk and last complete fact data line to enable concatenation
        of overhanging text, if needed.
    """
    nrow = len(text)
    pdata = []
    idx1 = 0
    idx2 = idx1 + 1
    while idx2 < nrow:
        line = text[idx1].split(',', 21)
        nline = text[idx2].split(',', 21)
        if len(line) == 22:
            if len(nline) == 22:
                if line[-1] != '"<factless>"':
                    pdata.append(line)
                idx1 += 1
                idx2 = idx1 + 1
            else:
                s = line[-1]
                while len(nline) != 22:
                    s = ' '.join([s, ','.join(nline)])
                    idx2 += 1
                    try:
                        nline = text[idx2].split(',', 21)
                    except IndexError:
                        line[-1] = s
                        # if s != '""':
                        #     pdata.append(line)
                        pdata = filter_data(pdata)
                        return pdata, ','.join(line)
                line[-1] = s
                if s != '""':
                    pdata.append(line)
                idx1 = idx2
                idx2 += 1
        else:
            print("Stucked in loop...")
    pdata = filter_data(pdata)
    return pdata, text[idx1]


if __name__ == '__main__':
    read_factd_outpatient('FACT_DATA.csv', 1000000000)
    print("\n\nTask ended")
