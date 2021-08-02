# Redundancy detection and investigation

1. Run `read_fact_data.py` to filter out outpatient notes from MSDW D_FACT_DATA.csv file (579 Gb) processing it 
   by chunks.

    ```
   source ../myenv/bin/activate
   python read_fact_data.py
   ```
   
   Output file: `psych-notes-outpatient-579.0Gb.csv`.
   
2. Prepare outpatient notes dataset similar to `PSYCH_NOTES_092020.csv` in MSDW. Output: 
   `PSYCH_OUTPATIENT_NOTES_date.csv`
   
   ```
    python prepare_dataset.py 
    ``` 

    Columns refactoring:
    
    > AUDIT_KEY == NOTE_ID (from fact data)

    > VALUE == LINE, NOTE_TEXT (from fact data) 

    > PERSON_KEY == PAT_ID (from fact data)

    > LEVEL3_ACTION_NAME == STATUS (from metadata table)

    > LEVEL2_EVENT_NAME == NOTE_TYPE (from metadata table)

    > CALENDAR_DATE == CONTACT_DATE (from calendar table)

    > HOURS, MINUTES == SPECIFIED_DATETIME (from time of day table)

    > MEDICAL_RECORD_NUMBER == MRN (from person data)

    Other columns: ENCOUNTER_KEY, BEGIN_DATE_TIME, END_DATE_TIME,
    EFFECTIVE_START_DATE_TIME, ENCOUNTER_KEY, ENCOUNTER_VISIT_ID, META_DATA_KEY
    
    Output files: `PSYCH_OUTPATIENT_NOTES_*.csv` with date in which the file was created.
    
3. Create outpatient notes concatenating lines. 
    
   ```
   python concatenate_notes.py
   ```
   
    Output: `outpatient_notes.csv` file. 
    Notes that miss few lines are concatenated regardless, notes with only one line (i.e., LINE==0) are dropped, 
    lines with LINE=0 at the beginning of a multi-line note are dropped. Empty notes (i.e., NOTE_TEXT==NA) are discarded.
    
    **Remark**: after investigation, it seems like notes longer than 20 lines are shortened. `META_DATA_KEY==4015` 
    is assigned to `LEVEL4_FIELD_NAME=='Note Text[21]'` which stores the following text in `NOTE_TEXT` column created 
    by `prepare_dataset.py` script: "Additional Progress Note Text is available in Epic or upon request from the Data 
    Warehouse team". Hence, we append the text at the end of those notes that are not reported in full.
    
    Columns: `'PAT_ID', 'MRN', 'SYSTEM_NAME', 'CONTACT_DATE', 'ENCOUNTER_KEY',
       'FACILITY_KEY', 'NOTE_ID', 'SPECIFIED_DATETIME', 'META_DATA_KEY',
       'NOTE_TYPE', 'STATUS', 'DATA_FEED_KEY', 'DATA_STATE_KEY',
       'ACCOUNTING_GROUP_KEY', 'KEYWORD_GROUP_KEY', 'DATA_QUALITY_KEY',
       'PROCEDURE_GROUP_KEY', 'DIAGNOSIS_GROUP_KEY', 'FACT_KEY',
       'PAYOR_GROUP_KEY', 'OPERATION_KEY', 'CAREGIVER_GROUP_KEY',
       'UOM_KEY', 'ORGANIZATION_GROUP_KEY', 'MATERIAL_GROUP_KEY',
       'NOTE_TEXT'`
    
4. To easily load the notes we saved the outpatient notes dataframe as a numpy array in a `pkl` object `outpatient_notes.pkl` 
   which includes a tuple with first element an array with the column names listed in 3. and as second element an array with the actual 
   values. This object can be investigated with notebook `len_notes_stats.ipynb` where character length of notes is first 
   log transformed and then three different gaussian distributions are fitted to the data. Only notes with character length 
   (after special characters removal and lower case transformation) in the medium/long distributions are considered. This, 
   after computing the average Tfi-df weights for each word and considering that the shortest notes are uninformative templates. 
   The output is a pickle object `medlong_outpatient_notes.pkl` which only includes notes from medium/long distributions 
   and also two new columns, i.e., "TOKENS" and "SENTENCES" with tokenized notes, by word and sentence, respectively.
   **Remark**: the object with medium/long outpatient notes can be also created running `len_notes_stats.py`.
   
   **Update**: we labeled note types when available and sample from labeled notes those from patients with a certain 
   number of notes and note lengths. Selected note ids with labels can be found in `opud.pkl`, which is now loaded at 
   the beginning of the `len_notes_stats.py` script to create `filtered_outpatient_notes.pkl`. 
   
5. Redundancy investigation.
   The module `redundancy_detection.py` includes three different methods to investigate different types of redundancy:
   
   - **Within-note redundancy**: this method 
   
   > `python -m redundancy_detection -r wn -o within_note_redundancy.pkl`

   > `python -m redundancy_detection -r bn -o between_note_redundancy.pkl`

   To run the between patient redundancy, first initialize the redundancy threshold to consider in utils, then run:
   
   > `python -m redundancy_detection -r bp -o between_patient_redundancy.pkl`

6. Create datasets for language modeling
    The module `create_dataset.py` prepares the input to language modeling. In particular:
    
    > `python -m create_dataset -dt wn -co wn_sentences -o wn_dataset`

    Creates two datasets, one with tokenized raw notes and the other one with tokenized notes with within-note redundancy 
    dropped.
    
    Output files:
    
    - wn_sentences_train|test.csv (csv file with tokenized sentences, one per row);
    - raw_wn_sentences_train|test.csv (csv file with tokenized raw sentences, one per row)
    - wn_datasets.pkl (pkl object with tuple of train and test dictionaries {mrn: idx: [tokenized sentences]})
    - raw_wn_datasets.pkl (pkl object with tuple of train and test dictionaries {mrn: idx: [tokenized raw sentences]})
    
    > `python -m create_dataset -dt bn -co bn_sentences -o bn_dataset`

    Creates two datasets, one with tokenized raw notes and one with the tokenized notes with between-note redundancy 
    dropped. **Remark**: only specific MRNs with same number of notes are selected for this task.
    
    Output files:
    - bn_sentences_train|test.csv (csv file with tokenized sentences, one per row);
    - raw_bn_sentences_train|test.csv (csv file with tokenized raw sentences, one per row)
    - bn_datasets.pkl (pkl object with tuple of train and test dictionaries {mrn: idx: [tokenized sentences]})
    - raw_bn_datasets.pkl (pkl object with tuple of train and test dictionaries {mrn: idx: [tokenized raw sentences]})
   
7. Train word2vec embeddings. After choosing the hyperparameters in `utils.py` run the CBOW model as:
    ```
   python -m cbow
   ```
     