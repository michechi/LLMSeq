> Historical exploratory workflow. Clinical records and generated clinical text must remain within credentialed storage; references below to external review do not authorize public release.

# Notes on different MIMIC-IV preprocessing approaches

With PyHealth we are making a subset of the whole patients (at least, it seems to be), here instead we prefer to focus more on the data, trying to understand all the info contained.

## Description of text-summary data preprocessing in preprocessiong_no_pyhealth.ipynb

This Python script processes clinical data from the MIMIC-IV database to generate structured clinical histories for patients. The data includes patient demographics, hospital admissions, ICU stays, diagnoses, prescriptions, and procedures. The script performs the following key steps:

1. **Loading Data:** Load various datasets representing different aspects of patient information, such as demographics, hospital admissions, ICU stays, prescriptions, and medical procedures.

2. **Mapping ICD Codes to Descriptions:** Merge diagnoses and procedures with their respective ICD code descriptions to provide a more comprehensive understanding of patient records.

3. **Computing Age at Events:** Calculate the age of patients at the time of different medical events such as hospital admissions, ICU stays, and prescriptions. This step includes calculating the age at death for deceased patients.

4. **Generating Structured Clinical History:** For each patient, compile a structured clinical history that includes details about admissions, diagnoses, procedures, prescribed medications, and ICU stays. The script also determines the patient's age and records any deceased status.

5. **Generating and Saving Summaries:** Create clinical history summaries for each patient and save them to a file for further analysis or reporting.

### Detailed Documentation

#### Step 1: Load Data

- Import necessary libraries and load datasets using `pandas.read_csv` to read from CSV files.
- The datasets include:
  - `patients`: Demographic information.
  - `admissions`: Hospital admission records.
  - `icustays`: ICU stay records.
  - `diagnoses`: ICD codes for patient diagnoses.
  - `prescriptions`: Medications administered to patients.
  - `procedures`: ICD codes for medical procedures.
  - `icd_diagnoses`, `icd_procedures`: Descriptions of ICD codes for diagnoses and procedures respectively.
- Convert ICU stay timestamps ('intime' and 'outtime') to `datetime` format for further processing.

#### Step 2: Map ICD Codes to Descriptions

- Merge diagnosis and procedure data with their respective ICD descriptions using a left join (`merge`) on `icd_code` and `icd_version` to add informative descriptions to the datasets.
- Handle missing descriptions by filling them with "Unknown diagnosis" or "Unknown procedure".
- Retain only necessary columns for further analysis:
    * diagnosis: `['subject_id', 'hadm_id', 'icd_version', 'diagnosis_description', 'icd_code']`
    * procedures: `['subject_id', 'hadm_id', 'icd_version', 'procedure_description', 'icd_code']`

#### Step 3: Compute Age at Events

- Filter patient data to retain only relevant i.e. `subject_id`, `gender`, `anchor_age`, `anchor_year`, `dod`.
- Merge patient demographic data with other datasets (admissions, ICU stays, diagnoses, procedures, prescriptions) to include age information.
- Convert relevant date fields (`admittime`, `intime`, and `starttime`) to `datetime` objects.
- Calculate patient age at event time using the auxiliary function `compute_age_at_event`, which adapts age calculations to each type of event's timestamp.
- Directly assign age for events (diagnoses and procedures) lacking precise timestamps.

#### Step 4: Generate Structured Clinical History

- Define the function `generate_patient_history` to compile clinical histories for individual patients.
- For each patient, include gender, death status, and age at death if applicable.
- Sort and iterate through patient hospitalizations, providing details such as:
  - Admission type (*emergency* or *normal*) and discharge status.
  - Diagnoses and procedures associated with each hospitalization.
  - Medications prescribed, detailing dosage and administration route.
  - Summary of ICU stays with length of stay, handling missing discharge dates (if missing we do not compute the icu stay length) to maintain informational integrity.

#### Step 5: Generate and Save Summaries

- Compile summaries for a selection of patients (restrained to 100 for testing) TODO: FOR ALL.
- Utilize a dictionary comprehension to generate and store histories in `summaries`.
- Save all clinical histories to a text file (`clinical_histories.txt`) for external review and use.


## Description of tabular-summary data preprocessing in preprocessiong_no_pyhealth.ipynb

This Python script processes clinical data from the MIMIC-IV database to generate a tabular dataset from textual patient history. It compiles key clinical features like admissions, diagnoses, procedures, ICU stays, and prescriptions into a structured format suitable for machine learning and analysis.

### Detailed Documentation

#### Step 1: Load Data

- Imports necessary libraries and loads datasets using `pandas.read_csv`.
- Datasets include demographics (`patients`), admissions (`admissions`), ICU stays (`icustays`), medical records (`diagnoses`, `procedures`), and medication prescriptions (`prescriptions`).
- Descriptions for ICD codes are loaded to provide context to diagnoses and procedures.

#### Step 2: Map ICD Codes to Descriptions

- Merges diagnosis and procedure data with descriptions using a left merge on `icd_code` and `icd_version`.
- Ensures the inclusion of a human-readable description of diagnoses and procedures within patient records.

#### Step 3: Compute Age at Events & Mortality

- Filters patient data to retain core columns including `subject_id`, `gender`, `anchor_age`, etc.
- Merges demographic data with main tables to incorporate patient age and mortality status.
- Calculations:
  - `age_at_event` for hospital admissions and prescriptions.
  - Mortality indicators `death_flag` and `age_at_death`.

#### Step 4: Build Tabular Dataset

- Aggregates key features:
  - **ICU Stays**: Counts ICU admissions and calculates total stay days per hospitalization.
  - **Diagnoses**: Summarizes the number and descriptions of diagnoses per hospitalization.
  - **Procedures**: Counts and lists unique procedures performed.
  - **Medications**: Counts and lists prescribed drugs.
- Merges aggregated summaries with core admission features for a comprehensive dataset.
- Handles missing values by filling numerical with zeros and textual lists with "None".

#### Step 5: Save & Display Dataset

- Saves the compiled tabular dataset to a CSV file: `mimiciv_clinical_dataset_tabular.csv`.
- Prepares the dataset for further exploration or machine learning analysis, offering a concise view of patient histories in a structured, numerical format.

--- 

This description focuses on the essential functionality and steps performed by your script. Let me know if you need anything else!