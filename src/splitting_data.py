import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("/root/MIMICIV/src/landmark_df_evo_correct.csv", na_values=['', 'None', 'NaN', 'na', 'nan'])
data.fillna('', inplace=True)

# Fai una divisione separata per ciascuna sottopopolazione
train_list, val_list, test_list = [], [], []
visit_counts = data['subject_id'].value_counts()


for visit_count in range(1, 6):  # Visite 1 a 5
    selected_patients = visit_counts[visit_counts == visit_count].index
    subset = data[data['subject_id'].isin(selected_patients)].copy()

    patients = subset['subject_id'].unique()
    train_val_patients, test_patients = train_test_split(
        patients,
        test_size=0.2,
        random_state=9550,
        stratify=subset.groupby('subject_id')['death_in_90days'].max()
    )

    train_val_subset = subset[subset['subject_id'].isin(train_val_patients)]
    test_subset = subset[subset['subject_id'].isin(test_patients)]

    # Split ulteriore per validation
    train_patients, val_patients = train_test_split(
        train_val_subset['subject_id'].unique(), test_size=0.1, random_state=9550,
        stratify=train_val_subset.groupby('subject_id')['death_in_90days'].max()
    )

    train_subset = train_val_subset[train_val_subset['subject_id'].isin(train_patients)]
    val_subset = train_val_subset[train_val_subset['subject_id'].isin(val_patients)]

    train_list.append(train_subset)
    val_list.append(val_subset)
    test_list.append(test_subset)

# Unisci le varie sottopopolazioni
train_final = pd.concat(train_list).reset_index(drop=True)
val_final = pd.concat(val_list).reset_index(drop=True)
test_final = pd.concat(test_list).reset_index(drop=True)

# Salvo i set in file separati
train_final.to_csv("/mnt/vdb/data/landmark_evo_train.csv", index=False)
val_final.to_csv("/mnt/vdb/data/landmark_evo_vali.csv", index=False)
test_final.to_csv("/mnt/vdb/data/landmark_evo_test.csv", index=False)