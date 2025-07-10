import pandas as pd
import ast
from collections import Counter
import random

def no_narrative_prompt(row):
    narrative = "Variables: "
    if pd.notna(row['diag_text']) and row['diag_text'].strip():
        narrative += f" {row['diag_text']}"
    if pd.notna(row['med_text']) and row['med_text'].strip():
        narrative += f" {row['med_text']}"
    if pd.notna(row['proc_text']) and row['proc_text'].strip():
        narrative += f" {row['proc_text']}"

    return narrative

def naive_narrative_prompt(row):
    narrative = f"Patient is a {row['age_at_landmark']}-year-old {row['gender']}."
    narrative += f" This is the {row['num_total_visits']} visit."
    if row['days_since_previous_visit'] != -1:
        narrative += f" The last visit happened {row['days_since_previous_visit']} days ago."
    if pd.notna(row['diag_text']) and row['diag_text'].strip():
        narrative += f" Medical history includes: {row['diag_text']}."
    if pd.notna(row['med_text']) and row['med_text'].strip():
        narrative += f" Current medications are: {row['med_text']}."
    if pd.notna(row['proc_text']) and row['proc_text'].strip():
        narrative += f" Procedures performed: {row['proc_text']}."
    # Add explicit prediction question
    narrative += " Based on this information, what is the probability of mortality within 90 days?"
    return narrative

def compact_narrative_prompt(row):
    narrative = f"You are a Doctor.\nWhat is the probability of death in the next 90 days for {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
    current_visit = row['landmark_visit']
    type = row['admission_category']
    narrative += f"Visit number {current_visit} - {type} \n"

    if row['days_since_previous_visit'] != -1:
        narrative += f"Last visit happened {row['days_since_previous_visit']} days ago."

    # --- Diagnoses ---
    narrative += "\nDIAGNOSIS HISTORY:"
    diag_per_visit = ast.literal_eval(row['diag_per_visit'])
    
    # Frequency count of diagnoses
    all_diags = []
    for diags in diag_per_visit.values():
        all_diags.extend(diags)
    diag_counts = Counter(all_diags)

    # Chronic diagnoses are those that appear in at least 2 visits
    chronic_diags = [d for d, c in diag_counts.items() if c >= 2]

    # New diagnoses are those that appear only in the current visit
    current_diags = diag_per_visit[int(current_visit)]
    new_diags = [d for d in current_diags if diag_counts[d] == 1]

    if chronic_diags:
        narrative += f"\nChronic diagnoses: {'; '.join(chronic_diags)}."
    if new_diags:
        narrative += f"\nNew diagnoses in this visit: {'; '.join(new_diags)}."
    if not chronic_diags and not new_diags:
        narrative += "\nNo diagnoses recorded."

    # --- Medicines ---
    narrative += "\nPRESCRIPTIONS HISTORY:"
    meds_per_visit = ast.literal_eval(row['meds_per_visit'])
    
    all_meds = []
    for meds in meds_per_visit.values():
        all_meds.extend(meds)
    med_counts = Counter(all_meds)

    chronic_meds = [m for m, c in med_counts.items() if c >= 2]
    current_meds = meds_per_visit[int(current_visit)]
    new_meds = [m for m in current_meds if med_counts[m] == 1]

    if chronic_meds:
        narrative += f"\nChronic medications: {'; '.join(chronic_meds)}."
    if new_meds:
        narrative += f"\nNew medications in this visit: {'; '.join(new_meds)}."
    if not chronic_meds and not new_meds:
        narrative += "\nNo medications recorded."

    # --- Procedure ---
    narrative += "\nPROCEDURES HISTORY:"
    proc_per_visit = ast.literal_eval(row['proc_per_visit'])

    all_proc = []
    for procs in proc_per_visit.values():
        all_proc.extend(procs)
    proc_counts = Counter(all_proc)

    chronic_proc = [p for p, c in proc_counts.items() if c >= 2]
    current_proc = proc_per_visit[int(current_visit)]
    new_proc = [p for p in current_proc if proc_counts[p] == 1]

    if chronic_proc:
        narrative += f"\nChronic procedures: {'; '.join(chronic_proc)}."
    if new_proc:
        narrative += f"\nNew procedures in this visit: {'; '.join(new_proc)}."
    if not chronic_proc and not new_proc:
        narrative += "\nNo procedures recorded."

    return narrative

def full_narrative(row):
    narrative = f"You are a Doctor.\nWhat is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient\n"
    current_visit = row['landmark_visit']
    narrative += f"Today is the {current_visit} visit.\n"

    max_visit = int(row['landmark_visit'])

    # if row['days_since_previous_visit'] != -1:
    #     narrative += f"Last visit happened {row['days_since_previous_visit']} days ago."

    narrative += "\nDiagnosis history:"
    for past_visit, diags in reversed(list(ast.literal_eval(row['diag_per_visit']).items())):
        if past_visit == max_visit:
            narrative += f"\nToday: {'; '.join(diags)}."
        else:
            narrative += f"\n{int(row['days_since_previous_visit_cumulate_sum'][int(past_visit) -1])} days ago: {'; '.join(diags)}."

    narrative += "\nPrescriptions history:"
    for past_visit, meds in reversed(ast.literal_eval(row['meds_per_visit']).items()):
        if past_visit ==  max_visit:
            narrative += f"\nToday: {'; '.join(meds)}."
        else:
            narrative += f"\n{int(row['days_since_previous_visit_cumulate_sum'][past_visit-1])} days ago: {'; '.join(meds)}."

    narrative += "\nProcedures history:"
    for past_visit, proc in reversed(ast.literal_eval(row['proc_per_visit']).items()):
        if past_visit == max_visit:
            narrative += f"\nToday: {'; '.join(proc)}."
        else:
            narrative += f"\n{int(row['days_since_previous_visit_cumulate_sum'][past_visit-1])} days ago: {'; '.join(proc)}."
    
    return narrative

def full_narrative_no_time(row):
    narrative = f"You are a Doctor.\nWhat is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient?\n"

    max_visit = int(row['landmark_visit'])

    # if row['days_since_previous_visit'] != -1:
    #     narrative += f"Last visit happened {row['days_since_previous_visit']} days ago."

    narrative += "\nDiagnosis history:"
    for _, diags in reversed(list(ast.literal_eval(row['diag_per_visit']).items())):
        narrative += f"\n{'; '.join(diags)}."
        
    narrative += "\nPrescriptions history:"
    for _, meds in reversed(ast.literal_eval(row['meds_per_visit']).items()):
        narrative += f"\n{'; '.join(meds)}."
    
    narrative += "\nProcedures history:"
    for _, proc in reversed(ast.literal_eval(row['proc_per_visit']).items()):
        narrative += f"\n{'; '.join(proc)}."
    
    return narrative

def full_narrative_no_time_rnd(row):
    narrative = f"You are a Doctor.\nWhat is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient?\n"

    # Diagnosi
    all_diags = []
    diag_dict = ast.literal_eval(row['diag_per_visit'])
    for diags in diag_dict.values():
        all_diags.extend(diags)
    random.shuffle(all_diags)
    narrative += "\nDiagnosis history:\n" + '; '.join(all_diags) + "."

    # Prescrizioni
    all_meds = []
    meds_dict = ast.literal_eval(row['meds_per_visit'])
    for meds in meds_dict.values():
        all_meds.extend(meds)
    random.shuffle(all_meds)
    narrative += "\nPrescriptions history:\n" + '; '.join(all_meds) + "."

    # Procedure
    all_procs = []
    proc_dict = ast.literal_eval(row['proc_per_visit'])
    for procs in proc_dict.values():
        all_procs.extend(procs)
    random.shuffle(all_procs)
    narrative += "\nProcedures history:\n" + '; '.join(all_procs) + "."

    return narrative
