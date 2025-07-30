import random
import ast
import pandas as pd

from collections import Counter


def no_narrative_prompt(row):
    narrative = "Based on this information, what is the probability of mortality within 90 days?"
    if pd.notna(row['diag_text']) and row['diag_text'].strip():
        narrative += f"Diagnoses: {set(row['diag_text'])}"
    if pd.notna(row['med_text']) and row['med_text'].strip():
        narrative += f"Medications: {set(row['med_text'])}"
    if pd.notna(row['proc_text']) and row['proc_text'].strip():
        narrative += f"Procedures: {set(row['proc_text'])}"

    return narrative

def naive_narrative_prompt(row):
    narrative = f"Patient is a {row['age_at_landmark']}-year-old {row['gender']}."
    narrative += f" This is the {row['landmark_visit']} visit."
    if row['days_since_previous_visit'] != -1:
        narrative += f" The last visit happened {row['days_since_previous_visit']} days ago."
    if pd.notna(row['diag_text']) and row['diag_text'].strip():
        narrative += f" Medical history includes: {set(row['diag_text'])}."
    if pd.notna(row['med_text']) and row['med_text'].strip():
        narrative += f" Current medications are: {set(row['med_text'])}."
    if pd.notna(row['proc_text']) and row['proc_text'].strip():
        narrative += f" Procedures performed: {set(row['proc_text'])}."
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

def compact_no_time_prompt(row):
    narrative = f"You are a Doctor.\nWhat is the probability of death in the next 90 days for {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
    narrative += f"Visit type: {row['admission_category']}\n"

    current_visit = int(row['landmark_visit'])

    # --- Diagnosi ---
    narrative += "\nDIAGNOSIS HISTORY:"
    diag_per_visit = ast.literal_eval(row['diag_per_visit'])

    # Eventi da visite precedenti
    past_diags = [diag for key, values in diag_per_visit.items() if int(key) != current_visit for diag in values]
    past_diag_counts = Counter(past_diags)
    chronic_diags = [diag for diag, count in past_diag_counts.items() if count >= 2]

    # Eventi dell'ultima visita
    current_diags = diag_per_visit.get(current_visit, [])

    # Unione e conteggio finale
    final_diags = chronic_diags + current_diags
    diag_summary = Counter(final_diags)
    if diag_summary:
        diag_str = '; '.join([f"{d}" for d, _ in diag_summary.items()])
        narrative += f"\nDiagnoses: {diag_str}."
    else:
        narrative += "\nNo diagnoses recorded."

    # --- Farmaci ---
    narrative += "\nPRESCRIPTIONS HISTORY:"
    meds_per_visit = ast.literal_eval(row['meds_per_visit'])

    past_meds = [med for key, values in meds_per_visit.items() if int(key) != current_visit for med in values]
    past_meds_counts = Counter(past_meds)
    chronic_meds = [med for med, count in past_meds_counts.items() if count >= 2]

    current_meds = meds_per_visit.get(current_visit, [])
    final_meds = chronic_meds + current_meds
    med_summary = Counter(final_meds)
    if med_summary:
        med_str = '; '.join([f"{m}" for m, _ in med_summary.items()])
        narrative += f"\nMedications: {med_str}."
    else:
        narrative += "\nNo medications recorded."

    # --- Procedure ---
    narrative += "\nPROCEDURES HISTORY:"
    proc_per_visit = ast.literal_eval(row['proc_per_visit'])

    past_procs = [proc for key, values in proc_per_visit.items() if int(key) != current_visit for proc in values]
    past_proc_counts = Counter(past_procs)
    chronic_procs = [proc for proc, count in past_proc_counts.items() if count >= 2]

    current_procs = proc_per_visit.get(current_visit, [])
    final_procs = chronic_procs + current_procs
    proc_summary = Counter(final_procs)
    if proc_summary:
        proc_str = '; '.join([f"{p}" for p, _ in proc_summary.items()])
        narrative += f"\nProcedures: {proc_str}."
    else:
        narrative += "\nNo procedures recorded."

    return narrative

def compact_no_time_prompt_rnd(row):
    narrative = f"You are a Doctor.\nWhat is the probability of death in the next 90 days for {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
    narrative += f"Visit type: {row['admission_category']}\n"

    current_visit = int(row['landmark_visit'])

    # --- Diagnosi ---
    narrative += "\nDIAGNOSIS HISTORY:"
    diag_per_visit = ast.literal_eval(row['diag_per_visit'])

    past_diags = [diag for key, values in diag_per_visit.items() if int(key) != current_visit for diag in values]
    past_diag_counts = Counter(past_diags)
    chronic_diags = [diag for diag, count in past_diag_counts.items() if count >= 2]

    current_diags = diag_per_visit.get(current_visit, [])
    final_diags = chronic_diags + current_diags

    if final_diags:
        random.shuffle(final_diags)  # <-- MISCELAZIONE
        diag_str = '; '.join(final_diags)
        narrative += f"\nDiagnoses: {diag_str}."
    else:
        narrative += "\nNo diagnoses recorded."

    # --- Farmaci ---
    narrative += "\nPRESCRIPTIONS HISTORY:"
    meds_per_visit = ast.literal_eval(row['meds_per_visit'])

    past_meds = [med for key, values in meds_per_visit.items() if int(key) != current_visit for med in values]
    past_meds_counts = Counter(past_meds)
    chronic_meds = [med for med, count in past_meds_counts.items() if count >= 2]

    current_meds = meds_per_visit.get(current_visit, [])
    final_meds = chronic_meds + current_meds

    if final_meds:
        random.shuffle(final_meds)  # <-- MISCELAZIONE
        med_str = '; '.join(final_meds)
        narrative += f"\nMedications: {med_str}."
    else:
        narrative += "\nNo medications recorded."

    # --- Procedure ---
    narrative += "\nPROCEDURES HISTORY:"
    proc_per_visit = ast.literal_eval(row['proc_per_visit'])

    past_procs = [proc for key, values in proc_per_visit.items() if int(key) != current_visit for proc in values]
    past_proc_counts = Counter(past_procs)
    chronic_procs = [proc for proc, count in past_proc_counts.items() if count >= 2]

    current_procs = proc_per_visit.get(current_visit, [])
    final_procs = chronic_procs + current_procs

    if final_procs:
        random.shuffle(final_procs)  # <-- MISCELAZIONE
        proc_str = '; '.join(final_procs)
        narrative += f"\nProcedures: {proc_str}."
    else:
        narrative += "\nNo procedures recorded."

    return narrative

def full_narrative(row):
    narrative = f"You are a Doctor.\nWhat is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
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
            narrative += f"\n{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit -1])} days ago: {'; '.join(diags)}."

    narrative += "\nPrescriptions history:"
    for past_visit, meds in reversed(ast.literal_eval(row['meds_per_visit']).items()):
        if past_visit ==  max_visit:
            narrative += f"\nToday: {'; '.join(meds)}."
        else:
            narrative += f"\n{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit-1])} days ago: {'; '.join(meds)}."

    narrative += "\nProcedures history:"
    for past_visit, proc in reversed(ast.literal_eval(row['proc_per_visit']).items()):
        if past_visit == max_visit:
            narrative += f"\nToday: {'; '.join(proc)}."
        else:
            narrative += f"\n{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit-1])} days ago: {'; '.join(proc)}."
    
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

def compact_narrative_humanstyle_prompt(row):
    

    narrative = f"You are a Doctor.\n"
    narrative += f"A {row['age_at_landmark']}-year-old {row['gender']} patient is currently hospitalized for a {row['admission_category'].lower()} admission (visit number {row['landmark_visit']}).\n"

    # Temporal hint if available
    if row['days_since_previous_visit'] != -1:
        narrative += f"The previous visit occurred {row['days_since_previous_visit']} days ago.\n"

    current_visit = int(row['landmark_visit'])

    # --- Diagnoses ---
    diag_per_visit = ast.literal_eval(row['diag_per_visit'])
    all_diags = [d for visit_diags in diag_per_visit.values() for d in visit_diags]
    diag_counts = Counter(all_diags)

    chronic_diags = [d for d, c in diag_counts.items() if c >= 2]
    current_diags = diag_per_visit.get(current_visit, [])
    new_diags = [d for d in current_diags if diag_counts[d] == 1]

    if chronic_diags:
        narrative += f"The patient has a chronic history of: {', '.join(chronic_diags)}.\n"
    if new_diags:
        narrative += f"During the current visit, new diagnoses include: {', '.join(new_diags)}.\n"
    if not chronic_diags and not new_diags:
        narrative += "There are no recorded diagnoses so far.\n"

    # --- Medications ---
    meds_per_visit = ast.literal_eval(row['meds_per_visit'])
    all_meds = [m for visit_meds in meds_per_visit.values() for m in visit_meds]
    med_counts = Counter(all_meds)

    chronic_meds = [m for m, c in med_counts.items() if c >= 2]
    current_meds = meds_per_visit.get(current_visit, [])
    new_meds = [m for m in current_meds if med_counts[m] == 1]

    if chronic_meds:
        narrative += f"Ongoing medications include: {', '.join(chronic_meds)}.\n"
    if new_meds:
        narrative += f"New medications prescribed in this visit are: {', '.join(new_meds)}.\n"
    if not chronic_meds and not new_meds:
        narrative += "No medications have been prescribed so far.\n"

    # --- Procedures ---
    proc_per_visit = ast.literal_eval(row['proc_per_visit'])
    all_procs = [p for visit_procs in proc_per_visit.values() for p in visit_procs]
    proc_counts = Counter(all_procs)

    chronic_procs = [p for p, c in proc_counts.items() if c >= 2]
    current_procs = proc_per_visit.get(current_visit, [])
    new_procs = [p for p in current_procs if proc_counts[p] == 1]

    if chronic_procs:
        narrative += f"The patient has undergone repeated procedures such as: {', '.join(chronic_procs)}.\n"
    if new_procs:
        narrative += f"New procedures during this visit include: {', '.join(new_procs)}.\n"
    if not chronic_procs and not new_procs:
        narrative += "No procedures have been recorded.\n"

    narrative += "\nWhat is the probability of death in the next 90 days?"

    return narrative