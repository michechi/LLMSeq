import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score

# Reading data - already splitted!
X_train = pd.read_csv("data/simulation/X_s_train_2.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_train = pd.read_csv("data/simulation/y_s_train_2.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_val = pd.read_csv("data/simulation/X_s_val_2.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_val = pd.read_csv("data/simulation/y_s_val_2.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

# Estrai prima lettera
train_first = [seq.split('\x1f')[0] for seq in X_train['Sequences']]
val_first = [seq.split('\x1f')[0] for seq in X_val['Sequences']]

# Calcola P(label=1 | lettera) dal training
letter_stats = defaultdict(lambda: [0, 0])  # [count_label_0, count_label_1]
for letter, label in zip(train_first, y_train['Outcome']):
    letter_stats[letter][label] += 1

# Predici sul validation
val_preds = []
for letter in val_first:
    counts = letter_stats[letter]
    prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
    val_preds.append(prob)

baseline_auc = roc_auc_score(y_val['Outcome'], val_preds)
print(f"Baseline AUC (solo prima lettera): {baseline_auc:.4f}")

# Test on all letters
for pos in range(9):
    pos_letters = [seq.split('\x1f')[pos] for seq in X_train['Sequences']]
    
    letter_stats = defaultdict(lambda: [0, 0])
    for letter, label in zip(pos_letters, y_train['Outcome']):
        letter_stats[letter][label] += 1
    
    val_pos_letters = [seq.split('\x1f')[pos] for seq in X_val['Sequences']]
    val_preds = []
    for letter in val_pos_letters:
        counts = letter_stats[letter]
        prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
        val_preds.append(prob)
    
    pos_auc = roc_auc_score(y_val['Outcome'], val_preds)
    print(f"Position {pos+1} AUC: {pos_auc:.4f}")

# Test on all letters cumulating
def cumulative_position_analysis(X_train, y_train, X_val, y_val):
    """Test informazione cumulativa: prime k lettere"""
    
    results = []
    
    for k in range(1, 10):  # Da 1 a 9 lettere
        # Estrai prime k lettere da ogni sequenza
        train_prefixes = ['\x1f'.join(seq.split('\x1f')[:k]) for seq in X_train['Sequences']]
        val_prefixes = ['\x1f'.join(seq.split('\x1f')[:k]) for seq in X_val['Sequences']]
        
        # Calcola P(label=1 | prefix) dal training
        prefix_stats = defaultdict(lambda: [0, 0])
        for prefix, label in zip(train_prefixes, y_train['Outcome']):
            prefix_stats[prefix][label] += 1
        
        # Predici sul validation
        val_preds = []
        for prefix in val_prefixes:
            counts = prefix_stats[prefix]
            prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
            val_preds.append(prob)
        
        auc = roc_auc_score(y_val['Outcome'], val_preds)
        
        # Calcola anche coverage: quanti prefix unici visti in train
        unique_train = len(set(train_prefixes))
        unique_val = len(set(val_prefixes))
        coverage = len(set(val_prefixes) & set(train_prefixes)) / unique_val * 100
        
        print(f"Prime {k} lettere | AUC: {auc:.4f} | "
              f"Unique train: {unique_train:,} | "
              f"Coverage val: {coverage:.1f}%")
        
        results.append({
            'k': k,
            'auc': auc,
            'unique_train': unique_train,
            'coverage': coverage
        })
    
    return results

results = cumulative_position_analysis(X_train, y_train, X_val, y_val)

# Test on bigrams
def extract_bigrams(seq):
    letters = seq.split('\x1f')
    return [f"{letters[i]}-{letters[i+1]}" for i in range(len(letters)-1)]

def bigram_distribution_analysis(X_train, y_train, X_val, y_val):
    """Analizza se sequenze ordinate/non-ordinate hanno distribuzioni di bigrammi diverse"""
    
    # Conta bigrammi per classe
    bigrams_class_0 = Counter()
    bigrams_class_1 = Counter()
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        bigrams = extract_bigrams(seq)
        if label == 0:
            bigrams_class_0.update(bigrams)
        else:
            bigrams_class_1.update(bigrams)
    
    # Normalizza in probabilità
    total_0 = sum(bigrams_class_0.values())
    total_1 = sum(bigrams_class_1.values())
    
    prob_0 = {bg: count/total_0 for bg, count in bigrams_class_0.items()}
    prob_1 = {bg: count/total_1 for bg, count in bigrams_class_1.items()}
    
    # Trova bigrammi più discriminativi
    all_bigrams = set(prob_0.keys()) | set(prob_1.keys())
    bigram_scores = []
    
    for bg in all_bigrams:
        p0 = prob_0.get(bg, 0)
        p1 = prob_1.get(bg, 0)
        diff = abs(p1 - p0)
        bigram_scores.append((bg, p0, p1, diff))
    
    # Ordina per differenza
    bigram_scores.sort(key=lambda x: x[3], reverse=True)
    
    print("Top 20 bigrammi più discriminativi:")
    print("Bigram | P(bg|label=0) | P(bg|label=1) | Differenza")
    for bg, p0, p1, diff in bigram_scores[:20]:
        print(f"{bg:8} | {p0:13.4f} | {p1:13.4f} | {diff:10.4f}")
    
    return bigram_scores

def bigram_baseline_classifier(X_train, y_train, X_val, y_val):
    """Classifica basandosi sulla distribuzione di bigrammi nella sequenza"""
    
    # Calcola P(label=1 | bigram) per ogni bigram
    bigram_stats = defaultdict(lambda: [0, 0])  # [count_0, count_1]
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        bigrams = extract_bigrams(seq)
        for bg in bigrams:
            bigram_stats[bg][label] += 1
    
    # Converti in probabilità
    bigram_probs = {}
    for bg, counts in bigram_stats.items():
        total = sum(counts)
        bigram_probs[bg] = counts[1] / total if total > 0 else 0.5
    
    # Predici: usa media delle probabilità dei bigrammi nella sequenza
    val_preds = []
    for seq in X_val['Sequences']:
        bigrams = extract_bigrams(seq)
        probs = [bigram_probs.get(bg, 0.5) for bg in bigrams]
        avg_prob = np.mean(probs) if probs else 0.5
        val_preds.append(avg_prob)
    
    auc = roc_auc_score(y_val['Outcome'], val_preds)
    print(f"\nBigram Baseline AUC: {auc:.4f}")
    
    return auc

# Esegui le analisi
print("=" * 60)
print("ANALISI DISTRIBUZIONE BIGRAMMI")
print("=" * 60)
bigram_scores = bigram_distribution_analysis(X_train, y_train, X_val, y_val)

print("\n" + "=" * 60)
print("BASELINE CLASSIFIER BASATO SU BIGRAMMI")
print("=" * 60)
bigram_auc = bigram_baseline_classifier(X_train, y_train, X_val, y_val)

# Let's use also F1 score
def bigram_baseline_classifier_with_f1(X_train, y_train, X_val, y_val):
    """Classifica basandosi sulla distribuzione di bigrammi - con F1"""
    
    # Calcola P(label=1 | bigram)
    bigram_stats = defaultdict(lambda: [0, 0])
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        bigrams = extract_bigrams(seq)
        for bg in bigrams:
            bigram_stats[bg][label] += 1
    
    bigram_probs = {}
    for bg, counts in bigram_stats.items():
        total = sum(counts)
        bigram_probs[bg] = counts[1] / total if total > 0 else 0.5
    
    # Predici
    val_preds_prob = []
    for seq in X_val['Sequences']:
        bigrams = extract_bigrams(seq)
        probs = [bigram_probs.get(bg, 0.5) for bg in bigrams]
        avg_prob = np.mean(probs) if probs else 0.5
        val_preds_prob.append(avg_prob)
    
    # Calcola metriche
    auc = roc_auc_score(y_val['Outcome'], val_preds_prob)
    
    # Binarizza con soglia 0.5 per F1
    val_preds_binary = [1 if p >= 0.5 else 0 for p in val_preds_prob]
    f1 = f1_score(y_val['Outcome'], val_preds_binary)
    
    # Prova anche con soglia ottimale (quella che massimizza F1)
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in val_preds_prob]
        f1_scores.append(f1_score(y_val['Outcome'], preds))
    
    best_f1 = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    print(f"Bigram Baseline AUC: {auc:.4f}")
    print(f"Bigram Baseline F1 (threshold=0.5): {f1:.4f}")
    print(f"Bigram Baseline F1 (best threshold={best_threshold:.2f}): {best_f1:.4f}")
    
    return auc, f1, best_f1

# Esegui
auc, f1, best_f1 = bigram_baseline_classifier_with_f1(X_train, y_train, X_val, y_val)
