import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score


# Reading data - already splitted!
csv_number = 9
X_train = pd.read_csv(f"data/simulation/X_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_train = pd.read_csv(f"data/simulation/y_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_val = pd.read_csv(f"data/simulation/X_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_val = pd.read_csv(f"data/simulation/y_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_test = pd.read_csv(f"data/simulation/X_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_test = pd.read_csv(f"data/simulation/y_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')


# Estrai prima lettera
train_first = [seq.split('\x1f')[0] for seq in X_train['Sequences']]
val_first = [seq.split('\x1f')[0] for seq in X_val['Sequences']]
test_first = [seq.split('\x1f')[0] for seq in X_test['Sequences']]


# Calcola P(label=1 | lettera) dal training
letter_stats = defaultdict(lambda: [0, 0])  # [count_label_0, count_label_1]
for letter, label in zip(train_first, y_train['Outcome']):
    letter_stats[letter][int(label)] += 1

# Predici sul test
test_preds = []
for letter in test_first:
    counts = letter_stats[letter]
    prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
    test_preds.append(prob)

baseline_auc = roc_auc_score(y_val['Outcome'], test_preds)
print(f"Baseline AUC (solo prima lettera): {baseline_auc:.4f}")

# Test on all letters
for pos in range(len(X_train["Sequences"][0].split("\x1f"))):
    pos_letters = [seq.split('\x1f')[pos] for seq in X_train['Sequences']]
    
    letter_stats = defaultdict(lambda: [0, 0])
    for letter, label in zip(pos_letters, y_train['Outcome']):
        letter_stats[letter][int(label)] += 1
    
    test_pos_letters = [seq.split('\x1f')[pos] for seq in X_test['Sequences']]
    test_preds = []
    for letter in test_pos_letters:
        counts = letter_stats[letter]
        prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
        test_preds.append(prob)
    
    pos_auc = roc_auc_score(y_test['Outcome'], test_preds)
    print(f"Position {pos+1} AUC: {pos_auc:.4f}")

# Test on all letters cumulating
def cumulative_position_analysis(X_train, y_train, X_val, y_val):
    """Test informazione cumulativa: prime k lettere"""
    
    results = []
    
    for k in range(1, 11):  # Da 1 a 9 lettere
        # Estrai prime k lettere da ogni sequenza
        train_prefixes = ['\x1f'.join(seq.split('\x1f')[:k]) for seq in X_train['Sequences']]
        val_prefixes = ['\x1f'.join(seq.split('\x1f')[:k]) for seq in X_val['Sequences']]
        
        # Calcola P(label=1 | prefix) dal training
        prefix_stats = defaultdict(lambda: [0, 0])
        for prefix, label in zip(train_prefixes, y_train['Outcome']):
            prefix_stats[prefix][int(label)] += 1
        
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
              f"Coverage test: {coverage:.1f}%")
        
        results.append({
            'k': k,
            'auc': auc,
            'unique_train': unique_train,
            'coverage': coverage
        })
    
    return results

results = cumulative_position_analysis(X_train, y_train, X_test, y_test)

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
            bigram_stats[bg][int(label)] += 1
    
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

# Let's visualize bigrams
def plot_bigram_distribution(X_train, y_train, top_n=30):
    """
    Crea un grafico che mostra i bigrammi più discriminativi
    tra le due classi
    """
    
    # Funzione per estrarre bigrammi
    def extract_bigrams(seq):
        letters = seq.split('\x1f')
        return [f"{letters[i]}-{letters[i+1]}" for i in range(len(letters)-1)]
    
    # Conta bigrammi per classe
    bigrams_0 = Counter()
    bigrams_1 = Counter()
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        bigrams = extract_bigrams(seq)
        if label == 0:
            bigrams_0.update(bigrams)
        else:
            bigrams_1.update(bigrams)
    
    # Calcola frequenze normalizzate
    total_0 = sum(bigrams_0.values())
    total_1 = sum(bigrams_1.values())
    
    # Trova bigrammi più discriminativi
    all_bigrams = set(bigrams_0.keys()) | set(bigrams_1.keys())
    bigram_diff = {}
    
    for bg in all_bigrams:
        freq_0 = bigrams_0.get(bg, 0)
        freq_1 = bigrams_1.get(bg, 0)
        diff = abs(freq_0 - freq_1)
        bigram_diff[bg] = (freq_0, freq_1, diff)
    
    # Ordina per differenza e prendi top N
    top_bigrams = sorted(bigram_diff.items(), key=lambda x: x[1][2], reverse=True)[:top_n]
    
    # Prepara dati per il grafico
    bigram_labels = [bg for bg, _ in top_bigrams]
    counts_0 = [data[0] for _, data in top_bigrams]
    counts_1 = [data[1] for _, data in top_bigrams]
    
    # Crea il grafico
    x = np.arange(len(bigram_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width/2, counts_1, width, label='Label=1 (Ordered)', alpha=0.8)
    bars0 = ax.bar(x + width/2, counts_0, width, label='Label=0 (Unordered)', alpha=0.8)
    
    ax.set_xlabel('Bigrams', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Top {top_n} Most Discriminative Bigrams by Label', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bigram_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bigram_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Stampa statistiche
    print(f"\nTop {min(10, top_n)} most discriminative bigrams:")
    print(f"{'Bigram':<10} {'Count(Label=0)':<15} {'Count(Label=1)':<15} {'Difference':<12}")
    print("-" * 60)
    for bg, (c0, c1, diff) in top_bigrams[:10]:
        print(f"{bg:<10} {c0:<15} {c1:<15} {diff:<12.0f}")

# Esegui
plot_bigram_distribution(X_train, y_train, top_n=30)
