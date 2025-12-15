import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score


# Reading data - already splitted!
# 1: lags [7,5,4,2]
# 2: lags [9,8,7,6]

csv_number = '1_2'
X_train = pd.read_csv(f"data/simulation/X_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_train = pd.read_csv(f"data/simulation/y_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_val = pd.read_csv(f"data/simulation/X_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_val = pd.read_csv(f"data/simulation/y_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_test = pd.read_csv(f"data/simulation/X_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_test = pd.read_csv(f"data/simulation/y_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

df = pd.DataFrame({
    "Sequences":pd.concat([X_train.Sequences, X_val.Sequences, X_test.Sequences]),
    "Outcome":pd.concat([y_train.Outcome, y_val.Outcome, y_test.Outcome])
})

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
    
    for k in range(1, 21):  # Da 1 a 9 lettere
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
    bigrams_tot = Counter()
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        bigrams = extract_bigrams(seq)
        if label == 0:
            bigrams_class_0.update(bigrams)
        else:
            bigrams_class_1.update(bigrams)
        bigrams_tot.update(bigrams)
    
    # Normalizza in probabilità
    total_0 = sum(bigrams_class_0.values())
    total_1 = sum(bigrams_class_1.values())
    total_tot = sum(bigrams_tot.values())
    
    prob_0 = {bg: count/total_0 for bg, count in bigrams_class_0.items()}
    prob_1 = {bg: count/total_1 for bg, count in bigrams_class_1.items()}
    prob_tot={bg: count/total_tot for bg, count in bigrams_tot.items()}
    
    # Trova bigrammi più discriminativi
    all_bigrams = set(prob_0.keys()) | set(prob_1.keys())
    bigram_scores = []

    for bg in all_bigrams:
        p0 = prob_0.get(bg, 0)
        p1 = prob_1.get(bg, 0)
        ptot=prob_tot.get(bg, 0)
        diff = abs(p1 - p0)
        bigram_scores.append((bg, ptot, p0, p1, diff))
    
    # Ordina per differenza
    bigram_scores.sort(key=lambda x: x[4], reverse=True)
    
    print("Top 20 bigrammi più discriminativi:")
    print("Bigram | P(bg) | P(bg|label=0) | P(bg|label=1) | Differenza")
    for bg, ptot, p0, p1, diff in bigram_scores[:20]:
        print(f"{bg:8} | {ptot:13.4f} | {p0:13.4f} | {p1:13.4f} | {diff:10.4f}")
    
    # Ordina per probabilità
    bigram_scores.sort(key=lambda x: x[1], reverse=True)
    print("Top 20 bigrammi più frequenti:")
    print("Bigram | P(bg) | P(bg|label=0) | P(bg|label=1) | Differenza")
    for bg, ptot, p0, p1, diff in bigram_scores[:20]:
        print(f"{bg:8} | {ptot:13.4f} | {p0:13.4f} | {p1:13.4f} | {diff:10.4f}")
    
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
bigram_scores = bigram_distribution_analysis(X_train, y_train, X_val, y_val)

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


def extract_ngrams(seq, n):
    """Extract all n-grams from a sequence"""
    letters = seq.split('\x1f')
    if len(letters) < n:
        return []
    return ["-".join(letters[i:i+n]) for i in range(len(letters)-n+1)]

def ngram_baseline_classifier_with_f1(X_train, y_train, X_val, y_val, n=2):
    """
    Classify based on n-gram distribution - with F1
    
    Parameters:
    -----------
    n : int
        Size of n-gram (2=bigrams, 3=trigrams, etc.)
    """
    # Calculate P(label=1 | n-gram)
    ngram_stats = defaultdict(lambda: [0, 0])
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        ngrams = extract_ngrams(seq, n)
        for ng in ngrams:
            ngram_stats[ng][int(label)] += 1
    
    ngram_probs = {}
    for ng, counts in ngram_stats.items():
        total = sum(counts)
        ngram_probs[ng] = counts[1] / total if total > 0 else 0.5
    
    # Predict
    val_preds_prob = []
    for seq in X_val['Sequences']:
        ngrams = extract_ngrams(seq, n)
        probs = [ngram_probs.get(ng, 0.5) for ng in ngrams]
        avg_prob = np.mean(probs) if probs else 0.5
        val_preds_prob.append(avg_prob)
    
    # Calculate metrics
    auc = roc_auc_score(y_val['Outcome'], val_preds_prob)
    
    # Binarize with threshold 0.5 for F1
    val_preds_binary = [1 if p >= 0.5 else 0 for p in val_preds_prob]
    f1 = f1_score(y_val['Outcome'], val_preds_binary)
    
    # Try with optimal threshold (that maximizes F1)
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in val_preds_prob]
        f1_scores.append(f1_score(y_val['Outcome'], preds))
    
    best_f1 = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    ngram_name = {2: 'Bigram', 3: 'Trigram', 4: '4-gram', 5: '5-gram', 6: '6-gram', 7: '7-gram'}
    name = ngram_name.get(n, f'{n}-gram')
    
    print(f"{name} Baseline AUC: {auc:.4f}")
    print(f"{name} Baseline F1 (threshold=0.5): {f1:.4f}")
    print(f"{name} Baseline F1 (best threshold={best_threshold:.2f}): {best_f1:.4f}")
    
    return auc, f1, best_f1, best_threshold

# Run for all n-gram sizes
print("N-gram Baseline Classifier Results")


results = {}
for n in range(2, 8):  # 2 to 7
    print(f"\n{n}-gram results:")
    print("-"*70)
    auc, f1, best_f1, best_thresh = ngram_baseline_classifier_with_f1(
        X_train, y_train, X_val, y_val, n=n
    )
    results[n] = {
        'auc': auc,
        'f1': f1,
        'best_f1': best_f1,
        'best_threshold': best_thresh
    }


# Find most frequent bigrams in both


def plot_ngram_vignette(X_train, y_train, max_n=7, top_n=700):
    """
    Create a vignette (faceted) plot showing n-gram distributions from 2-grams to max_n-grams.
    Each subplot shows the top_n most discriminative n-grams.
    """
    
    def extract_ngrams(seq, n):
        """Extract n-grams from a sequence"""
        letters = seq.split('\x1f')
        if len(letters) < n:
            return []
        return ["-".join(letters[i:i+n]) for i in range(len(letters)-n+1)]
    
    # Prepare subplots
    n_grams = list(range(2, max_n + 1))
    n_plots = len(n_grams)
    n_cols = 2  # 2 columns
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, n in enumerate(n_grams):
        ax = axes[idx]
        
        # Count n-grams per class
        ngrams_0 = Counter()
        ngrams_1 = Counter()
        
        for seq, label in zip(pd.DataFrame(X_train)['Sequences'], pd.DataFrame(y_train)['Outcome']):
            ngrams = extract_ngrams(seq, n)
            if label == 0:
                ngrams_0.update(ngrams)
            else:
                ngrams_1.update(ngrams)
        
        # Find most discriminative n-grams
        all_ngrams = set(ngrams_0.keys()) | set(ngrams_1.keys())
        ngram_diff = {}
        
        for ng in all_ngrams:
            freq_0 = ngrams_0.get(ng, 0)
            freq_1 = ngrams_1.get(ng, 0)
            diff = abs(freq_0 - freq_1)
            ngram_diff[ng] = (freq_0, freq_1, diff)
        
        # Sort and take top N
        top_ngrams = sorted(ngram_diff.items(), key=lambda x: x[1][2], reverse=True)[:top_n]
        
        # Prepare data
        ngram_labels = [ng for ng, _ in top_ngrams]
        counts_0 = [data[0] for _, data in top_ngrams]
        counts_1 = [data[1] for _, data in top_ngrams]
        
        # Plot
        x = np.arange(len(ngram_labels))
        width = 0.35
        
        ax.bar(x - width/2, counts_1, width, label='Label=1', alpha=0.8, color='C0')
        ax.bar(x + width/2, counts_0, width, label='Label=0', alpha=0.8, color='C1')
        
        ngram_name = {2: 'Bigrams', 3: 'Trigrams', 4: '4-grams', 5: '5-grams', 6: '6-grams', 7: '7-grams'}
        ax.set_title(f'Top {len(ngram_labels)} {ngram_name.get(n, f"{n}-grams")}', fontsize=12, fontweight='bold')
        
        ax.set_xlabel(f'{ngram_name.get(n, f"{n}-grams")}', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        
        # Don't show x-tick labels for top_n=100 (too crowded)
        ax.set_xticks([])
        
        if idx == 0:  # Legend only on first plot
            ax.legend(fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        
        # Print top 5 for each n-gram
        print(f"\nTop 5 most discriminative {ngram_name.get(n, f'{n}-grams')}:")
        print(f"{f'{n}-gram':<30} {'Count(0)':<12} {'Count(1)':<12} {'Diff':<10}")
        print("-" * 70)
        for ng, (c0, c1, diff) in top_ngrams[:5]:
            print(f"{ng:<30} {c0:<12} {c1:<12} {diff:<10.0f}")
    
    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'N-gram Distribution Analysis (Top {top_n} Most Discriminative)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('ngram_vignette.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
plot_ngram_vignette(X_train, y_train, max_n=7, top_n=700)
