import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import random
import collections


from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from collections import Counter
from scipy.stats import bernoulli
from itertools import product
from typing import List, Union
from tqdm import tqdm

def rm_all():
    [globals().pop(var) for var in list(globals()) if not var.startswith('_')]

def generate_sequences(
    letters:list, n:int=10, m:int=10_000, replacement:bool=False, seed:int=None, batch_size:int=None, duplicates:bool=False
):
    """
    Generate exactly m unique sequences (rows) of length n where each event is 'Letter' + 'Digit' (A1, Z0, H4, ...).
    - If replacement=False, each *row* has no repeated letters and no repeated digits.
    - Across rows, duplicates are removed; generation continues until m unique rows are collected.
    """
    rng = np.random.default_rng(seed)
    L = np.asarray(letters, dtype='<U8')
    Llen = len(L)

    if not replacement and n > Llen:
        raise ValueError("n must be <= len(letters) when replacement=False")

    # Choose a sensible batch size (oversample a bit to reduce iterations)
    if batch_size is None:
        batch_size = min(10_000, m // 10)

    # Store unique rows via a hash set of compact string keys
    # Use a separator that won't appear in tokens (ASCII Unit Separator)
    SEP = "\x1f"
    seen = set()
    out_rows = []

    def _gen_batch(k):
        """k batch dimension"""
        if replacement:
            print("replacement!\n")
            li = rng.integers(0, Llen, size=(k, n))
        else:
            print("no replacement!\n")
            # Per-row permutations (no repeats within a row)
            l_scores = rng.random((k, Llen))
            li = np.argsort(l_scores, axis=1)[:, :n]
        seq = L[li]  # shape (k, n), dtype '<U...'
        return seq

    while len(out_rows) < m:
        need = m - len(out_rows)
        k = max(batch_size, need)  # at least batch_size to amortize costs
        seq = _gen_batch(k)        # (k, n)

        # Build vectorized keys for dedup: "A|B|..."
        keys = [SEP.join(key) for key in seq.tolist()]
        # Inefficient! (?)
        # for j in range(1, n):
        #     keys = np.char.add(np.char.add(keys, SEP), seq[:, j])

        # Filter rows not seen before
        mask_new = np.fromiter((key not in seen for key in keys), count=k, dtype=bool)
        if not mask_new.any():
            continue

        # new_keys = keys[mask_new] # Now keys it's a list not an array so this won't work
        new_keys = [key for key,to_keep in zip(keys, mask_new) if to_keep]

        out_rows+=new_keys
        for key in new_keys:
            seen.add(key)

    return out_rows

def generate_sequences_intervals():
    pass

# For parallelization
def worker_generate(args):
    letters, n, m_chunk, replacement, seed = args
    # Import inside worker if needed (like clusterEvalQ)
    # import pandas as pd  
    return generate_sequences(letters, n, m_chunk, replacement, seed)

def assign_outcome(seq:str, c_ord:dict, rnd:bool=False, sep:str="\x1f", already_splitted=False, pr_1=0.7) -> int:
    """
    function that, given a sequence, says 1 or 0, depending on ordering.
    """
    # test_seq = 'D7\x1fH5\x1fA7\x1fR5\x1fL1\x1fE4\x1fF8\x1fC0\x1fA8\x1fN0' # sequences[0]
    # test_seq_splt = test_seq.split("\x1f")
    np.random.seed(seed=123456)
    l_keys = c_ord.keys()

    if not already_splitted:
        test_seq_splt = [x for x in seq.split(sep) if x in l_keys] # avoiding noising letters
    else: 
        test_seq_splt = [x for x in seq if x in l_keys]
    
    tolerance=False # For the cyclic ordering
    if test_seq_splt:
        for x,y in zip(test_seq_splt[:-1], test_seq_splt[1:]):
            if ((2*c_ord[x[0]])>(2*c_ord[y[0]])):
                    if tolerance:
                        tolerance=False
                    else:
                        return bernoulli.rvs(1-pr_1) # 1 with pr 1-0.7 => 0 with 0.7 pr
        # if we exit for cycle then it's all ordered
        return bernoulli.rvs(pr_1)
    else:
        return bernoulli.rvs(1-pr_1) # if test_seq_splt is void then there are no keys so not ordered


def assign_outcome_numeric(seq:str, c_ord:dict, rnd:bool=False, sep:str="\x1f", already_splitted=False) -> int:
    """
    function that, given a sequence, says 1 or 0, depending on ordering.
    """
    # test_seq = 'D7\x1fH5\x1fA7\x1fR5\x1fL1\x1fE4\x1fF8\x1fC0\x1fA8\x1fN0' # sequences[0]
    # test_seq_splt = test_seq.split("\x1f")
    l_keys = c_ord.keys()

    if not already_splitted:
        test_seq_splt = [x for x in seq.split(sep) if x in l_keys] # avoiding noising letters
    else: 
        test_seq_splt = [x for x in seq if x in l_keys]
    
    # tolerance=False # For the cyclic ordering
    if sum(list(map(c_ord.get, test_seq_splt))) <= 9:
        return 1
    else:
        return 0

# def check_lag(seq:list, keys:list, lags:int=6):
#     """given a sequence, a list of keys check that the keys are happening just after the lag"""
    
#     positions_key = [(i, seq[i]) for i in range(len(seq)) if seq[i] in keys]
#     positions = [position_key[0] for position_key in positions_key]
#     keys = [position_key[1] for position_key in positions_key]
    
#     if keys:
#         obs_lags = [y-x for x,y in zip(positions[:-1], positions[1:])]
#         if list(set(obs_lags))==[lags]:
#             return keys
#         else:
#             return False
#     else:
#         return False

def check_lag(seq: List, keys: List, lags: int = 6) -> Union[List[List], bool]:
    """
    Trova tutte le catene di elementi dalle chiavi che sono distanziati esattamente di 'lags'.
    
    Args:
        seq: sequenza di elementi
        keys: elementi da cercare nella sequenza
        lags: distanza richiesta tra elementi consecutivi
    
    Returns:
        False se non trova pattern validi
        Lista di liste con i pattern trovati (es: [[A,B,C], [A,B,C,A]])
    """
    # Trova tutte le posizioni degli elementi chiave
    key_positions = [(i, elem) for i, elem in enumerate(seq) if elem in keys]
    
    if not key_positions:
        return False
    
    chains = []
    used_positions = set()
    
    # Per ogni posizione di partenza possibile
    for start_pos, start_elem in key_positions:
        if start_pos in used_positions:
            continue
            
        # Costruisci la catena più lunga possibile
        chain = [start_elem]
        chain_positions = [start_pos]
        current_pos = start_pos
        
        # Continua a cercare elementi a distanza 'lags'
        while True:
            next_pos = current_pos + lags
            
            # Cerca un elemento chiave alla posizione attesa
            found = False
            for pos, elem in key_positions:
                if pos == next_pos and elem in keys:
                    chain.append(elem)
                    chain_positions.append(pos)
                    current_pos = next_pos
                    found = True
                    break
            
            if not found:
                break
        
        # Salva la catena se ha almeno 2 elementi (pattern valido)
        if len(chain) >= 2:
            # Verifica che non sia sottoinsieme di una catena esistente
            is_subset = False
            for i, existing_chain in enumerate(chains):
                existing_positions = [p for p, _ in existing_chain]
                if set(chain_positions).issubset(set(existing_positions)):
                    is_subset = True
                    break
                # Se questa catena contiene una esistente, sostituiscila
                elif set(existing_positions).issubset(set(chain_positions)):
                    chains[i] = list(zip(chain_positions, chain))
                    used_positions.update(chain_positions)
                    is_subset = True
                    break
            
            if not is_subset:
                chains.append(list(zip(chain_positions, chain)))
                used_positions.update(chain_positions)
    
    if not chains:
        return False
    
    # Ritorna solo le liste di elementi (senza le posizioni)
    return [[elem for _, elem in chain] for chain in chains]

# TESTING
print("="*70)
print("ESEMPIO 1: Sequenza con una sola combinazione valida")
print("="*70)
seq1 = ['A', 'X', 'X', 'B', 'Y', 'Y', 'C']
keys = ['A', 'B', 'C']
lag = 3
check_lag(seq1, keys, lag)
check_lag(["Z","Z","Z","Z"]+seq1, keys,lag)


print("\n" + "="*70)
print("ESEMPIO 2: Sequenza con MULTIPLE combinazioni valide")
print("="*70)
# Sequenza progettata per avere multiple combinazioni valide
seq2 = ['A', 'X', 'X', 'B', 'Y', 'Y', 'C', 'Z', 'Z', 'A', 'W', 'W', 'B', 'V', 'V', 'C']
check_lag(seq2, keys, lag)

print("\n" + "="*70)
print("ESEMPIO 3: Combinazioni sovrapposte (stessi elementi usati più volte)")
print("="*70)
# A appare 3 volte, B 2 volte, C 2 volte - possibili combinazioni sovrapposte
seq3 = ['A', 'X', 'A', 'B', 'Y', 'C', 'A', 'Z', 'W', 'B', 'Q', 'R', 'C']
check_lag(seq3, keys, lag)

print("\n" + "="*70)
print("ESEMPIO 4: Il tuo esempio originale")
print("="*70)
seq4 = ['A', 'A', 'C', 'D', 'B', 'A', 'Z', 'H', 'C']
check_lag(seq4, keys, lag)


def assign_outcome_positional(seq:str, c_ord:dict, rnd:bool=False, sep:str="\x1f", already_splitted=False, pr_1 = 0.7, lags=8) -> int:
    """
    function that, given a sequence, says 1 or 0, depending on ordering.
    """
    # test_seq = 'D7\x1fH5\x1fA7\x1fR5\x1fL1\x1fE4\x1fF8\x1fC0\x1fA8\x1fN0' # sequences[0]
    # test_seq_splt = test_seq.split("\x1f")
    # np.random.seed(seed=123456)

    l_keys = c_ord.keys()

    if not already_splitted:
        test_seq_splt = seq.split(sep) # avoiding noising letters
    else: 
        test_seq_splt = seq
    
    # Check if there are key letters separated by lags.
    are_lagged_keys = check_lag(test_seq_splt, l_keys, lags)

    if are_lagged_keys:
        n_seq=len(are_lagged_keys)
        all_ordered = [True]*n_seq
        for pos, seq in enumerate(are_lagged_keys):
            # Check whether are ordered
            tolerance=True # For the cyclic ordering
            for x,y in zip(are_lagged_keys[:-1], are_lagged_keys[1:]):
                if ((2*c_ord[x[0]])>(2*c_ord[y[0]])):
                        if tolerance:
                            tolerance=False
                        else:
                            all_ordered[pos]=False
        if any(all_ordered):
            # If there is at least one true, then there is one ordered sequence=> high probabilities of 1
            pr_to_simulate = pr_1
        else:
            # If there is no true, then there are no one ordered sequence=> low probabilities of 1
            pr_to_simulate = 1-pr_1
    else:
        # If there no keys, then there are no one ordered sequence=> low probabilities of 1
        pr_to_simulate = 1-pr_1 # if test_seq_splt is void then there are no keys so not ordered
    simulate = bernoulli.rvs(pr_to_simulate)
    return(simulate)
    
seq4 = ['A', 'A', 'C', 'D', 'B', 'A', 'Z', 'H', 'C']
# seqT = ['M','V','D','V','M','T','L','C','C','G','X','J','C','Y','J','B','C','Q','F','M']
c_vocab = {w:p for p,w in enumerate(keys,start=0)}
# check_lag(seqT, ["W", "D", "Q", "J", "U"], 7)
assign_outcome_positional(seqT, c_vocab, lags=3, already_splitted=True)

random.seed(959693)

n_events = 20
n_seq = 10_000_000
k =4
n_0s = 500_000
n_1s = 500_000
n_tot = n_0s + n_1s
n_train = n_tot * 0.80 # 80% of n_tot
n_val = n_tot * 0.10
n_test = n_tot * 0.10

#sum([n_train, n_test, n_val]) == n_tot

letters = list(string.ascii_uppercase)
# letters_4_key = letters.copy()
# random.shuffle(letters_4_key)
letters_4_key = ["W", "D", "Q", "J", "U"]
# digits = list(map(str, range(10)))

# random.shuffle(letters_4_key)
# random.shuffle(digits)

c_vocab = {w:p for p,w in enumerate(letters_4_key,start=0)}

# Test mio numerico
# c_vocab = {
# 'C': 2, 
# 'Q': 1,
# 'O': 0,
# 'H': 2,
# 'Y': 1,
# 'X': 0,
# 'S': 2,
# 'U': 1,
# 'N': 0,
# 'J': 2,
# 'F': 1,
# 'E': 0,
# 'W': 2,
# 'Z': 1,
# 'B': 0,
# 'G': 2,
# 'L': 1,
# 'A': 0,
# 'K': 2,
# 'I': 1,
# 'V': 0,
# 'P': 2,
# 'T': 1,
# 'D': 0,
# 'M': 2,
# 'R': 1}


# This is sequential
sequences = generate_sequences(letters=letters, n=n_events, m=n_seq, replacement=True)

# # Parallel version
# n_cores = cpu_count()-1
# rng = np.random.default_rng(999)
# seeds = rng.integers(0, 2**31, size=n_cores)
# m_per_core = int(n_seq * 1.2 / n_cores)  # 20% oversample
# tasks = [(letters, n_events, m_per_core, True, seed) 
#              for seed in seeds]

# with Pool(processes=n_cores) as pool:
#     results = pool.map(worker_generate, tasks)

# # Deduplicate and trim
# all_sequences = [seq for result in results for seq in result]
# sequences = list(dict.fromkeys(all_sequences))[:n_seq]

set_seq = set(sequences)
n_set_seq = len(set_seq)

if n_seq != n_set_seq:
    print(f"Attention! There are {n_seq-n_set_seq} duplicates!\nRemoving them..")
    sequences = set_seq.copy()
    n_seq = n_set_seq
    del set_seq, n_set_seq
    print("Done!")

def efficient_check_v1(sequences, c_vocab, assign_outcome_positional):
    """Calcola una volta sola e poi usa i risultati"""
    # Calcola UNA SOLA VOLTA per ogni sequenza
    outcomes = []
    

    for seq in tqdm(sequences):
    
        outcomes += [assign_outcome_positional(seq, c_vocab, already_splitted=False, lags=7)]
    # outcomes = [assign_outcome_positional(seq, c_vocab, already_splitted=False, lags=8) for seq in sequences]
    
    # Ora usa zip per associare sequenze ai loro outcomes
    sequences_with_outcomes = list(zip(sequences, outcomes))
    
    # Filtra basandoti sui risultati già calcolati
    which_1s = [seq for seq, outcome in sequences_with_outcomes if outcome == 1]
    which_0s = [seq for seq, outcome in sequences_with_outcomes if outcome == 0]
    
    n_1s = len(which_1s)
    n_0s = len(which_0s)
    
    if n_1s > 0:
        print(f"There are {n_1s} ordered sequences! ({n_1s/len(sequences)*100:.2f}%)")
        
        if (n_0s + n_1s) == len(sequences):
            print("All good!")
        else:
            print("Figures do not add up!")
    else:
        print("No valid sequences found")
    
    return {
        'valid_sequences': which_1s,
        'invalid_sequences': which_0s,
        'outcomes': outcomes
    }


# Before moving to the cyclic-ordering
# l_all_valid_seq = list(itertools.combinations(letters, n_events)) # All possible valid letters combinations
# d_all_valid_seq = list(itertools.combinations(digits, n_events)) # Not really important in this case since for now we don't use
#                                                                  # repeated letters (-> numbers are not important then)

# all_valid_seq = [np.char.add(x,y).tolist() for x in l_all_valid_seq for y in  d_all_valid_seq]
# random.shuffle(all_valid_seq)

# # Double Check if they are all valid
# check_valid=list(map(lambda x: assign_outcome(x, already_splitted=True), all_valid_seq))
# sum(check_valid) == len(check_valid) # True, all 1!

# all_valid_seq_str = ["\x1f".join(x) for x in all_valid_seq]

check = efficient_check_v1(sequences, c_vocab, assign_outcome_positional)

def extract_characters(seq:str, sep:str="\x1f") -> str:
    seq_char = "-".join([x[0] for x in seq.split(sep)])
    return(seq_char)

# check.keys()
chr_seq_1s=list(map(extract_characters, check['valid_sequences']))
chr_seq_0s=list(map(extract_characters, check['invalid_sequences']))

i = 17
stats_1s, stats_0s = collections.Counter(x[i*2] for x in chr_seq_1s), collections.Counter(x[i*2] for x in chr_seq_0s)

letters_ord = string.ascii_uppercase
counts_1s = [stats_1s.get(letter, 0) for letter in letters_ord]
counts_0s = [stats_0s.get(letter, 0) for letter in letters_ord]

# Plot affiancato
x = np.arange(len(letters_ord))
width = 0.35  # Larghezza barre

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, counts_1s, width, label='Label=1', alpha=0.8)
ax.bar(x + width/2, counts_0s, width, label='Label=0', alpha=0.8)
ax.set_xlabel('Letters')
ax.set_ylabel('Count')
ax.set_title(f'{i+1}st/nd Letter Frequency by Label')
ax.set_xticks(x)
ax.set_xticklabels(letters_ord)
ax.legend()
plt.show()


# Create Dataset
df = pd.DataFrame({"Sequences":sequences})
df["Outcome"] = list(map(lambda x: assign_outcome_positional(x, c_vocab, lags=8), sequences))

df_1s = df.loc[df["Outcome"]==1, ]
df_0s = df.loc[df["Outcome"]==0, ]
df_full = pd.concat([df_1s, df_0s], ignore_index=True)

# Let's try to reduce to 500_000 sample
df_full = df_full.sample(n_1s)

# Diagnostics graphs
which_1s_sel = list(df_full.loc[df_full["Outcome"]==1, "Sequences"])
which_0s_sel = list(df_full.loc[df_full["Outcome"]==0, "Sequences"])

chr_seq_1s=list(map(extract_characters, which_1s_sel))
chr_seq_0s=list(map(extract_characters, which_0s_sel))

i = 0
stats_1s, stats_0s = collections.Counter(x[i*2] for x in chr_seq_1s), collections.Counter(x[i*2] for x in chr_seq_0s)

letters_ord = string.ascii_uppercase
counts_1s = [stats_1s.get(letter, 0) for letter in letters_ord]
counts_0s = [stats_0s.get(letter, 0) for letter in letters_ord]

# Plot affiancato
x = np.arange(len(letters_ord))
width = 0.35  # Larghezza barre

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, counts_1s, width, label='Label=1', alpha=0.8)
ax.bar(x + width/2, counts_0s, width, label='Label=0', alpha=0.8)
ax.set_xlabel('Letters')
ax.set_ylabel('Count')
ax.set_title(f'{i+1}st/nd Letter Frequency by Label')
ax.set_xticks(x)
ax.set_xticklabels(letters_ord)
ax.legend()
plt.show()

# Splitting Train Val Test
# Now train_val_test split:
X, y = df_full["Sequences"], df_full["Outcome"]
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=0.80, random_state=999)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.50, random_state=999)

# Uncomment just if you want to save data!
for df, name in zip([X_train, X_val, X_test, y_train, y_val, y_test], ["X_train_5", "X_val_5", "X_test_5", "y_train_5", "y_val_5", "y_test_5"]):
    df.to_csv(f"data/simulation/{name}.csv", index=False)


# Let's visualize bigrams
def plot_bigram_distribution(X_train, y_train, from_n=0, top_n=30):
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
    
    for seq, label in zip(pd.DataFrame(X_train)['Sequences'], pd.DataFrame(y_train)['Outcome']):
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
    top_bigrams = sorted(bigram_diff.items(), key=lambda x: x[1][2], reverse=True)[from_n:top_n]
    
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
plot_bigram_distribution(X_train, y_train, from_n=400, top_n=500)
