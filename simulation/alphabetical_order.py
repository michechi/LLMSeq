import pandas as pd
import numpy as np

def is_alphabetically_ordered(seq, strict=False):
    """
    Verifica se una sequenza è ordinata alfabeticamente.

    Args:
        seq: stringa con lettere separate da '\x1f'
        strict: se True, richiede ordine stretto (A < B < C)
                se False, permette uguaglianze (A <= B <= C)

    Returns:
        bool: True se la sequenza è ordinata
    """
    letters = seq.split('\x1f')
    for i in range(len(letters) - 1):
        if strict:
            if letters[i] >= letters[i+1]:
                return False
        else:
            if letters[i] > letters[i+1]:
                return False
    return True


def analyze_alphabetical_order(csv_number='9'):
    """Analizza la percentuale di sequenze ordinate alfabeticamente."""

    print(f"=== Analisi ordine alfabetico per csv_number={csv_number} ===\n")

    # Carica i dati
    X_train = pd.read_csv(f"data/simulation/X_train_{csv_number}.csv",
                          na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_train = pd.read_csv(f"data/simulation/y_train_{csv_number}.csv",
                          na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    X_val = pd.read_csv(f"data/simulation/X_val_{csv_number}.csv",
                        na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_val = pd.read_csv(f"data/simulation/y_val_{csv_number}.csv",
                        na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    X_test = pd.read_csv(f"data/simulation/X_test_{csv_number}.csv",
                         na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_test = pd.read_csv(f"data/simulation/y_test_{csv_number}.csv",
                         na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    datasets = {
        'Train': (X_train, y_train),
        'Val': (X_val, y_val),
        'Test': (X_test, y_test)
    }

    for name, (X, y) in datasets.items():
        print(f"--- {name} (n={len(X)}) ---")

        # Ordine non stretto (<=)
        ordered_nonstrict = X['Sequences'].apply(lambda s: is_alphabetically_ordered(s, strict=False))
        # Ordine stretto (<)
        ordered_strict = X['Sequences'].apply(lambda s: is_alphabetically_ordered(s, strict=True))

        pct_nonstrict = ordered_nonstrict.mean() * 100
        pct_strict = ordered_strict.mean() * 100

        print(f"  Ordine non stretto (A <= B): {ordered_nonstrict.sum():,} / {len(X):,} = {pct_nonstrict:.4f}%")
        print(f"  Ordine stretto     (A < B):  {ordered_strict.sum():,} / {len(X):,} = {pct_strict:.4f}%")

        # Breakdown per classe (label)
        for label in [0, 1]:
            mask = y['Outcome'] == label
            n_label = mask.sum()

            ordered_ns_label = ordered_nonstrict[mask].sum()
            ordered_s_label = ordered_strict[mask].sum()

            pct_ns = (ordered_ns_label / n_label * 100) if n_label > 0 else 0
            pct_s = (ordered_s_label / n_label * 100) if n_label > 0 else 0

            print(f"    Label={label}: non-stretto {ordered_ns_label:,}/{n_label:,} = {pct_ns:.4f}% | "
                  f"stretto {ordered_s_label:,}/{n_label:,} = {pct_s:.4f}%")

        print()

    # Analisi aggiuntiva: lunghezza massima di sottosequenza ordinata
    print("--- Lunghezza media della sottosequenza ordinata più lunga ---")

    def longest_ordered_subsequence(seq):
        """Trova la lunghezza della sottosequenza ordinata più lunga."""
        letters = seq.split('\x1f')
        max_len = 1
        current_len = 1
        for i in range(1, len(letters)):
            if letters[i] >= letters[i-1]:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 1
        return max_len

    for name, (X, y) in datasets.items():
        lengths = X['Sequences'].apply(longest_ordered_subsequence)
        print(f"  {name}: media={lengths.mean():.2f}, mediana={lengths.median():.1f}, max={lengths.max()}")

        for label in [0, 1]:
            mask = y['Outcome'] == label
            lengths_label = lengths[mask]
            print(f"    Label={label}: media={lengths_label.mean():.2f}, mediana={lengths_label.median():.1f}")


if __name__ == "__main__":
    analyze_alphabetical_order('9')
