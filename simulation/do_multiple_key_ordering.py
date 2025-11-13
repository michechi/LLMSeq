import numpy as np
import random
import string

letters = list(string.ascii_uppercase)

def do_keys(letters:list, seed:int) -> dict:
    """Create a different ordering for each letter of the alphabet"""
    key_dict = dict()
    for i, letter in enumerate(letters):
        letters_2_shuffle = letters.copy()
        random.seed(seed+i)
        random.shuffle(letters_2_shuffle)
        key_dict[letter] = {w:p for p,w in enumerate(letters_2_shuffle, start=0)}

    return key_dict

def do_lags(letters:list, seed:int) -> dict:
    """
    Associate for each letter a different lag.
    - 7 -> 0.45
    - 3 -> 0.3
    - 9 -> 0.1
    - 4 -> 0.1
    - 2 -> 0.05
    """
    random.seed(seed)
    lag_dict = dict()
    v = [7, 3, 9, 4, 2] # lags
    n_lags = 5 # no hard_coded
    # How can I distribute probability mass better and in a more automatical way?
    p = np.array(
        [[0.45, 0.2, 0.2, 0.1, 0.05], # first lag
        [0.35, 0.3, 0.1, 0.2, 0.05], # second lag
        [0.1, 0.4, 0.1, 0.3, 0.1], # third lags
        [0.05, 0.3, 0.05, 0.4, 0.2],
        [0.05, 0.3, 0.05, 0.1, 0.5]]) # probabilities for each lag

    n = len(letters) # number of letters
    
    for i in range(n_lags):
        indexes_lags = np.random.multinomial(1, p[i], n)
        all = v * indexes_lags
        mask = all > 0 
        lags = all[mask]
        lag_dict[i] = dict(zip(letters, lags))

    return lag_dict

def do_multiple_key_ordering(seq:str, key_dict:dict, lag_dict:dict):
    """
    Function to do multiple key ordering strategy on a sequence.
    seq: input sequence
    ordering_dict: dictionary containing ordering information for different keys (one for each letter of the alphabet)
    """
    
    # for each sequence:
    # - 1: given the first letter, understand the ordering and the lag
    ordered=True
    current_position, lag =0, 0
    n_max = len(seq)-1
    while ordered:
        pivot_letter = seq[current_position]
        current_dict = key_dict[pivot_letter]
        current_lag  = lag_dict[lag][pivot_letter]
        next_position=current_position+current_lag
        if next_position < n_max:
            next_letter  = seq[next_position]
        else:
            return ordered
        
        ordered = (current_dict[pivot_letter] <= current_dict[next_letter])
        current_position = next_position
        lag +=1
    return ordered