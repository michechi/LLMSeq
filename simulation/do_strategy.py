from simulation.do_check_lag import check_lag

def do_chek_order(are_lagged_keys, key_letters, cycle):
    n_seq=len(are_lagged_keys)
    all_ordered = [True]*n_seq
    tol=cycle # For the cyclic ordering
    for pos, subseq in enumerate(are_lagged_keys):
        # Check whether are ordered
        for x,y in zip(subseq[:-1], subseq[1:]):
            if ((2*key_letters[x[0]])>(2*key_letters[y[0]])):
                    if tol:
                        tol=False
                    else:
                        all_ordered[pos]=False
    
    if any(all_ordered):
        # If there is at least one true, then there is one ordered sequence=> high probabilities of 1
        return all_ordered, True
    else:
        # If there is no true, then there are no one ordered sequence=> low probabilities of 1
        return all_ordered, False

def do_strategy(subseq, lags1_2, key_letters, debugging, cycle=False):
    # Check if there are key letters separated by lags.
    all_ordered_1 = False
    all_ordered_2 = False

    lag_1, lag_2 = lags1_2
    are_lagged_keys_1 = check_lag(subseq, key_letters, lag_1)
    are_lagged_keys_2 = check_lag(subseq, key_letters, lag_2)
    
    
    # Check the order depending on the lagged keys
    if are_lagged_keys_1 and are_lagged_keys_2:
        all_ordered_1, first_order = do_chek_order(are_lagged_keys_1, key_letters, cycle)
        all_ordered_2, second_order = do_chek_order(are_lagged_keys_2, key_letters, cycle)
    elif are_lagged_keys_1:
        all_ordered_1, first_order = do_chek_order(are_lagged_keys_1, key_letters, cycle)
        second_order = False
    elif are_lagged_keys_2:
        first_order = False
        all_ordered_2, second_order = do_chek_order(are_lagged_keys_2, key_letters, cycle)
    else:
        all_ordered_1, all_ordered_2 = False, False
        first_order = False
        second_order = False

    if debugging:
        info_2_debug = dict(
            lagged_1 = are_lagged_keys_1,
            lagged_2 = are_lagged_keys_2,
            order = [all_ordered_1, all_ordered_2]
            )
    else:
        info_2_debug = None

    # Let us decide the strategy
    if first_order and second_order:
        return info_2_debug, "both_orders"
    elif first_order:
        return info_2_debug, "first_order"
    elif second_order:
        return info_2_debug, "second_order"
    else:
        return info_2_debug, "no_order"

def do_order(subseq,lag, key_letters, strategy, debugging, cycle=False):
    """on the second subsequence, check the order according to lag and key_letters and strategy"""
    
    if strategy == "no_order":
        # No need to check second subsequence
        are_lagged_keys = False

    elif strategy == "both_orders":
        # we could check the second subsequence with both subset of keys
        # we could think as well to do an randomic choice between the two keys
        # but for now let's just check both keys
        key_to_test = key_letters[0]
        are_lagged_keys = check_lag(subseq, key_to_test, lag)
        if not are_lagged_keys:
            key_to_test = key_letters[1]
            are_lagged_keys = check_lag(subseq, key_to_test, lag)
        key_letters = key_to_test
        
    elif (strategy == "first_order") | (strategy == "second_order"):
        # we could check the second subsequence with the second subset of keys
        are_lagged_keys = check_lag(subseq, key_letters, lag)
    else:
        print("Strategy not recognized!\n")
        return

    if debugging:
        info_2_debug = dict(
            lagged_3 = are_lagged_keys,
            final_order =  [False]
        )
    else: 
        info_2_debug = None

    if are_lagged_keys:
        all_ordered_3, is_ordered = do_chek_order(are_lagged_keys, key_letters, cycle)

        if debugging:
            info_2_debug['final_order'] = all_ordered_3

        return info_2_debug, is_ordered
    
    return info_2_debug, False
    


    
