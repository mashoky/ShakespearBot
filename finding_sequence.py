# tokens(observations) by num_states
import numpy as np
from numpy import random

def get_random(row):
    rand_prob = np.random.uniform(0.0,1.0)
    print rand_prob
    cumulative_prob = 0
    for idx in range(len(row)):
        cumulative_prob += row[idx]
        if rand_prob <= cumulative_prob:
            return idx
    return row.index(max(row))

def neseq(num_states, num_tokens, pi, A, O, len_seq):
    seq = []
    
    rand_init_state = get_random(pi)
    state = rand_init_state

    # states by observations now
    Ot = np.transpose(O)
     
    # next state essentially chosen randomly, need to find a way 
    # to take probability into account
    while len(seq) < len_seq:
        rand_obs = get_random(Ot[state])
        seq.append(rand_obs)
        next_state = get_random(A[state])
        next_state = np.random.choice(range(num_states))
        state = next_state
        
    return seq
        
        
        
        
        
        