# tokens(observations) by num_states
import numpy as np
from numpy import random

def neseq(num_states, num_tokens, pi, A, O, len_seq):
    seq = []
    
    rand_init_state = np.random.choice(range(num_states))
    state = rand_init_state

    # states by observations now
    Ot = np.transpose(O)
     
    # next state essentially chosen randomly, need to find a way 
    # to take probability into account
    while len(seq) < len_seq:
        rand_obs = np.random.choice(range(num_tokens))
        seq.append(observation)
        next_state = np.random.choice(range(num_states))
        state = next_state
        
    return seq
        
        
        
        
        
        