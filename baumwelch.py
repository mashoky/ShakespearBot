import numpy as np

def baum_welch(num_states, sequences, num_tokens):
    A = np.random.uniform(-0.5,0.5,(num_states, num_states))
    O = np.random.uniform(-0.5,0.5,(num_tokens, num_states))
    pi = np.random.uniform(0, num_tokens, (1, num_states))
    
    idx = np.random.randint(0, len(sequences))
    seq = sequences[idx]
    
    # Forward procedure
    alphas = []
    prev_alpha = []
    for j,w in enumerate(seq):
        curr = []
        for i in range(num_states):
            if j == 0:
                val = pi[1][i]
            else:
                val = 0
                for k in range(num_states):
                    val += prev_alpha[k] * A[k][i]
            curr.append(val * O[i][w])
        alphas.append(curr)
        prev_alpha = curr
    
    # Backward Procedure
    betas = []
    nxt_beta = []
    for j,w in reversed(enumerate(seq)):
        curr = []
        for i in range(num_states):
            if j == len(seq) - 1:
                curr.append(1)
            else:
                val = 0
                for k in range(num_states):
                    val += nxt_beta[k] * A[i][k] * O[k][w]
                curr.append(val)
        betas.insert(0, curr)
        nxt_beta = curr
  
    gammas = []  
    # Calculate temporary variables
    for t, word in enumerate(seq):
        curr = []
        for i in range(num_states): 
            num = alphas[t][i] * betas[t][i]       
            sum_val = sum(alphas[t][j] * betas[t][j] for j in range(num_states))
            curr.append(num / float(sum_val))
        gammas.append(curr)
        
    e_vec = []
    for t, word in enumerate(seq[:-1]):
        e_mat = np.zeros((num_states,num_states))
        for i in range(num_states): 
            for j in range(num_states):
                num = alphas[t][i] * A[i][j] * betas[t + 1][j]  * O[seq[t + 1]][j]
                sum_val = sum(alphas[num_states - 1][k] for k in range(num_states))
                e_mat[i][j] = num / float(sum_val)
        e_vec.append(e_mat)  
        
    # Update parameters
    for i in range(num_states):
        pi[1][i] = gammas[1][i]
        
    # Update Transition Matrix
    for i in range(num_states):
        for j in range(num_states):
            num = sum(e_vec[t][i][j] for t in range(len(seq) - 1))
            denominator = sum(gammas[t][i] for t in range(len(seq) - 1))
            A[i][j] = num / float(denominator)
        