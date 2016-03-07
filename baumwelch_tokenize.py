import numpy as np

file = open('C:/Users/Jagriti/Documents/CS155/project2data/shakespeare.txt', 'r')

int_list = []
punc_list = ['.', ',', ';', ':']
punc_string = '.,;:'

# list of sequences, where each word is a number corrresponding to balue in dictionary
sequence_list = []

for i in range(1, 155):
    num = str(i)
    int_list.append(num)


word_num_dict = {}
index = 0
for line in file:
    sequence_char = []
    sequence_num = []
    contains_num = False
    for i in int_list:
        if i in line:
            contains_num = True
    # if the line doesn't contain just the number of the sonnet and it isn't empty, 
    # split based on spaces
    if contains_num == False and line != "\n":
        words = line.strip()
        words =words.split(' ')
        
        for i in words:
            contains_punc = False
            
            if any((c in punc_string) for c in i):
                contains_punc = True
              
            # if the word contains some punctuation, add th epunctuation and word 
            # separately  
            if contains_punc == True:
                for j in punc_list:
                    if j in i:
                        punc = j
                        new_word = i.replace(j, "")
                        sequence_char.append(new_word)
                        
                        sequence_char.append(j)
                        if j not in word_num_dict.keys():
                            word_num_dict[j] = index
                            index += 1
                        if new_word not in word_num_dict.keys():
                            word_num_dict[new_word] = index
                            index += 1
            # otherwise, just add the word
            elif contains_punc == False:
                sequence_char.append(i)
                if i not in word_num_dict.keys():
                    word_num_dict[i] = index
                    index += 1
        for i in sequence_char:
            val = word_num_dict.get(i)
            sequence_num.append(val)
            
        sequence_list.append(sequence_num)
        

vals = word_num_dict.values()
vals.sort()
#print word_num_dict['.']

num_states = 10
sequences = sequence_list
num_tokens = len(word_num_dict.keys())


def baum_welch(num_states, sequences, num_tokens):
    A = np.random.uniform(0,1,(num_states, num_states))
    O = np.random.uniform(0,1,(num_tokens, num_states))
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
                val = pi[0][i]
            else:
                val = 0
                for k in range(num_states):
                    val += prev_alpha[k] * A[k][i]
            curr.append(val * O[w][i])
        alphas.append(curr)
        prev_alpha = curr
    
    # Backward Procedure
    betas = []
    nxt_beta = []
    for j,w in enumerate(reversed(seq)):
        curr = []
        for i in range(num_states):
            #if j == len(seq) - 1:
            if j == 0:
                curr.append(1)
            else:
                val = 0
                for k in range(num_states):
                    #val += nxt_beta[k] * A[i][k] * O[w][k]
                    val += nxt_beta[0] * A[i][k] * O[w][k]
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
                #sum_val = sum(alphas[num_states - 1][k] for k in range(num_states))
                sum_val = sum(alphas[len(seq) - 1][k] for k in range(num_states))
                e_mat[i][j] = num / float(sum_val)
        e_vec.append(e_mat)  
        
    # Update parameters
    for i in range(num_states):
        pi[0][i] = gammas[1][i]
        
    # Update Transition Matrix
    for i in range(num_states):
        for j in range(num_states):
            num = sum(e_vec[t][i][j] for t in range(len(seq) - 1))
            denominator = sum(gammas[t][i] for t in range(len(seq) - 1))
            A[i][j] = num / float(denominator)
            
    for i in range(num_states):
        for v_k in range(num_tokens):
            i_sum = 0
            denominator = 0
            for t in range(len(seq)):
                if seq[t] == v_k:
                    #print "hi"
                    #print gammas[t][i]
                    i_sum += gammas[t][i]
                    #print i_sum
                denominator += gammas[t][i]
                #print i_sum
            val = i_sum / float(denominator)
            O[v_k][i] = val
           # print val
        
    #print O
            
baum_welch(num_states, sequences, num_tokens)