import numpy as np
import nltk

from hyphen import Hyphenator, dict_info

from hyphen.dictools import *
import collections
h_en = Hyphenator('en_US')

#file = open('C:/Users/Jagriti/Documents/CS155/project2data/shakesare.txt', 'r')

file = open('C:\Users\manasa\Documents\Caltech\CS 155\ShakespeareProject\smallshakespear.txt')

int_list = []
punc_list = ['.', ',', ';', ':','?','(',')']
#punc_list = []
punc_string = '.,;:?()'
#punc_string = ''
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

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
            i = i.lower()
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
                        
                        #sequence_char.append(j)
                        #if j not in word_num_dict.keys():
                        #    word_num_dict[j] = index
                        #    index += 1
                        if new_word not in word_num_dict.keys():
                            word_num_dict[new_word] = index
                            index += 1
            # otherwise, just add the word
            elif contains_punc == False:
                # If word has apostrophe, and the first part of the word is a 
                # valid english word, just add this part
                parts = i.split("'")
                processed = ''
                if parts[0] in english_vocab and len(parts[0]) > 1:
                    processed = parts[0]
                else:
                    processed = i   
                sequence_char.append(processed)
                if processed not in word_num_dict.keys():
                    word_num_dict[processed] = index
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


def baum_welch(num_states, sequences, num_tokens, pi):
    #A = np.ones((num_states, num_states)) / num_states
    #O = np.ones((num_tokens, num_states)) / num_states
    A = np.random.uniform(0.0,1.0,(num_states, num_states))
    O = np.random.uniform(0.0,1.0,(num_tokens, num_states)) 
    for i in range(num_states):
        A[i] = A [i]/ np.sum(A, axis = 1)[i]
    for j in range(num_states):
        O [:,j] = O[:,j] / np.sum(O, axis = 0)[j]
  
    num_iter = 1
    prev_a_norm = 100000
    prev_o_norm = 100000
    print 'out'
    for it in range(num_iter):
        print 'Iteration'
        print it
        #if idx == len(sequences):
        #    order = np.random.permutation(len(sequences))
        #    idx = 0
        temp_a = np.zeros((num_states, num_states))
        temp_o = np.zeros((num_tokens, num_states))
        #seq_num = 0
        for seq in sequences:
            #print seq_num
            #seq_num += 1
            # Forward procedure
            alphas = []
            prev_alpha = []
            # Calculate probability of seeing a particular emission and being
            # in state i at position j
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
                curr = [float(i)/sum(curr) for i in curr]
                alphas.append(curr)
                prev_alpha = curr
            
            # Backward Procedure
            # Calculate probability of ending the partial sequence at a particular
            # observation, given the starting state i at position j
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


                curr = [float(i)/sum(curr) for i in curr]
                betas.insert(0, curr)
                nxt_beta = curr
        
            gammas = []  
            # Calculate temporary variables
            # Probability of being in state i at position t given the observed
            # sequence
            for t, word in enumerate(seq):
                curr = []
                for i in range(num_states): 
                    num = alphas[t][i] * betas[t][i]       
                    sum_val = sum(alphas[t][j] * betas[t][j] for j in range(num_states))
                    curr.append(num / float(sum_val))
                #print 'gamma_sum'
                #print sum(curr)
                gammas.append(curr)
                
            e_vec = []
            # Calculate probability of being in state i and state j at positions
            # t and t + 1 given the observed sequence
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
                    if denominator != 0:
                        temp_a[i][j] += num / float(denominator)

            for i in range(num_states):
                #print 'NEW Round'
                #print i
                for v_k in range(num_tokens):
                    i_sum = 0
                    denominator = 0
                    for t in range(len(seq)):
                        if seq[t] == v_k:
                            #print "hi"
                            #print gammas[t][i]
                            i_sum += gammas[t][i]
                            #print i_sum
                        #print i_sum
                        denominator += gammas[t][i]
                        #print i_sum
                    val = 0
                    if denominator != 0:
                        val += i_sum / float(denominator)
                    temp_o[v_k][i] += val
                    #if val != 0:
                        #print val
                # print val   
        for i in range(num_states):
            for j in range(num_states):
                A[i][j] = (temp_a[i][j] / float(len(sequences)))
        for v_k in range(num_tokens):
            for i in range(num_states):
                O[v_k][i] = temp_o[v_k][i] / float(len(sequences))
        a_norm = np.linalg.norm(A)
        o_norm = np.linalg.norm(O)
        if abs(a_norm - prev_a_norm) < 0.1 and abs(o_norm - prev_o_norm) < 0.1:
            break
        prev_a_norm = a_norm
        prev_o_norm = o_norm
    #print A   
    #print O
    #print np.sum(A, axis=0)
    #print np.sum(O, axis=0)
    return (A, O)
    
def get_random(row):
    total = np.sum(row)
    rand_prob = np.random.uniform(0.0,total)

    cumulative_prob = 0
    for idx in range(len(row)):
        cumulative_prob += row[idx]
        if rand_prob <= cumulative_prob:
            return idx
            
def get_syllable_info(tokens):
    syllables = collections.defaultdict(list)
    for k in tokens.keys():
        s = len(h_en.syllables(unicode(k)))
        # Sometimes pyhyphen treats 1 syllable words as having 0 syllables
        if s == 0 and k not in punc_list:
            s = 1
        syllables[s].append(k)
    return syllables

def neseq(num_states, num_tokens, pi, A, O, tokens, total_len):
    res = dict((v,k) for k,v in tokens.iteritems())
    syllables = get_syllable_info(tokens)
    print syllables[0]
    seq = []
    rand_init_state = get_random(pi[0])
    state = rand_init_state
    # states by observations now
    Ot = np.transpose(O)
    num_syllables = 0
    # next state essentially chosen randomly, need to find a way 
    # to take probability into account
    while num_syllables < total_len:
        if num_syllables == total_len - 2:
            indices = [tokens[x] for x in syllables[2]]
            row = [Ot[state][i] for i in indices]
            rand_idx = get_random(row)
            rand_obs = indices[rand_idx]
            print '2'
            num_syllables += 2
        elif num_syllables == total_len - 1:
            indices = [tokens[x] for x in syllables[1]]
            row = [Ot[state][i] for i in indices]
            rand_idx = get_random(row)
            rand_obs = indices[rand_idx]
            print '1'
            num_syllables += 1
        else:            
            while True:
                rand_obs = get_random(Ot[state])
                size = len(h_en.syllables(unicode(res[rand_obs], "utf-8")))
                # Account for 1 syllable words that are counted as 0 syllable
                if size == 0 and res[rand_obs] not in punc_list:
                    size += 1
                print size
                if num_syllables + size < 10:
                    num_syllables += size
                    break
        #rand_obs = 0
        #seq.append(res[rand_obs])
        print res[rand_obs]
        seq.append(res[rand_obs])
        next_state = get_random(A[state])
        #next_state = 0
        state = next_state
    seq[0] = seq[0].title()
    return " ".join(seq)


pi = np.random.uniform(0, 1, (1, num_states))
pi[0] = pi[0] / np.sum(pi[0])
    
(A, O) = baum_welch(num_states, sequences,num_tokens, pi)
#print A
#print O
#print np.sum(A, axis=1)
#print np.sum(O, axis=0)
poem = []
for i in range(num_states):
    A[i] = A [i]/ np.sum(A, axis = 1)[i]
for i in range(14):
    seq = neseq(num_states, num_tokens,pi, A, O,word_num_dict, 10)
    poem.append(seq)
print "\n".join(poem)