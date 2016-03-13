import numpy as np
import nltk

from hyphen import Hyphenator, dict_info

from hyphen.dictools import *
import collections
h_en = Hyphenator('en_US')

#file = open('C:/Users/Jagriti/Documents/CS155/project2data/shakesare.txt', 'r')

file = open(r'C:\Users\manasa\Documents\Caltech\CS 155\ShakespeareProject\shakespeare.txt')

int_list = []
punc_list = ['.', ',', ';', ':','?','(',')']
#punc_list = []
punc_string = '.,;:?()'
#punc_string = ''
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

# list of sequences, where each word is a number corrresponding to value in dictionary
sequence_list = []

for i in range(1, 155):
    num = str(i)
    int_list.append(num)


word_num_dict = {}
counts = {}
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
                        if new_word not in counts.keys():
                            counts[new_word] = 1
                        else:
                            counts[new_word] += 1
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
                if processed not in counts.keys():
                    counts[processed] = 1
                else:
                    counts[processed] += 1
        for i in sequence_char:
            val = word_num_dict.get(i)
            sequence_num.append(val)
            
        sequence_list.append(sequence_num)
        

vals = word_num_dict.values()
vals.sort()
#print word_num_dict['.']

num_states = 5
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
  
    num_iter = 100
    prev_a_norm = 100000
    prev_o_norm = 100000
    print 'out'
    for it in range(num_iter):
        print 'Iteration'
        print it
        temp_a = np.zeros((num_states, num_states))
        temp_o = np.zeros((num_tokens, num_states))
        for seq in sequences:
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
                            i_sum += gammas[t][i]
                        denominator += gammas[t][i]
                    val = 0
                    if denominator != 0:
                        val += i_sum / float(denominator)
                    temp_o[v_k][i] += val  
        for i in range(num_states):
            for j in range(num_states):
                A[i][j] = (temp_a[i][j] / float(len(sequences)))
        for v_k in range(num_tokens):
            for i in range(num_states):
                O[v_k][i] = temp_o[v_k][i] / float(len(sequences))
        a_norm = np.linalg.norm(A)
        o_norm = np.linalg.norm(O)
        if abs(a_norm - prev_a_norm) < 0.001 and abs(o_norm - prev_o_norm) < 0.001:
            break
        prev_a_norm = a_norm
        prev_o_norm = o_norm
    return (A, O)

pi = np.random.uniform(0, 1, (1, num_states))
pi[0] = pi[0] / np.sum(pi[0])
    
(A, O) = baum_welch(num_states, sequences,num_tokens, pi)

for i in range(num_states):
    A[i] = A [i]/ np.sum(A, axis = 1)[i]
    
print A
#word_lst = dict((v,k) for k,v in word_num_dict.iteritems())
#
#for word,idx in word_num_dict.iteritems():
#    freq = counts[word]
#    O[idx] = O[idx] / freq
#    
#
#Ot = np.transpose(O)
#for s in range(num_states):
#    s = np.argsort(Ot[s])
#    top_words = s[:20]
#    common = []
#    for w in top_words:
#        common.append(word_lst[w])
#    print common
#
#syllables = {}
#for word in word_num_dict.keys():
#    s = len(h_en.syllables(unicode(word)))
#    if s == 0:
#        s = 1
#    syllables[word] = s
#        
#for s in range(num_states):
#    prob = []
#    one_count = 0
#    two_count = 0
#    three_count = 0
#    four_count = 0
#    more_count = 0
#    for word in syllables.keys():
#        w = word_num_dict[word]
#        syll = syllables[word]
#        if syll == 1:
#            one_count += O[w][s]
#        elif syll == 2:
#            two_count += O[w][s]
#        elif syll == 3:
#            three_count += O[w][s]
#        elif syll == 4:
#            four_count += O[w][s]
#        else:
#            more_count += O[w][s]            
#    prob.append(one_count)
#    prob.append(two_count)
#    prob.append(three_count)
#    prob.append(four_count)
#    prob.append(more_count)
#    print prob
    
#tags = nltk.pos_tag(word_num_dict.keys())

#for s in range(num_states):
#    prob = []
#    noun_count = 0
#    verb_count = 0
#    adj_count = 0
#    prep_count = 0
#    pro_count = 0
#    for tup in tags:
#        word = tup[0]
#        w = word_num_dict[word]
#        tag = tup[1]
#        if tag == 'NN':
#            noun_count += O[w][s]
#        elif tag == 'VBD':
#            verb_count += O[w][s]
#        elif tag == 'IN':
#            prep_count += O[w][s]
#        elif tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
#            adj_count += O[w][s]
#        elif tag == 'PRP' or tag == 'PRP$':
#            pro_count += O[w][s]            
#    prob.append(noun_count)
#    prob.append(verb_count)
#    prob.append(adj_count)
#    prob.append(prep_count)
#    prob.append(pro_count)
#    print prob