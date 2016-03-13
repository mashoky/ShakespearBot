import numpy as np
import nltk

from hyphen import Hyphenator, dict_info

from hyphen.dictools import *
import collections
h_en = Hyphenator('en_US')

#file = open('C:/Users/Jagriti/Documents/CS155/project2data/shakesare.txt', 'r')

file = open('C:\Users\manasa\Documents\Caltech\CS 155\ShakespeareProject\shakespeare.txt')

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

num_states = 40
sequences = sequence_list
num_tokens = len(word_num_dict.keys())
    
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
        if s == 0:
            s = 1
        syllables[s].append(k)
    return syllables

def neseq(num_states, num_tokens, pi, A, O, tokens, total_len):
    res = dict((v,k) for k,v in tokens.iteritems())
    syllables = get_syllable_info(tokens)
    seq = []
    rand_init_state = get_random(pi[0])
    state = rand_init_state
    # states by observations now
    Ot = np.transpose(O)
    num_syllables = 0
    while num_syllables < total_len:
        if num_syllables == total_len - 1:
            indices = [tokens[x] for x in syllables[1]]
            row = [Ot[state][i] for i in indices]
            rand_idx = get_random(row)
            rand_obs = indices[rand_idx]
            num_syllables += 1
        else:           
            while True:
                rand_obs = get_random(Ot[state])
                size = len(h_en.syllables(unicode(res[rand_obs], "utf-8")))
                # Account for 1 syllable words that are counted as 0 syllable
                if size == 0:
                    size += 1
                if num_syllables + size < 10:
                    num_syllables += size
                    break
        seq.append(res[rand_obs])
        next_state = get_random(A[state])
        state = next_state
    seq[0] = seq[0].title()
    return " ".join(seq)


pi = np.random.uniform(0, 1, (1, num_states))
pi[0] = pi[0] / np.sum(pi[0])
    
#(A, O) = baum_welch(num_states, sequences,num_tokens, pi)
#print A
#print O
#print np.sum(A, axis=1)
#print np.sum(O, axis=0)
#poem = []
#for i in range(num_states):
#    A[i] = A [i]/ np.sum(A, axis = 1)[i]
#
#l1 = neseq(num_states, num_tokens,pi, A, O,word_num_dict, 5)
#l2 = neseq(num_states, num_tokens, pi, A, O, word_num_dict, 7)
#l3 = neseq(num_states, num_tokens,pi, A, O,word_num_dict, 5)
#poem.append(l1)
#poem.append(l2)
#poem.append(l3)
#print "\n".join(poem)