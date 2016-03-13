import numpy as np
import nltk
import random

from hyphen import Hyphenator, dict_info

from hyphen.dictools import *
import collections
h_en = Hyphenator('en_US')

file = open('C:/Users/Jagriti/Documents/CS155/project2data/shakespeare.txt', 'r')

#file = open('C:\Users\manasa\Documents\Caltech\CS 155\ShakespeareProject\smallshakespear.txt')

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

num_states = 50
sequences = sequence_list
num_tokens = len(word_num_dict.keys())

s = []
for i in range(len(sequences)):
    seq = sequences[i]
    a = []
    #print i
    for j in range(len(seq)):
        a.append((seq[j], None))
    s.append(a)
    
print np.array(s)
#print sequences
seq = np.array(sequences)
#s = nltk.list()
states = range(20)
tokens = range(len(word_num_dict))
hmm_trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=tokens)

hmm = hmm_trainer.train_unsupervised(s, max_iterations=50)
num_word_dict = dict((v, k) for k, v in word_num_dict.iteritems())
for i in range(14):
    s = hmm.random_sample(random.Random(), 7)
    line = []
    for j in range(len(s)):
        word = num_word_dict[s[j][0]]
        line.append(word)
    b = ' '.join(line)
    print b