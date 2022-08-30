from collections import Counter
import csv
import string
import nltk
import pandas as pd
import random
import math
import datetime

tunigrams=[]
tbigrams=[]
   
def countWOrdUni(strg):    
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    exclude = set(string.punctuation)
    text = strg.translate(replace_punctuation).lower() 
    unig = Counter(nltk.ngrams(text.split(),1))
    dic={''.join(ch for ch in key if ch not in exclude):value for key, value in unig.items()}
    unigm=[''.join(ch for ch in key if ch not in exclude) for key, value in unig.items()]
    return dic,unigm

def countWOrdBi(strg):    
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    exclude = set(string.punctuation)
    text = strg.translate(replace_punctuation).lower() 
    bigmC = Counter(nltk.bigrams(text.split()))
    dic={' '.join(ch for ch in key if ch not in exclude):value for key, value in bigmC.items()}
    bigm=[' '.join(ch for ch in key if ch not in exclude) for key, value in bigmC.items()]
    return dic,bigm

def WordPro(P_word,N_word,thrs):
    uniProb={}    
    for tg, n_count in N_word.items():        
        if tg in P_word.keys() and P_word[tg]>=thrs:
            p_count=P_word[tg]
            c2 = float(p_count)
            p = (c2+1)/(c2+float(n_count)+2)
            uniProb[tg ]=(p,p_count,n_count)
           #print (uniProb[tg ],p)
    return uniProb

def classifier( thrs, unigrams, bigrams, U_list,B_list):
    WordProbMult = 1
    
    Positive_Bigams = []
    Positive_Unigams = []
    for unigram in unigrams:
        if unigram in U_list.keys():
            pr = float(U_list[unigram][0])
            if pr >= thrs:
                WordProbMult = WordProbMult * (1 - pr)
                Positive_Unigams.append(unigram)

    for bigram in bigrams:
        if bigram in B_list.keys():
            pr = float(B_list[bigram][0])    
            if pr >= thrs :
                WordProbMult = WordProbMult * (1 - pr)
                Positive_Bigams.append(bigram)
            else:
                Positive_Bigams.append(bigram)
    NoisyOR = 1 - WordProbMult
    return NoisyOR, Positive_Unigams, Positive_Bigams