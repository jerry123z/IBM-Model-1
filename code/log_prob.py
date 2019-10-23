from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
    Compute the LOG probability of a sentence, given a language model and whether or not to
    apply add-delta smoothing

    INPUTS:
    sentence :	(string) The PROCESSED sentence whose probability we wish to compute
    LM :		(dictionary) The LM structure (not the filename)
    smoothing : (boolean) True for add-delta smoothing, False for no smoothing
    delta : 	(float) smoothing parameter where 0<delta<=1
    vocabSize :	(int) the number of words in the vocabulary

    OUTPUT:
    log_prob :	(float) log probability of sentence
    """
    words = sentence.split()
    logProb = 0

    for i in range(len(words) - 1):
        if words[i] in LM['bi'] and words[i + 1] in LM['bi'][words[i]]:
            count2 = LM['bi'][words[i]][words[i + 1]]
        else:
            count2 = 0

        if words[i] in LM['uni']:
            count1 = LM['uni'][words[i]]
        else:
            count1 = 0

        # print(str(countw1w2) + " and " + str(countw1) + ' ' + words[i] + ' ' + words[i+1])
        if (count1 == 0 or count2 == 0) and delta == 0:  # special case, return 0, or -inf in log space
            return float('-inf')
        if smoothing:
            logProb += log((count2 + delta) / (count1 + delta * vocabSize), 2)
        else:
            logProb += log(count2 / count1, 2)

    return logProb
