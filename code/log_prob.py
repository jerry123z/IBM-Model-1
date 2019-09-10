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

    log_prob = float(0)
    words = sentence.split(' ')

    for word in range(1, len(words) - 1):
        if words[word - 1] in LM["bi"]:
            if words[word] in LM["bi"][words[word - 1]]:
                bi_count = LM["bi"][words[word - 1]][words[word]]
            else:
                bi_count = 0
        else:
            bi_count = 0

        if words[word] in LM["uni"]:
            uni_count = LM["uni"][words[word]]
        else:
            uni_count = 0

        if smoothing:
            bi_count = bi_count + delta
            uni_count = uni_count + delta * vocabSize

        # Cases for -inf
        if bi_count == 0 and uni_count == 0:
            log_prob = float('-Inf')
            break
        elif bi_count == 0 and uni_count != 0:
            log_prob = float('-Inf')
            break
        else:
            log_prob = log_prob + log(bi_count, 2.0)
            log_prob = log_prob - log(uni_count, 2.0)

    return log_prob
