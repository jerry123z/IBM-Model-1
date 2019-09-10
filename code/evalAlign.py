#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *
from preprocess import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model
    """
    if use_cached:
        LM = pickle.load(open(fn_LM+".pickle", "rb"))
    else:
        LM = lm_train(data_dir, language, fn_LM)

    return LM

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model
    """
    if use_cached:
        AM = pickle.load(open(fn_AM+".pickle", "rb"))
    else:
        AM = align_ibm1(data_dir, num_sent, max_iter, fn_AM)

    return AM

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    scores = []
    for i in range(len(eng_decoded)):
        refs = []
        refs.append(eng[i])
        refs.append(google_refs[i])
        scores.append(BLEU_score(eng_decoded[i], refs, n, brevity=False))

    return scores

def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    train_dir = r"C:\Users\Jerry\Documents\CSC401\A2_SMT\data\Hansard\Training\\"
    LM = _getLM(train_dir, 'e', 'lm', use_cached=True)
    print('1k')
    AM1k = _getAM(train_dir, 1000, 100, 'am1k', use_cached=True)
    print('10k')
    AM10k = _getAM(train_dir, 10000, 100, 'am10k', use_cached=True)
    print('15k')
    AM15k = _getAM(train_dir, 15000, 100, 'am15k', use_cached=True)
    print('30k')
    AM30k = _getAM(train_dir, 30000, 100, 'am30k', use_cached=True)
    AM_list = [AM1k, AM10k, AM15k, AM30k]
    AM_names = ["1k", '10k', '15k', '30k']

    french_file = r"C:\Users\Jerry\Documents\CSC401\A2_SMT\data\Hansard\task 5\Task5.f"
    english_file = r"C:\Users\Jerry\Documents\CSC401\A2_SMT\data\Hansard\task 5\Task5.e"
    google_file = r"C:\Users\Jerry\Documents\CSC401\A2_SMT\data\Hansard\task 5\Task5.google.e"
    with open(french_file, 'r') as file:
        lines = file.readlines()
        french_sentences = []
        for line in lines:
            french_sentences.append(preprocess(line, 'f'))
    with open(english_file, 'r') as file:
        lines = file.readlines()
        english_ref = []
        for line in lines:
            english_ref.append(preprocess(line, 'f'))
    with open(google_file, 'r') as file:
        lines = file.readlines()
        google_ref = []
        for line in lines:
            google_ref.append(preprocess(line, 'f'))

    english_candidates = [[], [], [], []]

    # Iteratve over AM
    for i in range(4):
        for sent in range(0, 25):
            english_candidates[i].append(decode.decode(french_sentences[sent], LM, AM_list[i]))


    f = open("Task5.txt", 'w+')
    f.write(discussion)
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")
    for i in range(4):
        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        all_evals = []
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(english_candidates[i], english_ref, google_ref, n)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()
    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    '''
    f = open("Task5.txt", 'w+')
    f.write(discussion) 
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    for i, AM in enumerate(AMs):
        
        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        # Decode using AM #
        # Eval using 3 N-gram models #
        all_evals = []
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(...)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()
    '''
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)
