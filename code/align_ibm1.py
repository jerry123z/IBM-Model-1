from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import pickle
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
    Implements the training of IBM-1 word alignment algoirthm.
    We assume that we are implemented P(foreign|english)

    INPUTS:
    train_dir :     (string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider
    max_iter :     	(int) the maximum number of iterations of the EM algorithm
    fn_AM :         (string) the location to save the alignment model

    OUTPUT:
    AM :            (dictionary) alignment model structure

    The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word']
    is the computed expectation that the foreign_word is produced by english_word.

            LM['house']['maison'] = 0.5
    """
    AM = {}

    # Read training data
    models = read_hansard(train_dir, num_sentences)

    # Initialize AM uniformly
    AM = initialize(models[0], models[1])

    # Iterate between E and M steps
    for i in range(max_iter):
        em_step(AM, models[0], models[1])

    # set SENTSTART and SENTEND
    AM['SENTSTART'] = {}
    AM['SENTSTART']['SENTSTART'] = 1
    AM['SENTEND'] = {}
    AM['SENTEND']['SENTEND'] = 1

    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM

# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
    Read up to num_sentences from train_dir.

    INPUTS:
    train_dir :     (string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider


    Make sure to preprocess!
    Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

    Make sure to read the files in an aligned manner.
    """
    english = []
    french = []
    all_files = os.listdir(train_dir)
    curr_sentence = 0
    # while curr_sentence < num_sentences:
    for file in all_files:
        if curr_sentence >= num_sentences:
            break

        # only hit each file once
        if file.endswith(".e"):
            name = os.path.splitext(file)[0]
            # open both languages
            english_file = open(os.path.join(train_dir, name + ".e"))
            french_file = open(os.path.join(train_dir, name + '.f'))
            english_lines = english_file.readlines()
            french_lines = french_file.readlines()

            for i in range(len(english_lines)):
                if curr_sentence >= num_sentences:
                    break
                english_processed_line = preprocess(english_lines[i], "e")
                french_processed_line = preprocess(french_lines[i], "f")
                english.append(english_processed_line.split(" "))
                french.append(french_processed_line.split(" "))
                curr_sentence += 1

    return[english, french]


def initialize(eng, fre):
    """
    Initialize alignment model uniformly.
    Only set non-zero probabilities where word pairs appear in corresponding sentences.
    """
    AM = {}
    for sen in range(len(eng)):
        for i in range(1, len(eng[sen])-1):
            for j in range(1, len(fre[sen])-1):
                if not eng[sen][i] in AM:
                    AM[eng[sen][i]] = {}
                if not fre[sen][j] in AM[eng[sen][i]]:
                    AM[eng[sen][i]][fre[sen][j]] = 1

    # Normalize
    english_words = AM.keys()

    for i in english_words:
    # Normalizing
        count = 0
        french_words = AM[i].keys()
        for j in french_words:
            count = count + AM[i][j]
        for j in french_words:
            AM[i][j] = AM[i][j] / count

    return AM


def em_step(t, eng, fre):
    """
    One step in the EM algorithm.
    Follows the pseudo-code given in the tutorial slides.
    """
    # initialize stuff
    english_words = t.keys()
    tcount = {}
    total = {}

    for sen in range(len(eng)):
        # change the sentences into unique lists
        unique_french = fre[sen]
        unique_english = eng[sen]
        # Remove SENTSTART and SENTEND

        if 'SENTEND' in unique_french:
            unique_french.remove('SENTEND')
        if 'SENTSTART' in unique_french:
            unique_french.remove('SENTSTART')
        if 'SENTEND' in unique_english:
            unique_english.remove('SENTEND')
        if "SENTSTART" in unique_english:
            unique_english.remove('SENTSTART')
        unique_french = list(dict.fromkeys(unique_french))
        unique_english = list(dict.fromkeys(unique_english))

        for french_word in unique_french:
            denom_c = 0
            for english_word in unique_english:
                # denom_c += P(f|e) * F.count(f)
                denom_c = denom_c + t[english_word][french_word] * fre[sen].count(french_word)

            for english_word in unique_english:
                tcount[english_word][french_word] += t[english_word][french_word] * unique_french.count(french_word) * unique_english.count(
                    english_word) / denom_c
                total[english_word] += t[english_word][french_word] * unique_french.count(french_word) * unique_english.count(
                    english_word) / denom_c

    # update model
    for i in english_words:
        french_words = t[i].keys()
        for j in french_words:
            t[i][j] = tcount[i][j] / total[i]
