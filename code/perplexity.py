from log_prob import *
from preprocess import *
from lm_train import *
import os


def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
    Computes the preplexity of language model given a test corpus

    INPUT:

    LM : 		(dictionary) the language model trained by lm_train
    test_dir : 	(string) The top-level directory name containing data
                e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    language : `(string) either 'e' (English) or 'f' (French)
    smoothing : (boolean) True for add-delta smoothing, False for no smoothing
    delta :     (float) smoothing parameter where 0<delta<=1
    """

    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])

    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue

        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)

            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
        pp = 2**(-pp/N)
    return pp


# test
if __name__ == '__main__':
    test_dir = r"..\data\Hansard\Testing\\"
    train_dir = r"..\data\Hansard\Training\\"
    test_LM_f = lm_train(train_dir, "f", "f_temp")
    test_LM_e = lm_train(train_dir, "e", "e_temp")
    print(preplexity(test_LM_e, test_dir, "e", False, 0))
    print(preplexity(test_LM_f, test_dir, "f", False, 0))
    print(preplexity(test_LM_e, test_dir, "e", True, 0.2))
    print(preplexity(test_LM_f, test_dir, "f", True, 0.2))
    print(preplexity(test_LM_e, test_dir, "e", True, 0.4))
    print(preplexity(test_LM_f, test_dir, "f", True, 0.4))
    print(preplexity(test_LM_e, test_dir, "e", True, 0.8))
    print(preplexity(test_LM_f, test_dir, "f", True, 0.8))
