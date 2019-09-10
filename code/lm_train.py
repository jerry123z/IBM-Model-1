from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
    This function reads data from data_dir, computes unigram and bigram counts,
    and writes the result to fn_LM

    INPUTS:

    data_dir	: (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language	: (string) either 'e' (English) or 'f' (French)
    fn_LM		: (string) the location to save the language model once trained

    OUTPUT

    LM			: (dictionary) a specialized language model

    The file fn_LM must contain the data structured called "LM", which is a dictionary
    having two fields: 'uni' and 'bi', each of which holds sub-structures which
    incorporate unigram or bigram counts

    e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
          LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """

    LM = dict()
    LM["uni"] = dict()
    LM['bi'] = dict()

    all_files = os.listdir(data_dir)
    for file in all_files:
        if file.endswith("." +language):
            print(file)
            with open(os.path.join(data_dir, file)) as curr_file:
                lines = curr_file.readlines()
                for line in lines:
                    processed_line = preprocess(line, language)
                    words = processed_line.split(' ')

                    # Start at index 1 to avoid SENTSTART and end at index -2 to avoid SENTEND for Unigram model
                    for word in range(1, len(words)-1):
                        # Unigrams
                        if words[word] in LM["uni"]:
                            LM["uni"][words[word]] += 1
                        else:
                            LM["uni"][words[word]] = 1

                        # Bigrams
                        if words[word-1] in LM["bi"]:
                            if words[word] in LM["bi"][words[word-1]]:
                                LM["bi"][words[word - 1]][words[word]] += 1
                            else:
                                LM["bi"][words[word - 1]][words[word]] = 1
                        else:
                            LM["bi"][words[word-1]] = dict()
                            LM["bi"][words[word - 1]][words[word]] = 1

                    # Last word for bigrams, SENTEND
                    if words[-2] in LM["bi"]:
                        if words[-1] in LM["bi"][words[-2]]:
                            LM["bi"][words[-2]][words[-1]] += 1
                        else:
                            LM["bi"][words[-2]][words[-1]] = 1
                    else:
                        LM["bi"][words[-2]] = dict()
                        LM["bi"][words[-2]][words[-1]] = 1

    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(LM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return LM

if __name__ == "__main__":
    lm_train("C:\\Users\\Jerry\\Documents\\CSC401\\A2_SMT\\data\\Hansard\\Training", "e", "e")
