import re


def preprocess(in_sentence, language):
    """
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation

    INPUTS:
    in_sentence : (string) the original sentence to be processed
    language	: (string) either 'e' (English) or 'f' (French)
                   Language of in_sentence

    OUTPUT:
    out_sentence: (string) the modified sentence
    """
    out_sentence = in_sentence

    # Separate sentence ending punctuation
    end_punctuation = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""

    punctuation_pattern = re.compile("([" + end_punctuation + "]+)$")
    out_sentence = re.sub(punctuation_pattern, " " + r'\1', out_sentence)

    # Separate all other punctuation
    other_punctuation = r""",:;()+-<>"="""

    punctuation_pattern = re.compile("([" + other_punctuation + r"])")
    out_sentence = re.sub(punctuation_pattern, " " + r'\1' + " ", out_sentence)

    if language == 'f':
        # also contains the l' pattern
        single_consonant_pattern = r"\b([a-zA-Z]')([a-zA-Z])"
        out_sentence = re.sub(single_consonant_pattern, r'\1' + " " + r'\2', out_sentence)

        # Recombine the undesired words
        out_sentence = re.sub(r"\bd' abord", "d'abord", out_sentence)
        out_sentence = re.sub(r"\bd' accord", "d'accord", out_sentence)
        out_sentence = re.sub(r"\bd' ailleurs", "d'ailleurs", out_sentence)
        out_sentence = re.sub(r"\bd' habitude", "d'habitude", out_sentence)

        que_pattern = r"(qu')([a-zA-Z])"
        out_sentence = re.sub(que_pattern, r'\1' + " " + r'\2', out_sentence)
        puisque_pattern = r"(puisqu')([a-zA-Z])"
        out_sentence = re.sub(puisque_pattern, r'\1' + " " + r'\2', out_sentence)
        lorsque_pattern = r"(lorsqu')([a-zA-Z])"
        out_sentence = re.sub(lorsque_pattern, r'\1' + " " + r'\2', out_sentence)

    # Add sentstart and sentend
    out_sentence = "SENTSTART " + out_sentence + " SENTEND"

    # remove extra white space
    out_sentence = re.sub(" +", " ", out_sentence)

    return out_sentence


if __name__ == "__main__":
    preprocess("d'accord l'aeaa", 'f')
