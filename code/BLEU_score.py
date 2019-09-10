import math


def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n
    specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on

    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.

    INPUTS:
    sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
    references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
    n :			(int) one of 1,2,3. N-Gram level.


    OUTPUT:
    bleu_score :    (float) The BLEU score
    """

    candidate_list = candidate.split(" ")
    reference_list = []
    for i in references:
        reference_list.append(i.split(" "))

    #brevity penalty
    if brevity:
        # remove -2 for SENTSTART and SENT END
        c = len(candidate_list) - 2
        r = len(reference_list[0]) - 2
        for i in range(1, len(reference_list)):
            ri = abs(len(reference_list[i]) - 2 - c)
            ri2 = abs(r - c)
            if ri < ri2:
                r = len(reference_list[i]) - 2
        if r < c:
            bp = 1
        else:
            bp = math.exp(1 - r / c)
    else:
        bp = 1

    precision = 1
    for i in range(1, n+1):
        precision = precision * p_n(candidate_list, reference_list, i)
    bleu_score = bp * math.pow(precision, float(1/n))

    return bleu_score


def p_n(candidate_list, reference_list, n):
    denominator = 0.0
    for i in reference_list:
        denominator += len(i) - n + 1
    numerator = 0
    for ref in reference_list:
        for i in range(n-1, len(candidate_list)):
            for j in range(n-1, len(ref)):
                for gram in range(n):
                    if candidate_list[i-gram] != ref[j-gram]:
                        break
                    elif gram == n-1:
                        numerator += 1

    return numerator/denominator
