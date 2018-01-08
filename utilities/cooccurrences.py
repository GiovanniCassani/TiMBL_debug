import numpy as np
from collections import defaultdict


def dict2matrix(input_dict, row_indices, col_indices):

    """
    :param input_dict:  a dictionary of dictionaries mapping word strings to all the contexts they occurred with in a
                        corpus and each word-context co-occurrence count
    :param row_indices: a dictionary mapping word strings to numerical indices, to be used as row indices in the oputput
                        matrix
    :param col_indices: a dictionary mapping context strings to numerical indices, to be used as column indices in the
                        output matrix
    :return m:          a numpy array with as many rows as there are words and as many columns as there are contexts,
                        with each cell containing the word-context co-occurrence count
    """

    m = np.zeros((len(row_indices), len(col_indices)))

    # loop through all words and contexts, getting the corresponding row and column indices. then get the co-occurrence
    # count for the word-context pair from the input dictionary. If there isn't such a pair (because the word is
    # missing, the context is missing, or the two never occurred together in the input corpus), set the corresponding
    # cell in the output matrix to 0
    for w in row_indices:
        for k in col_indices:
            r = row_indices[w]
            c = col_indices[k]
            try:
                m[r, c] = input_dict[w][k]
            except KeyError:
                m[r, c] = 0

    return m


########################################################################################################################


def file2dict(input_file, sep='\t'):

    """
    :param input_file:      the path to a file containing a string as first item, followed by a variable number of
                            numbers, and finally another string. The first string is treated as the target word. The
                            numbers (in the columns from the second to the second-to-last) are interpreted as
                            co-occurrence  counts between the word and a number of context-words, that do not need to be
                            specified.
    :param sep:             the character delimiting the columns in the input file (default to tab)
    :return co_occurrences: a dictionary of dictionaries mapping each word to all the features (columns) from the input
                            file with which it had a non-zero count, i.e. the columns where the number was not 0.
                            co_occurrence[w][c] thus stores the co-occurrence count between word 'w' and context 'c'
    :return contexts:       a set containing the names automatically given to the contexts, i.e. the columns from the
                            input file
    """

    co_occurrences = defaultdict(dict)
    contexts = set()

    with open(input_file, "r") as f:
        for line in f:
            record = line.strip().split(sep)
            word = record[0].split('~')[0]
            counts = record[1:-1]
            pos = record[-1]
            for c in range(len(counts)):
                # contexts are automatically named as 'cN' where N is the index of the corresponding column in the input
                # file, starting from 0 and excluding the first column containing the wordform
                context = '_'.join(['c', str(c)])
                contexts.add(context)
                co_occurrence = counts[c]
                if int(co_occurrence):
                    co_occurrences['|'.join([pos,word])][context] = co_occurrence

    return co_occurrences, contexts


########################################################################################################################


def get_counts(co_occurrences, word, target_contexts):

    """
    :param co_occurrences:      a dictionary of dictionaries, mapping strings to strings to integers
    :param word:                the target word to consider
    :param target_contexts:     the list (needs to be ordered to preserve alignement across contexts) of contexts being
                                considered
    :return instance:           a list where the first element is the target word and the following are the
                                co-occurrence counts over the entire set of contexts (not just those with which the
                                target word co-occurred), encoded as strings
    """

    # get the set of contexts that co-occurred with the target word
    contexts = set(co_occurrences[word].keys())
    instance = [word]
    for context in target_contexts:
        if context in contexts:
            instance.append(str(co_occurrences[word][context]))
        else:
            instance.append(str(0))

    return instance


########################################################################################################################


def sort_words(l):

    """
    :param l:           an iterable containing PoS|word strings
    :return indices:    a dictionary mapping the same PoS|word strings to integers indicating the numerical index
                        corresponding to each word. Keys are sorted before generating the indices, so to have words from
                        a same PoS tag clustered together
    """

    indices = {}
    idx = 0

    for k in sorted(l):
        # map strings to integers, to make look-up by string easier
        indices[k] = idx
        idx += 1

    return indices
