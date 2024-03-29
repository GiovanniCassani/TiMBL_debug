import numpy as np
from collections import Counter


def get_nearest_indices(cosine_similarities, idx, nn=1):

    """
    :param cosine_similarities: a NumPy 2-dimensional array
    :param idx:                 an integer indicating which column to consider
    :param nn:                  an integer indicating the number of nearest neighbours to consider (the function uses
                                nearest distances rather than neighbours: if two or more words are at the same closest
                                distance they're all consider - when nn=1, as in the default)
    :return nearest_indices:    a tuple whose first element contains the row indices from the input NumPy array
                                indicating the cells with the highest values in the column indicated by the input
                                parameter idx. The second element of the tuple is empty.
    :return 1-t:                the cosine distance (i.e. 1 - the cosine similarity) of the closest neighbour(s)
    """

    # sort all the columns in the NumPy array indepently and in descending order
    cosine_similarities_sorted = np.sort(cosine_similarities, axis=0)[::-1]

    # get the value corresponding to the closest distance (if nn=1) or second closest distance (if nn=2), and so on
    # if the vector is shorter then the chosen value for nn, the function simply takes the smallest value in the column,
    # which is the last one since the column is sorted in descending order
    try:
        t = sorted(set(cosine_similarities_sorted[:, idx]), reverse=True)[nn-1]
    except IndexError:
        t = sorted(set(cosine_similarities_sorted[:, idx]), reverse=True)[-1]

    # get the vector of row indices from the original, unsorted NumPy array that have a distance equal or higher than
    # the value of the desired number of neighbours (distances) set by nn
    nearest_indices = np.where(cosine_similarities[:, idx] >= t)

    return nearest_indices, 1 - t


########################################################################################################################


def get_nearest_neighbors(nearest_indices, words):

    """
    :param nearest_indices: a tuple whose first element contains the row indices from the input NumPy array indicating
                            the cells with the highest values in the column indicated by the input parameter idx. The
                            second element of the tuple is empty.
    :param words:           a dictionary mapping numerical indices to word strings
    :return neighbors:      a set of strings containing those strings that match the indices in the input tuple
    """

    neighbors = set()
    for i in nearest_indices:
        neighbors.add(words[i])

    return neighbors


########################################################################################################################


def tally_tags(l):

    """
    :param l:               an iterable of strings, consisting of a word form and a PoS tag separated by a tilde ("~")
    :return tallied_tags:   an sorted list of tuples, each containing a string as first element (a PoS tag) and a
                            frequency count as second element, indicating the frequency count of the PoS tag among the
                            nearest neighbors provided in the input iterable
    """

    pos_tags = list()
    for i in l:
        # isolate the PoS tag
        tag = i.split("|")[0]
        pos_tags.append(tag)

    # count frequencies of occurrence for each tag in the list of neighbours and return the resulting list of tuples
    tallied_tags = Counter(pos_tags).most_common()
    return tallied_tags


########################################################################################################################


def categorize(tallied_tags, nearest_neighbours, training_matrix, word_indices):

    """
    :param tallied_tags:        a list of tuples where the first element is a string (a PoS tag) and the second is its
                                frequency of occurrence among the strings in a list (see the documentation of the
                                function get_nearest_neighbors() for further details)
    :param nearest_neighbours:
    :param training_matrix:     a 2-dimensional NumPy array containing co-occurrence counts
    :param word_indices:        a dictionary mapping strings to indices to be used as row indices in accessing the NumPy
                                array
    :return predicted:          a string indicating the predicted PoS tag given the tallied tags and the nearest
                                neighbours together with the frequency information contained in the training matrix
    """

    f = 0
    most_frequent = []
    for t in tallied_tags:
        # for every tuple check whether the frequency count is higher than the previous one (only possible for the first
        # element since the list of tuples is sorted by frequency) and if this is the case store the corresponding PoS
        # tag
        if t[1] > f:
            f = t[1]
            most_frequent.append(t[0])
        # if the frequency count is the same of the previous one, i.e. there are two PoS tags that occur equally
        # frequently among the nearest neighbours, store the current PoS tag
        elif t[1] == f:
            most_frequent.append(t[0])
        else:
            break

    # Resolve ties by picking the PoS tag of the nearest neighbour that occurred more frequently in the training set, by
    # looking at the co-occurrence pattern: occurrences of the words outside of relevant contexts are not considered
    # If frequency is also not enough to break the tie, pick randomly a tag from the most frequent words
    if len(most_frequent) > 1:
        max_freq = 0
        most_frequent = []
        for neighbour in nearest_neighbours:
            r_idx = word_indices[neighbour]
            freq = sum(training_matrix[r_idx, :])
            pos = neighbour.split("|")[0]
            if freq > max_freq:
                max_freq = freq
                most_frequent.append(pos)
        if len(most_frequent) > 1:
            i = int(np.random.randint(0, high=len(most_frequent), size=1))
            predicted = most_frequent[i]
        else:
            predicted = most_frequent[0]
    else:
        predicted = most_frequent[0]

    return predicted
