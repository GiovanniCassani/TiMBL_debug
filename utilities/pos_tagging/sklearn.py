import warnings
import utilities.pos_tagging.knn as knn
from collections import defaultdict
from utilities.cooccurrences import dict2matrix, get_counts, sort_words

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics.pairwise import cosine_similarity as cos
    import numpy as np


def sklearn_experiment(training_space, test_space, target_contexts,
                       nn=1, diag_value=None, extra_info=False):

    """
    :param training_space:  a dictionary of dictionaries mapping each word to all the contexts it co-occurred with in
                            the training set, and then to the corresponding co-occurrence count
    :param test_space:      a dictionary of dictionaries mapping each word to all the contexts it co-occurred with in
                            the test set, and then to the corresponding co-occurrence count
    :param target_contexts: an iterable containing the all the contexts that were used to collect co-occurrences
    :param nn:              the number of nearest neighbours to be considered when categorizing a test word
    :param diag_value:      the value to which all the cells on the main diagonal of the matrix of cosine similarities
                            between test and training vectors are set (default is 0, meaning that cells on the main
                            diagonal don't impact the nearest neighbour computation). This option makes it possible to
                            force the model to categorize a test word while ignoring the vector from the training space
                            that correspond to the same word type, thus enforcing generalization
    :param extra_info:      if True, each word in the output dictionary is not only mapped to its correct category, its
                            predicted category, and the categorization accuracy, but also to the list of nearest
                            neighbors, to the cosine distance within which the nearest neighbors are located, and to the
                            distribution of classes of the set of nearest neighbor(s)
    :return hits:           a dictionary mapping each word in the test set to three fields and the corresponding value:
                            'predicted' is the PoS tag that the learner predicted for a test word
                            'correct' is the correct PoS tag as found in the CHILDES corpus
                            'accuracy' is a binary value indicating if 'predicted' and 'correct' match (1) or not (0)
    """

    hits = defaultdict(dict)

    # First get the set of words to be categorized (those in the test set) and the union with the words in the
    # training set. Then get numerical indices for all the words and the target contexts. Finally, store the training
    # and test input spaces in two NumPy 2-dimensional arrays and compute the cosine similarity between words in the
    # test space and words in the training space, setting the values in the diagonal to the desired value. Words from
    # the test set will be the columns, words in the training set will be the rows.
    test_words = set(test_space.keys())
    words = test_words.union(set(training_space.keys()))
    context_indices = sort_words(target_contexts)
    word_indices = sort_words(words)
    inverted_word_indices = {v: k for k, v in word_indices.items()}
    training_matrix = dict2matrix(training_space, word_indices, context_indices)
    test_matrix = dict2matrix(test_space, word_indices, context_indices)
    cosine_similarities = cos(training_matrix, test_matrix)
    if diag_value is not None:
        cosine_similarities[np.diag_indices_from(cosine_similarities)] = diag_value

    # Use the derived cosine similarities to find which words from the training set are closer to each word in the test
    # set to be able to categorize the latter ones. Nearest neighbors are computes using a nearest distance approach,
    # meaning that when two or more words from the training set are at the same closest distance from a test word, they
    # are all considered to assign a PoS tag to the test word (using a majority voting). In case the majority voting
    # also results in a tie, random sampling of one of the PoS tags is performed.
    for word in test_words:
        # get the column index of the test word to be categorized, and get the indices of all the rows that have a
        # cosine similarity to the word to be categorized that is at least as high as the closest distance (if k is 1,
        # otherwise get the cosine similarity value corresponding to the second closest distance (k=2), third closest
        # distance (k=3), and so on)
        c_idx = word_indices[word]
        nearest_indices, closest_distance = knn.get_nearest_indices(cosine_similarities, c_idx, nn=nn)

        # get all the word strings having a high enough cosine similarity value to the word to be categorized
        nearest_neighbors = knn.get_nearest_neighbors(nearest_indices[0], inverted_word_indices)

        # store the PoS tags of the nearest neighbors. if a mapping dictionary is passed, store the PoS tags as
        # indicated in the mapping, otherwise store the PoS tags as found in the strings (it is assumed that wordforms
        # and PoS tags are separated by a tilde ('~'). Count how many times each PoS tag occurs across the nearest
        # neighbors and tally PoS tags by frequency
        tallied_tags = knn.tally_tags(nearest_neighbors)

        # count how many times every PoS tag occurring in the list of nearest neighbors occur, tally PoS tags by
        # frequency and select the PoS tag that occurs more often among the nearest neighbors.
        predicted = knn.categorize(tallied_tags, nearest_neighbors, training_matrix, word_indices)
        hits[word]['predicted'] = predicted
        hits[word]['correct'] = word.split('|')[0]
        hits[word]['accuracy'] = 1 if hits[word]['predicted'] == hits[word]['correct'] else 0
        if extra_info:
            hits[word]['neighbors'] = nearest_neighbors
            hits[word]['cosine'] = closest_distance
            hits[word]['tag_distribution'] = tallied_tags

    return hits, cosine_similarities, word_indices


########################################################################################################################


def print_categorization_outcome(hits, test_space, train_space, target_contexts, file_path):

    """
    :param hits:            a dictionary mapping each word to several fields identified by strings (see the doc string
                            for the sklearn_cosine_similarity() module in context_utils.py)
    :param test_space:      a dictionary of dictionaries mapping test words to the contexts they co-occurred with in the
                            test corpus and the corresponding co-occurrence count
    :param train_space:     a dictionary of dictionaries mapping training words to the contexts they co-occurred with in
                            the training corpus and the corresponding co-occurrence count
    :param target_contexts: an iterable containing the contexts to be considered (the entire set of contexts)
    :param file_path:       the path where the categorization outcome of the sklearn-based classifier are printed.
                            for each item in the test set, its test vector is printed, followed by the correct PoS tag,
                            the PoS tag predicted by the classifier, and the cosine similarity of the item from the
                            training set that was retrieved as the nearest neighbor. The line(s) below each test item,
                            starting with four '#' symbols, show the training vector(s) corresponding to the nearest
                            neighbors
    """

    target_contexts = list(target_contexts)

    with open(file_path, 'a+') as f:

        for k, v in sorted(hits.items()):

            pos = v['correct']
            predicted = v['predicted']
            neighbors = v['neighbors']
            cosine = v['cosine']

            test_instance = get_counts(test_space, k, target_contexts)
            test_instance.append(pos)
            test_instance.append(predicted)
            test_instance.append(str(cosine))
            f.write('\t'.join(test_instance))
            f.write('\n')

            for neighbor in neighbors:
                training_instance = get_counts(train_space, neighbor, target_contexts)
                f.write('\t'.join(['####', '\t'.join(training_instance)]))
                f.write('\n')
