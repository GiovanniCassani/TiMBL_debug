import os
import argparse
import numpy as np
import utilities.pos_tagging.sklearn as sk
from utilities.cooccurrences import file2dict
from utilities.pos_tagging.timbl import timbl_experiment


def debug(training_file, test_file, output_folder, nn=1, verbose=False):

    """
    :param training_file:   the path to the file to be used for training
    :param test_file:       the path to the file to be used for testing
    :param output_folder:   the folder where output files will be printed
    :param nn:              the number of nearest neighbors to be used for classification
    :param verbose:         a boolean specifying whether to print extra information about the outcome of classification
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    basename = os.path.basename(test_file)
    timbl_output = os.path.join(output_folder, '.'.join([basename, 'timbl']))
    timbl_experiment(training_file, test_file, timbl_output, nn=nn, verbose=verbose)

    training_co_occurrences, contexts = file2dict(training_file)
    test_co_occurrences, contexts = file2dict(test_file)
    # the two lines above assume that contexts in the training and test file are aligned (I know they are in the example
    # files I provide because they have been generated using code I wrote that controls for this)
    sklearn_hits, matrix, word_indices = sk.sklearn_experiment(training_co_occurrences, test_co_occurrences, contexts,
                                                               extra_info=verbose, nn=nn)

    sklearn_output = os.path.join(output_folder, '.'.join([basename, 'sklearn']))
    sk.print_categorization_outcome(sklearn_hits, test_co_occurrences, training_co_occurrences,
                                    contexts, sklearn_output)

    np.set_printoptions(precision=4, suppress=True, linewidth=2000)
    print()
    print("Cosine matrix - computed using sklearn.metrics.pairwise.cosine_similarity:")
    print(matrix)
    print()
    print("Mapping between rows/columns and words:")
    print("  columns with 0s are words in the training set but not in the test set, that don't need to be categorized")
    print("  rows with 0s are words in the test set but not in the training set, "
          "so never retrievable as nearest neighbors")
    print(sorted(list(word_indices.items())))


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description="Select distributional contexts and perform "
                                                 "a PoS tagging experiment using TiMBL.")

    parser.add_argument("-f", "--training_file", required=True, dest="training",
                        help="Specify the path to the training file.")
    parser.add_argument("-t", "--test_file", required=True, dest="test",
                        help="Specify the path to the test file.")
    parser.add_argument("-o", "--output_folder", required=True, dest="output_folder",
                        help="Specify the folder where summary files and experiments' output files are stored.")
    parser.add_argument("-n", "--nearest_neighbors", dest="nn", default=1,
                        help="Set the number of nearest neighbors to consider when categorizing a test word.")
    parser.add_argument("-v", "--verbose", action="store_true", dest='verbose',
                        help="Specify whether to print extra information to output as a result of the categorization"
                             "experiments in TiMBL and sklearn.")

    args = parser.parse_args()

    debug(args.training, args.test, args.output_folder, nn=args.nn, verbose=args.verbose)


########################################################################################################################


if __name__ == '__main__':
    main()
