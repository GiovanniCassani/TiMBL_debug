import os


def timbl_experiment(training_file, test_file, output_file, nn=1, verbose=False):

    """
    :param training_file:   the path to the file to be used for training
    :param test_file:       the path to the file to be used for testing
    :param output_file:     the path where the categorization output will be printed
    :param nn:              the number of nearest neighbors to be considered
    :param verbose:         a boolean specifying whether to print extra information
    """

    if verbose:
        # di prints the cosine distance of the nearest neighbor(s)
        # db prints the class distribution of the nearest neighbor(s) set
        # n prints the nearest neighbor(s) set
        timbl_cmd = 'timbl -k' + str(nn) + ' -mC:I1 -w0 +v di+db+n -f ' \
                    + training_file + ' -t ' + test_file + ' -o ' + output_file
    else:
        timbl_cmd = 'timbl -k' + str(nn) + ' -mC:I1 -w0 -f ' \
                    + training_file + ' -t ' + test_file + ' -o ' + output_file

    os.system(timbl_cmd)
