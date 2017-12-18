The main function is debug.py. It can be called directly from command line and it requires three arguments to be passed: 
- a training file
- a test file
- the path to a folder
The function depends on TiMBL to be installed and training and test spaces formatted as in the files provided, i.e. tab-separated columns with the first containing the word form, the following n columns containing numbers, and the last column containing the PoS tag of the word. Numbers are co-occurrence counts of words and contexts in an input corpus (this step of the experiment is not included here: this code assumes the co-occurrence files are already available).

The function carries out a PoS tagging experiment using cosine as a distance metric, using both TiMBL and a custom Python implementation that relies on sklearn functionalities. It generates two output files, one per tagging algorithm.

The code is in Python3 and requires standard scientific libraries.

For any further detail, mail to: giovanni DOT cassani AT uantwerpen DOT be
