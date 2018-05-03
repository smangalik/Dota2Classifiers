from sys import argv
import csv
import numpy as np
import random
from numpy.linalg import eig

'''
Trains on the given csv
Outputs a regression model
'''
def train_regression(data_file):
    print('Training...')

    # Read in train data
    X,y = read_csv(data_file)

    # TODO Consider Normalizing

    # train
    A = np.dot(X.T, X)
    b = np.dot(X.T, y.T)

    if is_singular(A): # A is singular
        w = np.dot(A.I, b)
    else:              # A is non-singular
        print('A is non-singular')
        D, V = eig(A)
        D_plus = [1/d if d != 0 else 0.0 for d in list(D)]
        D_plus = np.diag(D_plus)
        A_plus = np.dot(V, D_plus).dot(V.T)
        w = A_plus * b

    #  Output regression model
    regression_model_file = open(regression_model_name,'w')
    for weight in np.array(w):
        weight = np.asscalar(weight)
        regression_model_file.write(str(weight))
        regression_model_file.write('\n')
    regression_model_file.close()

    print('Regression Model created with name',
        regression_model_name)



'''
Tests on the given csv
Reads latest generated regression model
'''
def test_regression(data_file):
    print('Testing...')

    # TODO check that a regression model exists

    # TODO Use a 0-1 activation to decide final labels

    # Read in test data
    X,y = read_csv(data_file)

    # TODO test


# Parse CSV into X and y
def read_csv(data_file):
    csv_reader = csv.reader(data_file)
    labels = []
    features = []
    for row in csv_reader:
        labels.append(float(row[0]))
        features.append([float(i) for i in row[1:]])
    X = np.matrix(features)
    y = np.matrix(labels)
    print('File Read')
    return X,y

'''
Activation funtion for the regression
For Binary Classification this is the signum function
'''
def activation(result):
    if result == 0: # Random guess if 0
        classes = [-1,1]
        return classes[random.randint(len(classes))]
    return np.sign(result)


# TODO (Rewrite) Check if Singular
def is_singular(matrix):
    return matrix.shape[0] == matrix.shape[1] and np.linalg.matrix_rank(matrix) == matrix.shape[0]

'''
How To Run:
python linear_regression.py -train train_data.csv
python linear_regression.py -test train_data.csv

Assumes that first column is the labels and
that the train or test csv has no headers
'''
if __name__ == '__main__':

    if len(argv) != 3:
        print('Invalid number of parameters')

    test_train = argv[1]
    file_str = argv[2]

    regression_model_name = 'regression_model.csv'

    test_train = test_train.replace('-','').lower()
    data_file = open(file_str,'r')

    if test_train == 'train':
        train_regression(data_file)
    elif test_train == 'test':
        test_regression(data_file)
    else:
        print('Expected "train" or "test", got ' + test_train)
