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

    # Read in perceptron model
    w = []
    w_file = open(regression_model_name,'r')
    for line in w_file:
        if line == '\n':
            continue
        w.append( float(line) )
    w = np.array(w)

    # Read in test data
    X,y_actual = read_csv(data_file)

    y_actual = y_actual.T
    y_pred = np.ones(len(y_actual))

    # Predictions
    for i in range(len(X)):
        x_i = X[i]
        y_pred[i] = np.dot(w, x_i.T)
        y_pred[i] = activation(y_pred[i])

    print('y_actual',y_actual.T)
    print('y_pred',y_pred)

    # 1 is YES and -1 is NO
    TP,TN,FP,FN = 0,0,0,0
    ALL = len(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_actual[i] == 1:
            TP += 1
        elif y_pred[i] == -1 and y_actual[i] == -1:
            TN += 1
        elif y_pred[i] == 1 and y_actual[i] == -1:
            FP += 1
        else:
            FN += 1

    # Calculate Accuracy
    accuracy = (TP + TN) /  ALL
    # Calculate Recall
    recall = TP/(TP + FN)
    # Calculate Precision
    precision = TP/(TP + FP)
    # Calculate F1
    F1 = 2*(precision * recall)/(precision + recall)

    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1:', F1)



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
