from sys import argv
import csv
import numpy as np

'''
Trains on the given csv
Outputs a regression model
'''
def train_regression(data_file):
    print('Training...')

    # Read in train data
    X,y = read_csv(data_file)

    # TODO train
    # Select randomized initial linear boundary
    # Calculate Squared Loss
    # Adjust parameters of linear boundary
    # Repeat until amount of loss scales down
    # Use a 0-1 activation to decide final labels

    # TODO Output regression model
    regression_model_name = 'regression_model.csv'
    regression_model_file = open(regression_model_name,'w')

    print('Regression Model created with name',
        regression_model_name)

    regression_model_file.close()
'''
Tests on the given csv
Reads latest generated regression model
'''
def test_regression(data_file):
    print('Testing...')

    # TODO check that a regression model exists

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

    test_train = test_train.replace('-','').lower()
    data_file = open(file_str,'r')

    if test_train == 'train':
        train_regression(data_file)
    elif test_train == 'test':
        test_regression(data_file)
    else:
        print('Expected "train" or "test", got ' + test_train)
