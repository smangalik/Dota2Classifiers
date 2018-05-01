from sys import argv
import csv
import numpy as np

'''
Trains on the given csv
Outputs a perceptron model
'''
def train_perceptron(data_file):
    print('Training...')

    # Read in train data
    X,y = read_csv(data_file)

    # TODO Train Perceptron
    # Initialize Weights Randomly
    # Iterate n times
        # Calculate error
        # Update vectors with error * input * learning_rate


    # TODO Output perceptron model
    perceptron_model_name = 'perceptron_model.csv'
    perceptron_model_file = open(perceptron_model_name,'w')
    print('Perceptron Model created with name',
        perceptron_model_name)


'''
Tests on the given csv
Reads latest generated perceptron model
'''
def test_perceptron(data_file):
    print('Testing...')

    # TODO check that a perceptron model exists

    # Read in test data
    X,y = read_csv(data_file)

    # TODO test


# Parse CSV into X and y
def read_csv(data_file):
    csv_reader = csv.reader(data_file)
    labels = []
    features = []
    for row in csv_reader:
        labels.append(row[0])
        features.append(row[1:])
    X = np.matrix(features)
    y = np.matrix(labels)
    print('File Read')
    return X,y


'''
How To Run:
python perceptron.py -train train_data.csv
python perceptron.py -test train_data.csv

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
        train_perceptron(data_file)
    elif test_train == 'test':
        test_perceptron(data_file)
    else:
        print('Expected "train" or "test", got ' + test_train)
