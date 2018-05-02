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
    print(X.shape)
    print(y.shape)
    # Initial values
    w  = np.asmatrix(np.random.rand(X.shape[1])) 
                                    # Weights randomized at the beggining
    errors = np.zeros(X.shape[1])   # Initialize a place holder for errors
    lr = 0.01                       # Learning rate
    epochs = 100                    # Number of iterations
    print('Training with ' + str(epochs), ' iterations...')
    ###################################
    #print('These are the labels')
    #print(y) 
    ###################################
    
    # Initialize Weights Randomly
    # Iterate n times
    for epoch in range(epochs):

        print('Epoch ' + str(epoch + 1) + ', current weights:')
        print(w)

        for i in range(len(X)):
            #print(i)
            x = X[i,:]
            # Calculate error

            ###################################
            #print('current imputs')
            #print(x)
            #print('current weights')
            #print(w)
            ###################################

            result = np.dot(w,x.T)
            error = y - activation(result)
            # Update vectors with error * input * learning_rate
            w += lr * error * X

    # TODO Output perceptron model
    perceptron_model_name = 'perceptron_model.csv'
    perceptron_model_file = open(perceptron_model_name,'w')
 
    weights = ''
    for i in w:
        weights += str(i) + ','
    weights[len(weights) - 1] = ''
    perceptron_model_file.write(weights)
 
    print('Perceptron Model created with name',
        perceptron_model_name)

    perceptron_model_file.close()
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
        labels.append(float(row[0]))
        features.append([float(i) for i in row[1:]])
    X = np.matrix(features)
    y = np.matrix(labels)
    print('File Read')
    return X,y

'''
Activation funtion for the perceptron, takes the inner product of the 
weights and the inputs and returns a list with the predicted outputs.
'''

def activation(result):
    return np.tanh(result[0])
    #return 1/(1 + np.exp(-result[0]))

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
