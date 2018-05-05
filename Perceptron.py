from sys import argv
import csv
import numpy as np
import collections

'''
Trains on the given csv
Outputs a perceptron model
'''
def train_perceptron(data_file):
    print('Training... ',end='')

    # Read in train data
    X,y = read_csv(data_file)
    X = min_max_normalize(X)
    # print(X.shape)
    # print(y.shape)

    # Initial values
    w  = np.asmatrix(np.random.rand(X.shape[1]))
                                    # Weights randomized at the beggining
    errors = np.zeros(X.shape[1])   # Initialize a place holder for errors
    lr = 0.001                      # Learning rate
    epochs = 101                     # Number of iterations
    print('Training with ' + str(epochs),
          ' epochs, learning rate is ' + str(lr))

    # Initialize Weights Randomly
    # Iterate n times
    for epoch in range(epochs):

        # TODO Vectorize
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]

            # Calculate Error
            prediction = np.dot(w,x_i.T)
            error = y_i - activation(prediction)

            # Update Weights
            w += (lr * error * x_i)

        if epoch % 5 == 0:
            print('Epoch:',epoch)

    # Output perceptron model
    perceptron_model_file = open(perceptron_model_name,'w')

    for weight in np.array(w)[0]:
        perceptron_model_file.write(str(weight))
        perceptron_model_file.write('\n')

    print('Perceptron Model created with name',perceptron_model_name)

    perceptron_model_file.close()



'''
Tests on the given csv
Reads latest generated perceptron model
'''
def test_perceptron(data_file):
    print('Testing... ', end='')

    # Read in perceptron model
    w = []
    w_file = open(perceptron_model_name,'r')
    for line in w_file:
        if line == '\n':
            continue
        w.append( float(line) )
    w = np.array(w)

    # Read in test data
    X,y_actual = read_csv(data_file)
    X = min_max_normalize(X)

    y_pred = np.ones(len(y_actual))

    # TODO k-fold validation

    # Predictions
    for i in range(len(X)):
        x_i = X[i]
        y_pred[i] = np.dot(w, x_i.T)
        y_pred[i] = np.sign(activation(y_pred[i]))

    print('Guess Counts:',collections.Counter(y_pred))

    print('y_actual',y_actual)
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
    y = np.array(labels)
    print('File Read')
    return X,y

# Normalize Columns by Min and Mix
def min_max_normalize(data):
    normalized = data.T
    for i in range(len(normalized)):
        column = normalized[i]
        if np.min(column) == -1 and np.max(column) == 1:
            continue
        if np.min(column) == 0 and np.max(column) == 0:
            continue
        column = 2 * (column - np.min(column))/(np.max(column)-np.min(column)) - 1
        normalized[i] = column
    return normalized.T


'''
Activation funtion for the perceptron, takes the inner product of the
weights and the inputs and returns a list with the predicted outputs.
'''

def activation(result):
    return np.tanh(np.asscalar(result))
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

    perceptron_model_name = 'perceptron_model.csv'

    if test_train == 'train':
        train_perceptron(data_file)
    elif test_train == 'test':
        test_perceptron(data_file)
    else:
        print('Expected "train" or "test", got ' + test_train)
