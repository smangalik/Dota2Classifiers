from sys import argv

'''
Trains on the given csv
Outputs a perceptron model
'''
def train_perceptron(data_file):
    print('Training...')


'''
Tests on the given csv
Reads latest generated perceptron model
'''
def test_perceptron(data_file):
    print('Testing...')


'''
How To Run
python -train train_data.csv
python -test train_data.csv
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
