NAME
  perceptron.py - A single perceptron for binary classification
  linear_regression.py - Multiple Linear Regression for binary classification

SYNOPSIS
  python perceptron.py -train train_data.csv
  python perceptron.py -test test_data.csv

  python linear_regression.py -train train_data.csv
  python linear_regression.py -test test_data.csv

DESCRIPTION
  Binary Classifiers for numerical values with labels -1 and 1
  Assumes that the first column of the CSV is the sample label

  Options:
    -train
      outputs a model_file with a displayed default model file name.
    -test
      reads a file with the default model file name.
      Prints out Accuracy, Precision, Recall, and F1
