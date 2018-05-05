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

  "test" outputs a model_file with a displayed default file name.
  Perceptron will display epoch progress and final weight vector

  "train" reads a file with that default file name.
  Both will also print out Accuracy, Precision, Recall, and F1
