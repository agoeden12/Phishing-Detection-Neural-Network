import numpy as np
import pandas


def method1():
    # Step 1 load training data to neural network
    print("Training data set matrix x: " + "\n")
    data1 = pandas.read_csv('Phishing_Legitimate_full.csv')
    x = data1.iloc[0]
    x = np.matrix(x)
    print(x)


def main():
    method1()


if __name__ == '__main__':
    main()

