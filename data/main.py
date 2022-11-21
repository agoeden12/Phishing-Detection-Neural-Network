import pandas
import numpy as np
import keras as k

def method1():
    #Step 1 load training data to neural network
    print("Training data set matrix x: " + "\n")
    data1 = pandas.read_csv('Phishing_Legitimate_full.csv')
    x = data1.iloc[0]
    x = np.matrix(x)
    print(x)

    # works cited below:
    # https://nam04.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmachinelearningmastery.com%2Ftutorial-first-neural-network-python-keras%2F&data=05%7C01%7Cbbaker74%40students.kennesaw.edu%7Ccc10fb3ca18c461aecf608dabdbe69e7%7C45f26ee5f134439ebc93e6c7e33d61c2%7C1%7C0%7C638030922760593105%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=zE6kyZai%2B2qs%2B5oHjHzSi4AhBhL8pUcxkbTx%2Fgp9TWM%3D&reserved=0

    #---------------------------------------------------------------------------------------------------------------------------
    #my stuff from hw3
    # first neural network with keras tutorial
    #import tensorflow as tf
    # first neural network with keras tutorial
    #from numpy import loadtxt
    # load the dataset
    #dataset = loadtxt('MNIST_HW3Edit2.csv', delimiter=',')  # has no labels

    #z = dataset[:, 0:8] # x divided into 5 smaller groups
    #y = dataset[:, 8]  # target

    #model = k.Sequential()
    #model.add(tf.keras.layers.Dense(12, input_shape=(8,), activation='relu'))
    #model.add(tf.keras.layers.Dense(8, activation='relu'))
    #model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # compile the keras model
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    #model.fit(z, y, epochs=150, batch_size=10)
    # evaluate the keras model
    #accuracy = model.evaluate(z, y)
    #print('Accuracy: %.2f' % (accuracy*100))


def main():
    method1()


if __name__ == '__main__':
    main()

