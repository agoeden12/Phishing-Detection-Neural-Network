import pandas
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold


First_layer = 100
First_dropout = 0.2
Second_layer = 200
Second_dropout = 0.2

# Class with functions to help data modification
class DataTool():

    def __init__(self, input_file: str) -> None:

        pd_file = pandas.read_csv(input_file)
        pd_file.pop('id')

        self.data: Dict(str, np.ndarray) = {'labels': None, 'features': None}
        self.data['labels'] = np.array(pd_file.pop('CLASS_LABEL'))
        self.data['features'] = np.array(pd_file)

    def get_labels(self):
        return self.data['labels']

    def get_features(self):
        return self.data['features']

class PhishingNN():

    def __init__(self) -> None:
        self.build_model()

    def build_model(self):
        # Dropout randomly sets input units to 0 with a frequency rate, this helps prevent overfitting.
        # model parameters
        
        first_layer = First_layer 
        first_dropout = First_dropout 
        second_layer = Second_layer
        second_dropout = Second_dropout


        self.model = Sequential()

        self.model.add(Dense(first_layer, activation='relu', input_shape=(48,)))
        self.model.add(Dropout(first_dropout))
        self.model.add(Dense(second_layer, activation='relu'))
        self.model.add(Dropout(second_dropout))
        self.model.add(Dense(2, activation='softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':

    data: DataTool = DataTool('data/Phishing_Legitimate_full.csv')
    kfold = KFold(n_splits=5, shuffle=True)

    # Fitting parameters
    epochs = 10

    k_results = {
        'loss': [],
        'accuracy_avg': [],
        'accuracies': []
    }

    for train, test in kfold.split(data.get_features(), data.get_labels()):

        model = PhishingNN().model
        results = model.fit(data.get_features()[train], data.get_labels()[train], epochs=epochs)
        eval = model.evaluate(data.get_features()[test], data.get_labels()[test], verbose=0)

        k_results['loss'].append(eval[0])
        k_results['accuracy_avg'].append(eval[1])
        k_results['accuracies'].append(results.history['accuracy'])

    averageAccuracy = np.mean(k_results['accuracy_avg'])
    print(f"Average Accuracy: {(averageAccuracy*100):.2f}%")

    fig, axs = plt.subplots(1, 1)
    colors = ["red", "green", "blue", "yellow", "black"]

    for i in range(5):
        axs.plot(k_results['accuracies'][i], color=colors[i])

    axs.set_title('model accuracy')
    axs.set_ylabel('accuracy')
    axs.set_xlabel('epoch')
    #First_layer  
    #First_dropout 
    #Second_layer 
    #Second_dropout
    text = f"""
    First_layer    = {First_layer} 
    First_dropout  = {First_dropout}
    Second_layer   = {Second_layer} 
    Second_dropout = {Second_dropout}
    """
    plt.figtext(0.0,0.0,text, ha="left")
    plt.subplots_adjust(bottom=.2)
    #plt.figtext(0,-0.05,f"First_dropout  {    First_dropout }", ha="left")
    #plt.figtext(0,-0.10,f"Second_layer   {    Second_layer } ", ha="left")
    #plt.figtext(0,-0.15,f"Second_dropout {    Second_dropout}", ha="left")




    plt.show()