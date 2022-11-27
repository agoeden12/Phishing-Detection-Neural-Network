import pandas
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold

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

        self.model = Sequential()

        self.model.add(Dense(100, activation='relu', input_shape=(48,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(200, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2, activation='softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':

    data: DataTool = DataTool('data/Phishing_Legitimate_full.csv')
    kfold = KFold(n_splits=5, shuffle=True)

    k_results = {
        'loss': [],
        'accuracy_avg': [],
        'accuracies': []
    }

    for train, test in kfold.split(data.get_features(), data.get_labels()):

        model = PhishingNN().model
        results = model.fit(data.get_features()[train], data.get_labels()[train], epochs=10)
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
    plt.show()