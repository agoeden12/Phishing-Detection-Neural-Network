import pandas
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold


first_layer = 100
first_dropout = 0.2
second_layer = 200
second_dropout = 0.2

# Class with functions to help data modification
class DataTool:
    def __init__(self, input_file: str) -> None:

        pd_file = pandas.read_csv(input_file)
        pd_file.pop("id")

        self.data: Dict(str, np.ndarray) = {"labels": None, "features": None}
        self.data["labels"] = np.array(pd_file.pop("CLASS_LABEL"))
        self.data["features"] = np.array(pd_file)

    def get_labels(self):
        return self.data["labels"]

    def get_features(self):
        return self.data["features"]

def build_model():
    # Dropout randomly sets input units to 0 with a frequency rate, this helps prevent overfitting.
    # model parameters

    model = Sequential()

    model.add(Dense(first_layer, activation="relu", input_shape=(48,)))
    model.add(Dropout(first_dropout))
    model.add(Dense(second_layer, activation="relu"))
    model.add(Dropout(second_dropout))

    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":

    data: DataTool = DataTool("data/Phishing_Legitimate_full.csv")
    kfold = KFold(n_splits=5, shuffle=True)

    # Fitting parameters
    epochs = 10

    k_results = {"loss": [], "accuracy_avg": [], "accuracies": []}

    for train, test in kfold.split(data.get_features(), data.get_labels()):

        model = build_model()
        results = model.fit(
            data.get_features()[train], data.get_labels()[train], epochs=epochs
        )
        eval = model.evaluate(
            data.get_features()[test], data.get_labels()[test], verbose=0
        )

        k_results["loss"].append(eval[0])
        k_results["accuracy_avg"].append(eval[1])
        k_results["accuracies"].append(results.history["accuracy"])

    averageAccuracy = np.mean(k_results["accuracy_avg"])
    print(f"Average Accuracy: {(averageAccuracy*100):.2f}%")

    fig, axs = plt.subplots(1, 1)
    colors = ["red", "green", "blue", "yellow", "black"]

    for i in range(5):
        axs.plot(k_results["accuracies"][i], color=colors[i])

    axs.set_title("model accuracy")
    axs.set_ylabel("accuracy")
    axs.set_xlabel("epoch")

    text = f"""
    First_layer    = {first_layer}
    First_dropout  = {first_dropout}
    Second_layer   = {second_layer}
    Second_dropout = {second_dropout}
    """
    plt.figtext(0.0, 0.0, text, ha="left")
    plt.subplots_adjust(bottom=0.2)

    plt.show()