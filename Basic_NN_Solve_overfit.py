import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sn

from sklearn.metrics import confusion_matrix,roc_auc_score,recall_score,f1_score,precision_score



# JSON path bc
DATA_PATH = "C:\\Users\\Kartikeya\\PycharmProjects\\Industrial_Project_VII\\processing_with_yt.json"


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y

def predict(model,X):
    X=X[np.newaxis,...]
    prediction = model.predict(X)         #x = (1,dims)
    #its 2D [[0.1,0.3,.....]] result of softmax
    #extract max
    p1_index=np.argmax(prediction,axis = 1)


    return p1_index

def predict_2(model,X):
    X=X[np.newaxis,...]
    prediction = model.predict(X)         #x = (1,dims)
    #its 2D [[0.1,0.3,.....]] result of softmax
    #extract max
    pre=prediction.flatten()


    return pre



def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")

    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    # load data
    X, y = load_data(DATA_PATH)
    #X here is an 3D array

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # build network topology
    #using L2 regularization & Dropout to reduce overfitting
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),  #

        # 1st dense layer
        keras.layers.Dense(1024, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.30),

        # 2nd dense layer
        keras.layers.Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.25),

        # 3rd dense layer
        keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # output layer
        keras.layers.Dense(9, activation='softmax')

    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0008)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    print("shape of layer is{} {} ".format(X.shape[1], X.shape[2]))

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=5)
    #display plot of error and accuracy
    plot_history(history)


    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("accuracy on test set:{}".format(test_accuracy))

    y_predict = []
    # make predictions
    for i in X_test:
        y_predict.append(predict(model, i))
    print("done")

    conf_matrix = confusion_matrix(y_test, y_predict, normalize='true')
    labels = ["cel", "cla", "gac", "gel", "org", "pia", "sax", "tru", "vio"]
    a = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.1)
    sn.heatmap(a, annot=True, cmap="BuPu")
    plt.show()

    # roc_curve,recall_score,f1_score,precision_score,log_loss
    ab = []
    ab.append(f1_score(y_test, y_predict, average='weighted'))
    ab.append(f1_score(y_test, y_predict, average='micro'))
    ab.append(f1_score(y_test, y_predict, average='macro'))
    ab.append(recall_score(y_test, y_predict, average='weighted'))
    ab.append(recall_score(y_test, y_predict, average='micro'))
    ab.append(recall_score(y_test, y_predict, average='macro'))
    ab.append(precision_score(y_test, y_predict, average='weighted'))
    ab.append(precision_score(y_test, y_predict, average='micro'))
    ab.append(precision_score(y_test, y_predict, average='macro'))

    print(ab)
    y_score = []
    for i in X_test:
        y_score.append(predict_2(model, i).tolist())
    print("...")
    roc = []
    roc.append(roc_auc_score(y_test, y_score, average='macro', multi_class='ovo'))
    roc.append(roc_auc_score(y_test, y_score, average='weighted', multi_class='ovr'))
    roc.append(roc_auc_score(y_test, y_score, average='weighted', multi_class='ovo'))
    roc.append(roc_auc_score(y_test, y_score, average='macro', multi_class='ovr'))
    print(roc)
