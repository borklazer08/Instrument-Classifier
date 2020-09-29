import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sn

from sklearn.metrics import confusion_matrix,roc_auc_score,recall_score,f1_score,precision_score



DATA_PATH = "C:\\Users\\Kartikeya\\PycharmProjects\\Industrial_Project_VII\\processing_with_YT.json"

#visualiser nneural nnetwork

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


def dataset_generator(test_set,validation_set):

   X,y=load_data(DATA_PATH)
   X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_set)
   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_set)

   #3d array required for CNN
   X_train=X_train[...,np.newaxis]
   X_val =X_val[..., np.newaxis]
   X_test=X_test[...,np.newaxis]
   #4D array now [num_samples,87,15,1]

   return X_train,X_val,X_test,y_train,y_val,y_test



def build_model(input_shape):
    #create model
    #1
    model=keras.Sequential()

    model.add(keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=input_shape))   # no. of kernals, kernel size,act,input_shape
    model.add(keras.layers.MaxPool2D((3,3), strides=(1,1), padding='same'))
    model.add(keras.layers.BatchNormalization())     #standardises and normalises activation in layer and subsequent layer

     #2
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))   # no. of kernals, kernel size,act,input_shape
    model.add(keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same'))
    model.add(keras.layers.BatchNormalization())   # standardises and normalises activation in layer and subsequent layer


   #3
    model.add(keras.layers.Conv2D(128, (2, 2), activation='relu',input_shape=input_shape))   # no. of kernals, kernel size,act,input_shape
    model.add(keras.layers.MaxPool2D((2, 2), strides=(1, 1), padding='same'))
    model.add(keras.layers.BatchNormalization())      # standardises and normalises activation in layer and subsequent layer

    #flatten output and feed to dense
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.25))

    # 2nd dense layer
    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.25))

    # 3rd dense layer
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.26))

    model.add(keras.layers.Dense(9,activation='softmax'))

    return model


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




if __name__=='__main__':

    #create train,validation and test sets
    X_train,X_validation,X_test,y_train,y_validation,y_test=  dataset_generator(0.20,0.17)

    #CNN net
    input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])

    model=build_model(input_shape)


    #compile
    optimisez=keras.optimizers.Adam(learning_rate=0.000122)
    model.compile(optimizer=optimisez,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()


    #train
    history=model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=256, epochs=14)

    #stats
    plot_history(history)

    #evaluate
    test_error,test_accuracy=model.evaluate(X_test,y_test,verbose=1)
    print("accuracy on test set:{}".format(test_accuracy))



    y_predict=[]
   #make predictions

    #ADDITIONAL METRICS
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





