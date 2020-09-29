import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "C:\\Users\\Kartikeya\\PycharmProjects\\Industrial_Project_VII\\precessing_modified.json"




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

     # no. of kernals, kernel size,act,input_shape
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

    model.add(keras.layers.BatchNormalization()) #standardises and normalises activation in layer and subsequent layer
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))

     #2
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) # no. of kernals, kernel size,act,input_shape
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())  # standardises and normalises activation in layer and subsequent layer


   #3
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu',input_shape=input_shape)) # no. of kernals, kernel size,act,input_shape
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())  # standardises and normalises activation in layer and subsequent layer

    #flatten output and feed to dense
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(11,activation='sigmoid'))

    return model

def predict(model,X,y):
    X=X[np.newaxis,...]
    prediction = model.predict(X)   #x = (1,dims)
    #its 2D [[0.1,0.3,.....]] result of softmax
    #extract max
    p1_index=np.argmax(prediction,axis = 1)


    print("Expected index:{}, Predicted indexes:{}, final matrix:{} ".format(y,p1_index,prediction))









if __name__=='__main__':
    #create train,validation and test sets
    X_train,X_validation,X_test,y_train,y_validation,y_test=  dataset_generator(0.25,0.22)

    #CNN net
    input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])

    model=build_model(input_shape)


    #compile
    optimisez=keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimisez,loss="binary_crossentropy",metrics=['accuracy'])


    #train
    model.fit(X_train,y_train,validation_data=(X_validation,y_validation),batch_size=64,epochs=30)


    #evaluate
    test_error,test_accuracy=model.evaluate(X_test,y_test,verbose=1)
    print("accuracy on test set:{}".format(test_accuracy))


   #make predictions
    for i in range(10):
        X=X_test[100+i]
        y=y_test[100+1]
        predict(model,X,y)
