from sklearn.datasets import fetch_mldata
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray
from skimage.io import imread

from keras.models import Sequential

from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from keras.models import model_from_json


def to_categorical(labels, n):
        retVal = np.zeros((len(labels), n), dtype='int')
        ll = np.array(list(enumerate(labels)))
        retVal[ll[:,0],ll[:,1]] = 1
        return retVal

def train():
    mnist = fetch_mldata('MNIST original')
    
    data   = mnist.data / 255.0
    labels = mnist.target.astype('int')
    
    train_rank = 5000
    
    #------- MNIST subset --------------------------
    train_subset = np.random.choice(data.shape[0], train_rank)
    # train dataset
    train_data = data[train_subset]
    train_labels = labels[train_subset]
    
    
    
    
    train_out = to_categorical(train_labels, 10)
    
    print train_out
    
    #--------------- ANN ------------------
    
    
    # prepare model
    global model 
    model= Sequential()
    model.add(Dense(70, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    
    
    # compile model with optimizer
    sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    
    
    # training
    training = model.fit(train_data, train_out, nb_epoch=500, batch_size=400, verbose=0)
    print training.history['loss'][-1]


    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
def readData():
    # load json and create model
    print "reading data"
    json_file = open('model.json', 'r')
    print "reading json..."
    loaded_model_json = json_file.read()
    print "closed"
    json_file.close()
    global loaded_model
    print "model from json"
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print "loading weights..."
    loaded_model.load_weights("model.h5")
    print "Loaded model from disk"



def prepoznaj(img):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    
    img_test = (img>120)
    img=img_test*img
    #plt.imshow(img, cmap="Greys")
    #plt.show()
    
    
    image = img.reshape(784)
    #print imgB_test
    
    image = image/255.
    #print imgB_test.shape
    #print imgB_test
    #print "predicting"
    tt = loaded_model.predict(np.array([image]), verbose=1)
    #print tt, "TT"
    
    #plt.xticks(x)
    #plt.bar(x, tt[0], color="blue")
    
    rez_t = tt.argmax(axis=1)
    #print rez_t
    return rez_t
