from import_data_and_preprocessing import X,Y
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from numpy.random import seed
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import optimizers

seed(100)

Y = to_categorical(Y)

# Creating a network function for ANN
def create_network():
    
    model = Sequential()
    
    model.add(Dense(16, activation='relu'))

    model.add(Dense(32, activation='relu'))
    
    model.add(Dense(16, activation='relu'))
    
    model.add(Dropout(0.35))

    model.add(Dense(4, activation='softmax'))
    
    adam = optimizers.Adam(lr=0.0009)
    
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=adam)
    
    return model

# Using KerasClassifier wrapper to wrap the create_network function and hyperparameters
neural_network = KerasClassifier(build_fn=create_network, epochs=500, batch_size=20, verbose=1)

# Cross validation accuracy for k-folds
for i in range(2,11):
    acc = cross_val_score(neural_network, X, Y, cv=i)
    #print('Accuracy of k-folds: ', acc)
    acc1 = sum(acc)/i
    print('\nAverage accuracy for k-fold: ', acc1)
