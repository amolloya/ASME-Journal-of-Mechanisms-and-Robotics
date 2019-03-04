from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from numpy.random import seed
from data import X,Y
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import optimizers

#seed(100)

Y = to_categorical(Y)

#train_X,valid_X,train_label,valid_label = train_test_split(X, Y, test_size=0.2)
#
#batch_size = 10
#epochs = 100
#num_classes = 4
##def model(X,Y):
#model = Sequential()
#model.add(Dense(16, activation='relu'))
###model.add(Dropout(0.3))
#model.add(Dense(32, activation='relu'))
###model.add(Dropout(0.3))
#model.add(Dense(16, activation='relu'))
###model.add(Dropout(0.3))
##model.add(Dense(128, activation='linear'))
##model.add(Dropout(0.3))
##model.add(Dense(32, activation='linear'))
##model.add(Dropout(0.1))
#model.add(Dense(num_classes, activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#model_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data = (valid_X, valid_label))
##print(model.summary())
#predicted_labels = model.predict(valid_X)
#
## summarize history for accuracy
#fig = plt.figure()
#plt.plot(model_train.history['acc'])
#plt.plot(model_train.history['val_acc'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
## summarize history for loss
#fig1 = plt.figure()
#plt.plot(model_train.history['loss'])
#plt.plot(model_train.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

#def prediction(X):
#    prediction = model.predict(X)
#    return prediction

#x1 = pca.fit_transform(x1)
#res = prediction(x1)
#print(res)

def create_network():
    
    model = Sequential()
    
    model.add(Dense(16, activation='relu'))

    model.add(Dense(32, activation='relu'))
    
    model.add(Dense(16, activation='relu'))
    
    model.add(Dropout(0.35))

    model.add(Dense(4, activation='softmax'))
    
    adam = optimizers.Adam(lr=0.0009)
    
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=adam)
    
    # Return compiled network
    return model

neural_network = KerasClassifier(build_fn=create_network, epochs=500, batch_size=20, verbose=1)

acc = cross_val_score(neural_network, X, Y, cv=7)
print(acc)
acc1 = sum(acc)/7
print(acc1)

