from data_preprocessing import x, Y
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy.random import seed
import numpy as np
#from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

#X = X.reshape((1155,108))
#X = X.T
#pca = PCA(n_components= 14)
#X = pca.fit_transform(x)
#var = pca.explained_variance_ratio_
#tot_var = sum(var)
#print('Total variance covered by PCs', tot_var)
#print('\n')

Y1 = Y[:,:,5]
Y2 = Y[:,:,7]
Y3 = Y[:,:,8]
Y = np.row_stack((Y1,Y2,Y3))
Y = Y.T

X, Y = shuffle(x, Y, random_state=1)
#
train_X,valid_X,train_label,valid_label = train_test_split(x, Y, test_size = 0.2)
#
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(16, activation = 'relu', input_dim = 14))

# Adding the second hidden layer
model.add(Dense(units = 16, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 16, activation = 'relu'))

model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 231, activation = 'linear'))

model.compile(optimizer = 'adam',loss = 'mse')

model_train = model.fit(train_X, train_label, batch_size = 10, epochs = 100, validation_data = (valid_X, valid_label))

y_pred = model.predict(valid_X)

mean_sq_error = mean_squared_error(valid_label, y_pred)
#print('Mean squared errRoot mean squared erroror:', mean_sq_error)
rms_error = np.sqrt(mean_sq_error)
print('Root Mean Square error:', rms_error)

# summarize history for loss
fig1 = plt.figure()
plt.plot(model_train.history['loss'])
plt.plot(model_train.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


#def create_network():
#    
#    model = Sequential()
#    
#    model.add(Dense(16, activation='relu', input_dim = 14))
#
#    model.add(Dense(16, activation='relu'))
#    
#    model.add(Dense(16, activation='relu'))
#    
##    model.add(Dropout(0.35))
#
#    model.add(Dense(231, activation='softmax'))
#    
#    model.compile(loss='root_mean_squared_error',metrics=['accuracy'], optimizer='rmsprop')
#    
#    # Return compiled network
#    return model
#
#neural_network = KerasClassifier(build_fn=create_network, epochs=100, batch_size=20, verbose=1)
#
#acc = cross_val_score(neural_network, X, Y, cv=3)
#print(acc)
#acc1 = sum(acc)/3
#print(acc1)

#dist = np.linalg.norm(valid_label-y_pred)   # euc dist
#print(dist)

#std = np.std(Y, axis = 0)
#std = std.T

np.save('y_pred_dnn.npy', y_pred)
np.save('valid_label_dnn.npy', valid_label)

#z1 = y_pred[0,:77]
#z2 = y_pred[0,77:154]
#z3 = y_pred[0,154:]
#z = np.column_stack((z1,z2,z3))

#plt.plot(y_pred[0,:77], color = 'blue', label = 'Hip angle trajectory')
#plt.plot(y_pred[0,77:154], color = 'red', label = 'Knee angle trajectory')
#plt.plot(y_pred[0,154:], color = 'yellow', label = 'Ankle angle trajectory')
#plt.title('Prediction of trajectories for a person')
#plt.legend()
#plt.show()

#f=np.load('angle_traj.npy')

#f,z = [[] for i in range(len(valid_label))], [[] for i in range(len(valid_label))]

#for i in range(0, len(valid_label)):
#    f[i] = plt.figure()
#    plt.plot(y_pred[i,:77], color = 'blue', label = 'Predicted trajectory')
#    plt.plot(valid_label[i,:77], color = 'red', label = 'Actual trajectory')
##    plt.plot(Y.mean(0), color = 'black', label = 'Average trajectory of the entire set')
#    #plt.plot(y_pred[0,:] + std, color = 'yellow', label = 'Standard deviation of the trajectories' )
#    #plt.plot(y_pred[0,:] - std, color = 'yellow')
#    plt.title('Prediction of Right Hip extension')
#    plt.legend()
#    plt.show()
#
#for i in range(0, len(valid_label)):
#    f[i] = plt.figure()
#    plt.plot(y_pred[i,77:154], color = 'blue', label = 'Predicted trajectory')
#    plt.plot(valid_label[i,77:154], color = 'red', label = 'Actual trajectory')
##    plt.plot(Y.mean(0), color = 'black', label = 'Average trajectory of the entire set')
#    #plt.plot(y_pred[0,:] + std, color = 'yellow', label = 'Standard deviation of the trajectories' )
#    #plt.plot(y_pred[0,:] - std, color = 'yellow')
#    plt.title('Prediction of Right Knee flexion')
#    plt.legend()
#    plt.show()
#    
#for i in range(0, len(valid_label)):
#    f[i] = plt.figure()
#    plt.plot(y_pred[i,154:], color = 'blue', label = 'Predicted trajectory')
#    plt.plot(valid_label[i,154:], color = 'red', label = 'Actual trajectory')
##    plt.plot(Y.mean(0), color = 'black', label = 'Average trajectory of the entire set')
#    #plt.plot(y_pred[0,:] + std, color = 'yellow', label = 'Standard deviation of the trajectories' )
#    #plt.plot(y_pred[0,:] - std, color = 'yellow')
#    plt.title('Prediction of Right Ankle plantar flexion')
#    plt.legend()
#    plt.show()