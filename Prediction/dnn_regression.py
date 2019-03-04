from data_preprocessing import X, Y
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Importing the hip, knee and ankle joint angle trajectories
Y1 = Y[:,:,5]
Y2 = Y[:,:,7]
Y3 = Y[:,:,8]
Y = np.row_stack((Y1,Y2,Y3))
Y = Y.T

# Shuffling the data
X, Y = shuffle(X, Y, random_state=0)

# Spiltting the data into train and test set
train_X,valid_X,train_label,valid_label = train_test_split(x, Y, test_size = 0.2)

# Creating a sequential model for the ANN
model = Sequential()

# First hidden layer
model.add(Dense(16, activation = 'relu', input_dim = 14))

# Second hidden layer
model.add(Dense(units = 16, activation = 'relu'))

# Third hidden layer
model.add(Dense(units = 16, activation = 'relu'))

# Dropout for regularization
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units = 231, activation = 'linear'))

# Compiling the model with 'adam' optomizer and 'mse' loss function
model.compile(optimizer = 'adam',loss = 'mse')

# Fitting the model to the data
model_train = model.fit(train_X, train_label, batch_size = 10, epochs = 100, validation_data = (valid_X, valid_label))

# Predicting the trajectories on unseen data from our trained model
y_pred = model.predict(valid_X)

# Root mean square error (RMSE)
mean_sq_error = mean_squared_error(valid_label, y_pred)
rms_error = np.sqrt(mean_sq_error)
print('Root Mean Square error:', rms_error)

# Model loss
fig1 = plt.figure()
plt.plot(model_train.history['loss'])
plt.plot(model_train.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

f = [[] for i in range(len(valid_label))]

# Plot for actual vs predicted hip joint trajectory
for i in range(0, len(valid_label)):
    f[i] = plt.figure()
    plt.plot(y_pred[i,:77], color = 'blue', label = 'Predicted trajectory')
    plt.plot(valid_label[i,:77], color = 'red', label = 'Actual trajectory')
    plt.title('Prediction of Right Hip extension')
    plt.legend()
    plt.show()

# Plot for actual vs predicted knee joint trajectory    
for i in range(0, len(valid_label)):
    f[i] = plt.figure()
    plt.plot(y_pred[i,77:154], color = 'blue', label = 'Predicted trajectory')
    plt.plot(valid_label[i,77:154], color = 'red', label = 'Actual trajectory')
    plt.title('Prediction of Right Knee flexion')
    plt.legend()
    plt.show()

# Plot for actual vs predicted ankle joint trajectory
for i in range(0, len(valid_label)):
    f[i] = plt.figure()
    plt.plot(y_pred[i,154:], color = 'blue', label = 'Predicted trajectory')
    plt.plot(valid_label[i,154:], color = 'red', label = 'Actual trajectory')
    plt.title('Prediction of Right Ankle plantar flexion')
    plt.legend()
    plt.show()
