from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from import_data_and_preprocessing import X, Y
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.utils import shuffle
import pickle

# Slicing the hip, knee and ankle joint trajectories
Y1 = Y[:,:,5]
Y2 = Y[:,:,7]
Y3 = Y[:,:,8]
Y = np.row_stack((Y1,Y2,Y3))
Y = Y.T

# Shuffling the data randomly
X, Y = shuffle(X, Y, random_state=0)

# Cross-validation with Leave one out
loo = LeaveOneOut()
err = 0

# Splitting the data into k-folds ((k-1)-fold for training and 1-fold for testing)
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # Making the features into polynomial features of degree 3
    poly = PolynomialFeatures(degree=3)
    train_X_ = poly.fit_transform(X_train)
    valid_X_ = poly.fit_transform(X_test)
    
    # Fitting the Polynomial regression model to the data
    clf = linear_model.LinearRegression()
    clf.fit(train_X_, y_train)
    y_pred = clf.predict(valid_X_)
        
    mean_sq_error = mean_squared_error(y_test, y_pred) 
    rms = np.sqrt(mean_sq_error)
    err = err + rms

# Mean RMSE
err = err/X.shape[0]
print('RMSE of the model wiht Loo:', err)

# Saving the model
filename = 'poly_reg_loo.sav'
pickle.dump(clf, open(filename, 'wb'))

# Cross-validation with k-fold
for k in (range(2,8)):
    kfold = KFold(k, True, 1)
    err = 0
    
    # Splitting the data into k-folds ((k-1)-fold for training and 1-fold for testing)
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # Making the features into polynomial features of degree 3
        poly = PolynomialFeatures(degree=3)
        train_X_ = poly.fit_transform(X_train)
        valid_X_ = poly.fit_transform(X_test)
        
        # Fitting the Polynomial regression model to the data
        clf = linear_model.LinearRegression()
        clf.fit(train_X_, y_train)
        y_pred = clf.predict(valid_X_)
        
        mean_sq_error = mean_squared_error(y_test, y_pred)
        rms = np.sqrt(mean_sq_error)
        err = err + rms
    
    # Mean RMSE
    err = err/k
    print(k,'-fold CV RMSE of the model:', err)

# Saving the model
filename = 'poly_reg_kfold.sav'
pickle.dump(clf, open(filename, 'wb'))

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
