from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from data_preprocessing import x, Y
from sklearn.model_selection import train_test_split
from numpy.random import seed
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut
import pickle

#seed(100)

#X = X.reshape((1155,108))
#X = X.T
Y1 = Y[:,:,5]
Y2 = Y[:,:,7]
Y3 = Y[:,:,8]
Y = np.row_stack((Y1,Y2,Y3))
Y = Y.T

from sklearn.utils import shuffle
X, Y = shuffle(x, Y, random_state=0)

#train_X,valid_X,train_label,valid_label = train_test_split(x, Y, test_size = 0.2)

#poly = PolynomialFeatures(degree=3)
#train_X_ = poly.fit_transform(train_X)
#valid_X_ = poly.fit_transform(valid_X)
#
#clf = linear_model.LinearRegression()
#clf.fit(train_X_, train_label)
#y_pred = clf.predict(valid_X_)
##print(pred)
#
#mean_sq_error = mean_squared_error(valid_label, y_pred)
##print('Mean squared error:', mean_sq_error)
#rms_error = np.sqrt(mean_sq_error)
#print('Root mean squared error:', rms_error)
#

loo = LeaveOneOut()
err = 0
square_error_sum = 0.0
count = 0

for train_index, test_index in loo.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
        
    poly = PolynomialFeatures(degree=3)
    train_X_ = poly.fit_transform(X_train)
    valid_X_ = poly.fit_transform(X_test)

    clf = linear_model.LinearRegression()
    clf.fit(train_X_, y_train)
    y_pred = clf.predict(valid_X_)
        
    mean_sq_error = mean_squared_error(y_test, y_pred)
        
    rms = np.sqrt(mean_sq_error)
        
    err = err + rms
#        count += 1
 
err = err/X.shape[0]
print('RMSE of the model wiht Loo:', err)

filename = 'poly_reg_loo.sav'
pickle.dump(clf, open(filename, 'wb'))


for k in (range(2,8)):
    kfold = KFold(k, True, 1)
    
    err = 0
    square_error_sum = 0
    count = 0
    for train_index, test_index in kfold.split(X):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        poly = PolynomialFeatures(degree=3)
        train_X_ = poly.fit_transform(X_train)
        valid_X_ = poly.fit_transform(X_test)

        clf = linear_model.LinearRegression()
        clf.fit(train_X_, y_train)
        y_pred = clf.predict(valid_X_)
        
        mean_sq_error = mean_squared_error(y_test, y_pred)
        
        rms = np.sqrt(mean_sq_error)
        
        err = err + rms
        count += 1
    
    err = err/k
    print(k,'-fold CV RMSE of the model:', err)

#filename = 'poly_reg_kfold.sav'
#pickle.dump(clf, open(filename, 'wb'))

np.save('y_pred_poly.npy', y_pred)
np.save('valid_label_poly.npy', valid_label)

#f = [[] for i in range(len(valid_label))]
#
#for i in range(0, len(valid_label)):
#    f[i] = plt.figure()
#    plt.plot(pred[i,:], color = 'blue', label = 'Predicted trajectory')
#    plt.plot(valid_label[i,:], color = 'red', label = 'Actual trajectory')
##    plt.plot(Y.mean(0), color = 'black', label = 'Average trajectory of the entire set')
#    #plt.plot(y_pred[0,:] + std, color = 'yellow', label = 'Standard deviation of the trajectories' )
#    #plt.plot(y_pred[0,:] - std, color = 'yellow')
#    plt.title('Prediction of Right Ankle plantar flexion')
#    plt.legend()
#    plt.show()