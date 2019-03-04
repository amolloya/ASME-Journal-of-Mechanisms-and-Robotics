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

train_X,valid_X,train_label,valid_label = train_test_split(X, Y, test_size = 0.2)

clf = linear_model.Lasso(alpha=0.42)
clf.fit(train_X, train_label)
y_pred = clf.predict(valid_X)
#print(pred)

mean_sq_error = mean_squared_error(valid_label, y_pred)
#print('Mean squared error:', mean_sq_error)
rms_error = np.sqrt(mean_sq_error)
print('Root mean squared error:', rms_error)

#loo = LeaveOneOut()
#err = 0
#square_error_sum = 0.0
#count = 0
#
#for train_index, test_index in loo.split(X):
#    
#        X_train, X_test = X[train_index], X[test_index]
#        y_train, y_test = Y[train_index], Y[test_index]
#        
#        clf = linear_model.Lasso(alpha=5.0)
#        clf.fit(X_train, y_train)
#        y_pred = clf.predict(X_test)
#        
#        mean_sq_error = mean_squared_error(y_test, y_pred)
#        
#        rms = np.sqrt(mean_sq_error)
#        
#        err = err + rms
##        count += 1
# 
#err = err/X.shape[0]
#print('RMSE of the model wiht Loo:', err)
#
#filename = 'lasso_reg_loo.sav'
#pickle.dump(clf, open(filename, 'wb'))
#
#
#for k in (range(2,8)):
#    kfold = KFold(k, True, 1)
#    
#    err = 0
#    square_error_sum = 0
#    count = 0
#    for train_index, test_index in kfold.split(X):
#    
#        X_train, X_test = X[train_index], X[test_index]
#        y_train, y_test = Y[train_index], Y[test_index]
#        
#        clf = linear_model.Lasso(alpha=5.0)
#        clf.fit(X_train, y_train)
#        y_pred = clf.predict(X_test)
#        
#        mean_sq_error = mean_squared_error(y_test, y_pred)
#        
#        rms = np.sqrt(mean_sq_error)
#        
#        err = err + rms
#        count += 1
#    
#    err = err/k
#    print(k,'-fold CV RMSE of the model:', err)
#
##filename = 'lasso_reg_kfold.sav'
##pickle.dump(clf, open(filename, 'wb'))

np.save('y_pred_lasso.npy', y_pred)
np.save('valid_label_lasso.npy', valid_label)

#f = [[] for i in range(len(y_test))]
#
#for i in range(0, len(y_test)):
#    f[i] = plt.figure()
#    plt.plot(y_pred[i,:], color = 'blue', label = 'Predicted trajectory')
#    plt.plot(y_test[i,:], color = 'red', label = 'Actual trajectory')
##    plt.plot(Y.mean(0), color = 'black', label = 'Average trajectory of the entire set')
#    #plt.plot(y_pred[0,:] + std, color = 'yellow', label = 'Standard deviation of the trajectories' )
#    #plt.plot(y_pred[0,:] - std, color = 'yellow')
#    plt.title('Prediction of Right Ankle plantar flexion')
#    plt.legend()
#    plt.show()