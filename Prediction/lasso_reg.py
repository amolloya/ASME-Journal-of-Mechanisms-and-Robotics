from sklearn import linear_model
from import_data_and_preprocessing import X, Y
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut
import pickle

Y1 = Y[:,:,5]
Y2 = Y[:,:,7]
Y3 = Y[:,:,8]
Y = np.row_stack((Y1,Y2,Y3))
Y = Y.T

from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=0)

# Leave one out Cross-validation
loo = LeaveOneOut()
err = 0

# 
for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # 
        clf = linear_model.Lasso(alpha=5.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        mean_sq_error = mean_squared_error(y_test, y_pred)
        rms = np.sqrt(mean_sq_error)
        err = err + rms

# Mean error
err = err/X.shape[0]
print('RMSE of the model wiht Loo:', err)

# Save the model
filename = 'lasso_reg_loo.sav'
pickle.dump(clf, open(filename, 'wb'))

# K-fold Cross-validation
for k in (range(2,8)):
    kfold = KFold(k, True, 1)   
    err = 0

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        clf = linear_model.Lasso(alpha=5.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        mean_sq_error = mean_squared_error(y_test, y_pred)        
        rms = np.sqrt(mean_sq_error)        
        err = err + rms

    # Mean error
    err = err/k
    print(k,'-fold CV RMSE of the model:', err)

filename = 'lasso_reg_kfold.sav'
pickle.dump(clf, open(filename, 'wb'))

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
