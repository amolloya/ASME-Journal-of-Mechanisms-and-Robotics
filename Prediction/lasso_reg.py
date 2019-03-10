import numpy as np
from import_data_and_data_preprocessing import X,Y
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import pickle

print('\nGait Prediction using Lasso Regressor:\n')

# Train-test split
x,x_t,y,y_t = train_test_split(X, Y, test_size = 0.2, random_state=1)

# Grid-search over these parameters
grid_param = {'alpha': [0.0001,0.05,0.01,0.1,0.5,1,5,10,100]}

# Define regressor
regressor = Lasso(random_state=0)

#print(regressor.get_params().keys())

print('Searching for the best parameters...\n')

# Grid search wrapper with 10-fold CV for finding the best parameters
lasso_grid = GridSearchCV(estimator=regressor, param_grid=grid_param, cv=10, 
                         scoring= 'neg_mean_squared_error', verbose=3, n_jobs=-1)

# Fitting the data to the function
grid_result = lasso_grid.fit(x, y)

# Mean and std of the CV results for each parameters set
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
means = np.sqrt(-means)
stds = np.sqrt(stds)

print('\nRMS error for different parameter set:\n')
for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# Best parameters on the CV set
best_parameters = grid_result.best_params_
print('\nBest parameter selection by grid-search:\n', best_parameters)

# RMSE of best parameter on CV set
best_result = grid_result.best_score_
print('\nCross-validation RMS error obtained by best parameters: ', round(np.sqrt(-best_result),4)) 

# Predicting the trajectories and RMSE on test set with the best-parameter model
pred = grid_result.predict(x_t)
res = grid_result.score(x_t,y_t)
print('\nPrediction RMS error on test set: ', round(np.sqrt(-res),4))
print('')

# Saving the model
filename = 'lasso_regression.sav'
pickle.dump(grid_result, open(filename, 'wb'))

f = [[] for i in range(len(y_t))]

# Plot for actual vs predicted hip joint trajectory
for i in range(0, len(y_t)):
    f[i] = plt.figure()
    plt.plot(pred[i,:77], color = 'blue', label = 'Predicted trajectory')
    plt.plot(y_t[i,:77], color = 'red', label = 'Actual trajectory')
    plt.title('Prediction of Right Hip extension')
    plt.legend()
    plt.show()

# Plot for actual vs predicted knee joint trajectory    
for i in range(0, len(y_t)):
    f[i] = plt.figure()
    plt.plot(pred[i,77:154], color = 'blue', label = 'Predicted trajectory')
    plt.plot(y_t[i,77:154], color = 'red', label = 'Actual trajectory')
    plt.title('Prediction of Right Knee flexion')
    plt.legend()
    plt.show()

# Plot for actual vs predicted ankle joint trajectory
for i in range(0, len(y_t)):
    f[i] = plt.figure()
    plt.plot(pred[i,154:], color = 'blue', label = 'Predicted trajectory')
    plt.plot(y_t[i,154:], color = 'red', label = 'Actual trajectory')
    plt.title('Prediction of Right Ankle plantar flexion')
    plt.legend()
    plt.show()
    
