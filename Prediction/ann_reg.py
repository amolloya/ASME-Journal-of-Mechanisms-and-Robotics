import numpy as np
from import_data_and_preprocessing import X,Y
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import pickle

print('\nGait Prediction using Aritifical Neural Network Regressor:\n')

# Train-test split
x,x_t,y,y_t = train_test_split(X, Y, test_size = 0.2, random_state=1)

# Grid-search over the following parameters
max_iter = [250,500,750]
activation = ['logistic', 'tanh', 'relu']
batch_size = [5,10,20]
solver = ['lbfgs', 'sgd', 'adam']
learning_rate = ['constant', 'invscaling', 'adaptive']
hidden_layer_sizes = [(100,),(200,),(300,)]
     
grid_param = dict(max_iter=max_iter, activation=activation, batch_size=batch_size, solver=solver, learning_rate=learning_rate, hidden_layer_sizes=hidden_layer_sizes)

# Define regressor
regressor = MLPRegressor(random_state=0)

#print(regressor.get_params().keys())

print('Searching for the best parameters...\n')
# Grid search wrapper with 10-fold CV for finding the best parameters
dnn_grid = GridSearchCV(estimator=regressor, param_grid=grid_param, cv=10, 
                         scoring= 'neg_mean_squared_error', verbose=3, n_jobs=-1)

# Fitting the train data to the function
grid_result = dnn_grid.fit(x, y)

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

# Saving the file
filename = 'ann_regression.sav'
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
