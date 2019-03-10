import numpy as np
from data_preprocessing import X,Y
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

print('\nGait Prediction using Polynomial Regressor:\n')

# Train-test split
x,x_t,y,y_t = train_test_split(X, Y, test_size = 0.2, random_state=1)

# Grid-search over following parameters
grid_param = {'polynomialfeatures__degree': [2,3,4,5,6]}
              
print('Searching for the best parameters...\n')

# Define regressor
regressor = make_pipeline(PolynomialFeatures(), LinearRegression())

#print(regressor.get_params().keys())

# Grid search wrapper
poly_grid = GridSearchCV(estimator=regressor, param_grid=grid_param, cv=10, 
                         scoring='neg_mean_squared_error', verbose=3, n_jobs=-1)

# Fitting the train data to the function
grid_result = poly_grid.fit(x, y)

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

print(classification_report(y_t, pred))

# Saving the file
filename = 'poly_regression.sav'
pickle.dump(grid_result, open(filename, 'wb'))

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
