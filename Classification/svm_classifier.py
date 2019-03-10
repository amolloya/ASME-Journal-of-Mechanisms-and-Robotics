from import_data_and_preprocessing import X,Y
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

print('\nGait Classification using Spport Vector Machines classifier:\n')

# Train-test split
x,x_t,y,y_t = train_test_split(X, Y, test_size = 0.2, random_state=1)

# Grid-search over these parameters
grid_param = {'gamma': ['scale',0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,500,1000],
              'C': [0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,500,1000],
              'kernel': ['linear', 'rbf', 'poly'],
              'decision_function_shape' : ['ovo', 'ovr']}

print('Searching for the best parameters...')

# Define classifier
classifier = svm.SVC(random_state=0)

# Grid search wrapper
gd_sr = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='accuracy', cv=10, verbose=3, n_jobs=-1)

# Fitting the data to the function
grid_result = gd_sr.fit(x, y)
  
# Mean and std of the CV results for each parameters set
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# Best parameters on the CV set
best_parameters = grid_result.best_params_  
print('\nBest parameter selection by grid-search:\n', best_parameters)

# Accuracy of best parameter on CV set
best_result = grid_result.best_score_
print('\nCross-validation accuracy obtained by best parameters: ', round(best_result,4)) 

# Predicting the classes on test set with the best-parameter model
pred = grid_result.predict(x_t)
res = grid_result.score(x_t,y_t)
print('\nClassification accuracy on test set: ', round(res,4))
print('')

print(classification_report(y_t, pred))

# Saving the model
filename = 'svm_classifier.sav'
pickle.dump(grid_result, open(filename, 'wb'))
