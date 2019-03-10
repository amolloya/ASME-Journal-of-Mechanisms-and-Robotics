from import_data_and_preprocessing import X,Y
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

print('\nGait Classification using Random Forests classifier:\n')

# Train-test split
x,x_t,y,y_t = train_test_split(X, Y, test_size = 0.2, random_state=1)

# Grid-search over these parameters
grid_param = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'max_depth': [5,6,7,8,9,10,11,12,13,14,15],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]}
#    'min_samples_split': [0.5,1.0,2,3],
#    'min_samples_leaf': [1,2,3]
    
print('Searching for the best parameters...\n')

# Define classifier
classifier = RandomForestClassifier(random_state=0)  

# Grid search wrapper with 10-fold CV for finding the best parameters
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

print(classification_report(y_t, pred))

# Saving the model
filename = 'rf_classifier.sav'
pickle.dump(grid_result, open(filename, 'wb'))
