from import_data_and_preprocessing import X,Y
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle

print('\nGait Classification using K-nearest neighbors classifier:\n')

# Train-test split
x,x_t,y,y_t = train_test_split(X, Y, test_size = 0.2, random_state=1)

# Grid-search over the following parameters
grid_param = {'n_neighbors': [2,3,4,5,6,7,8,9,10],
              'metric': ['euclidean', 'manhattan', 'minkowski','chebyshev'],
              'weights': ['uniform', 'distance']}

print('Searching for the best parameters...')

# Define classifier
classifier = KNeighborsClassifier() 

# Grid search wrapper with 10-fold CV for finding the best parameters
gd_sr = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='accuracy', cv=10, verbose=3, n_jobs=-1)

# Fitting the train data to the function
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

# Results
conf_matrix = confusion_matrix(y_t, pred) 
print('Confusion Matrix :\n', conf_matrix) 
print('\nClassification Report : \n', classification_report(y_t, pred)) 

# Heatmap of confusion matrix
df_cm = pd.DataFrame(conf_matrix, index = ['CA', 'HSP', 'PD', 'HC'],
                  columns = ['CA', 'HSP', 'PD', 'HC'])
ax = plt.axes()
sns.heatmap(df_cm, cmap='BuPu', linewidths = 2, square=False, annot=True)
ax.set_title('Confusion Matrix for KNN classifier', fontsize=15)
ax.set_xlabel('Target labels', fontsize=14)
ax.set_ylabel('Predicted labels', fontsize=14)

# Saving the model
filename = 'knn_classifier.sav'
pickle.dump(grid_result, open(filename, 'wb'))
