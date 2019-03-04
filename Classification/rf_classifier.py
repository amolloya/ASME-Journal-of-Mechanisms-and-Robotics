from import_data_and_preprocessing import X,Y
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, KFold

# Cross-validation with Leave one out
loo = LeaveOneOut()
acc1 = 0

# Splitting the data into k-folds ((k-1)-fold for training and 1-fold for testing)
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # Fitting the Random Forest Classifier to the data
    classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)  
    
    classifier = classifier.fit(X_train, y_train)
    predicted_y = classifier.predict(X_test)
    acc = classifier.score(X_test,y_test)
    acc1 = acc + acc1

# Mean Accuracy
acc1 = acc1/X.shape[0]
print('Accuracy of the model with LOO:', acc1)
print('\n')

# Cross-validation with k-fold
for k in (range(2,8)):
    kfold = KFold(k, True, 1)
    acc1 = 0
    
    # Splitting the data into k-folds ((k-1)-fold for training and 1-fold for testing)
    for train_index, test_index in kfold.split(X):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # Fitting the Random Forest Classifier to the data
        classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
    
        classifier = classifier.fit(X_train, y_train)
        predicted_y = classifier.predict(X_test)
        acc = classifier.score(X_test,y_test)
        acc1 = acc + acc1
    
    # Mean Accuracy
    acc1 = acc1/k
    print(k,'-fold CV Accuracy of the model:', acc1)
