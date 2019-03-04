from import_data_and_preprocessing import X,Y
from sklearn import svm
from sklearn.model_selection import LeaveOneOut, KFold

# Leave one out Cross-validation
loo = LeaveOneOut()
acc1 = 0

# Splitting the data into k-folds ((k-1)-fold for training and 1-fold for testing)
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # Fitting the Support Vector Machine classsifier to the data
    classifier = svm.SVC(gamma= 10 , C = 7, decision_function_shape='ovo', kernel = 'rbf') 
    
    classifier = classifier.fit(X_train, y_train)
    predicted_y = classifier.predict(X_test)
    acc = classifier.score(X_test,y_test)
    acc1 = acc + acc1

# Mean Accuracy
acc1 = acc1/X.shape[0]
print('Accuracy of the model with LOO:', acc1)
print('\n')

# K-Fold Cross-validation
for k in (range(2,8)):
    kfold = KFold(k, True, 1)
    acc1 = 0
    
    # Splitting the data into k-folds ((k-1)-fold for training and 1-fold for testing)
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # Fitting the Support Vector Machine classifier to the data
        classifier = svm.SVC(gamma= 10 , C = 7, decision_function_shape='ovo', kernel = 'rbf') 
    
        classifier = classifier.fit(X_train, y_train)
        predicted_y = classifier.predict(X_test)
        acc = classifier.score(X_test,y_test)
        acc1 = acc + acc1      
    
    # Mean Accuracy
    acc1 = acc1/k
    print(k,'-fold CV Accuracy of the model:', acc1)
