from sklearn.ensemble import RandomForestClassifier
from data import X,Y
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneOut, KFold

#train_X,valid_X,train_label,valid_label = train_test_split(X, Y, test_size = 0.15)

# Acc of 0.8028 with 7 PCs
loo = LeaveOneOut()
acc1 = 0
square_error_sum = 0.0
count = 0
for train_index, test_index in loo.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    classifier = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=1)  
    
    classifier = classifier.fit(X_train, y_train)
    predicted_y = classifier.predict(X_test)
#    square_error_sum += float(y_test[0] - predicted_y) ** 2
    acc = classifier.score(X_test,y_test)
    acc1 = acc + acc1
    count += 1

acc1 = acc1/X.shape[0]
print('Accuracy of the model with LOO:', acc1)
print('\n')

# Acc = 83.09 for n_est = 110, dep = 11 and 7-fold CV, PCA = 8
for k in (range(2,8)):
    kfold = KFold(k, True, 1)
    
    acc1 = 0
    square_error_sum = 0.0
    count = 0
    for train_index, test_index in kfold.split(X):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        classifier = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=1)
    
        classifier = classifier.fit(X_train, y_train)
        predicted_y = classifier.predict(X_test)
#    square_error_sum += float(y_test[0] - predicted_y) ** 2
        acc = classifier.score(X_test,y_test)
#        print(acc)
        acc1 = acc + acc1
        count += 1
    
    acc1 = acc1/k
    print(k,'-fold CV Accuracy of the model:', acc1)