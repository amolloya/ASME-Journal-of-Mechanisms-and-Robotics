from sklearn.neighbors import KNeighborsClassifier  
from data import X,Y
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
from numpy.random import seed
from sklearn.model_selection import LeaveOneOut, KFold
import matplotlib.pyplot as plt
    
#Acuracy of 0.9545 with 11 PCs and seed 40 and test_size = 0.15

#seed(40)
#
#train_X,valid_X,train_label,valid_label = train_test_split(X, Y, test_size = 0.15)
#
#classifier = KNeighborsClassifier(n_neighbors=5)  
#classifier.fit(train_X, train_label)
#
#y_pred = classifier.predict(valid_X)
#
#acc = classifier.score(valid_X,valid_label)
#print('Accuracy of the model:', acc)
#
#print('\nConfusion Matrix: \n', confusion_matrix(valid_label, y_pred))  
#print('\nClassification Report: \n',classification_report(valid_label, y_pred))

# Leave One Out CV
# Accuracy of 0.7605 with 8 PCs

loo = LeaveOneOut()
acc1 = 0
square_error_sum = 0.0
count = 0
for train_index, test_index in loo.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    classifier = KNeighborsClassifier(n_neighbors=3)  
    
    classifier = classifier.fit(X_train, y_train)
    predicted_y = classifier.predict(X_test)
    square_error_sum += float(y_test[0] - predicted_y) ** 2
    acc = classifier.score(X_test,y_test)
    acc1 = acc + acc1
    count += 1
    
acc1 = acc1/X.shape[0]
print('Accuracy of the model with LOO:', acc1)
print('\n')
#    
#mse = square_error_sum / X.shape[0]
#print ('-----------------------')
#print ('Mean Squared error with leave one out Cross-validation:' , mse)
#print ('-----------------------')

# K-Fold CV
# Accuracy of 0.7751 with 8 PCs and 3-fold CV

for k in (range(2,8)):
    kfold = KFold(k, True, 1)
    
    acc1 = 0
    square_error_sum = 0.0
    count = 0
    for train_index, test_index in kfold.split(X):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        classifier = KNeighborsClassifier(n_neighbors=3)  
    
        classifier = classifier.fit(X_train, y_train)
        predicted_y = classifier.predict(X_test)
#    square_error_sum += float(y_test[0] - predicted_y) ** 2
        acc = classifier.score(X_test,y_test)
#        print(acc)
        acc1 = acc + acc1
        count += 1
        
    acc1 = acc1/k
    print(k,'-fold CV Accuracy of the model:', acc1)
    plt.plot(k, acc1, 'rx')
#    plt.axis('equal')
    plt.show()
    
#mse = square_error_sum / X.shape[0]
#print ('-----------------------')
#print ('Mean Squared error with leave one out Cross-validation:' , mse)
#print ('-----------------------')
