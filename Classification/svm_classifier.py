from data import X,Y
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy.random import seed
from mlxtend.plotting import plot_decision_regions
import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold

#seed(100)

def prediction(X):
    prediction = classifier.predict(X)
    return prediction

# Accuracy of the model 0.9545 with gamma = 6, C = 1 and test_size = 0.15
    
#train_X,valid_X,train_label,valid_label = train_test_split(X, Y, test_size = 0.15)

#classifier = svm.SVC(gamma= 6 , C = 1, decision_function_shape='ovo', kernel = 'rbf')
#model_train = classifier.fit(train_X, train_label)
#predicted_label = classifier.predict(valid_X)
#
#acc = classifier.score(valid_X,valid_label)
#print('Accuracy of the model:', acc)
#
#conf_mat = confusion_matrix(valid_label, predicted_label)
#print('\nConfusion Matrix: \n', conf_mat)
#
#print('\nClassification Report: \n',classification_report(valid_label, predicted_label))

# Acc of 0.8521 w 14 PCA 
    
loo = LeaveOneOut()
acc1 = 0
square_error_sum = 0.0
count = 0
for train_index, test_index in loo.split(X):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    classifier = svm.SVC(gamma= 10 , C = 7, decision_function_shape='ovo', kernel = 'rbf') 
    
    classifier = classifier.fit(X_train, y_train)
    predicted_y = classifier.predict(X_test)
    acc = classifier.score(X_test,y_test)
    acc1 = acc + acc1
    count += 1
    
acc1 = acc1/X.shape[0]
print('Accuracy of the model with LOO:', acc1)
print('\n')

# K-Fold CV
# Accuracy of 0.8522 with 8 PCs and 12-fold CV

for k in (range(2,8)):
    kfold = KFold(k, True, 1)
    
    acc1 = 0
    square_error_sum = 0.0
    count = 0
    for train_index, test_index in kfold.split(X):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        classifier = svm.SVC(gamma= 10 , C = 7, decision_function_shape='ovo', kernel = 'rbf') 
    
        classifier = classifier.fit(X_train, y_train)
        predicted_y = classifier.predict(X_test)
        acc = classifier.score(X_test,y_test)
#        print(acc)
        acc1 = acc + acc1
        count += 1
        
        plt.plot(acc)
    acc1 = acc1/k
    print(k,'-fold CV Accuracy of the model:', acc1)
    
    
#print('\nMean squared error:', mean_squared_error(valid_label, predicted_labels))


#fig = plt.figure() 
#plot_decision_regions(X=train_X, y=train_label, clf=classifier)
#
## Update plot object with X/Y axis labels and Figure Title
#plt.xlabel('PC1', size=14)
#plt.ylabel('PC2', size=14)
#plt.title('SVM Decision Region Boundary', size=16)
#plt.show()
#
##x1 = pca.fit_transform(x1)
##res = prediction(x1)
##print(res)
#
#fig = plt.figure()
#plot_decision_regions(X=valid_X, y=valid_label, clf=classifier)
#
## Update plot object with X/Y axis labels and Figure Title
#plt.xlabel('PC1', size=14)
#plt.ylabel('PC2', size=14)
#plt.title('SVM Decision Region Boundary', size=16)
#plt.show()