#import tabula
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from numpy.random import seed

#seed(1456)

# Read pdf into DataFrame
#df11 = tabula.read_pdf('Dataset.pdf', nospreadsheet=True, pages = '6', stream=True)
##df1 = tabula.convert_into("Dataset.pdf", "data.csv", output_format="csv")

#tabula.convert_into("Dataset.pdf", "CA_anthropometric.csv", output_format="csv", pages = '2')
#tabula.convert_into("Dataset.pdf", "HSP_anthropometric.csv", output_format="csv", pages = '3')
#tabula.convert_into("Dataset.pdf", "PD_anthropometric.csv", output_format="csv", pages = '4')
#tabula.convert_into("Dataset.pdf", "HC_anthropometric.csv", output_format="csv", pages = '5')

#tabula.convert_into("Dataset.pdf", "CA.csv", output_format="csv", pages = '6')
#tabula.convert_into("Dataset.pdf", "HSP.csv", output_format="csv", pages = '7')
#tabula.convert_into("Dataset.pdf", "PD.csv", output_format="csv", pages = '8')
#tabula.convert_into("Dataset.pdf", "HC.csv", output_format="csv", pages = '9,10')

x1 = pd.read_excel('CA.xlsx')
x1 = x1.iloc[:,1:]

x2 = pd.read_excel('HSP.xlsx')
x2 = x2.iloc[:,1:]

x3 = pd.read_excel('PD.xlsx')
x3 = x3.iloc[:,1:]

x4 = pd.read_excel('HC.xlsx')
x4 = x4.iloc[:,1:]

X1 = np.concatenate((x1,x1,x1,x2,x2,x3,x3,x4))


x1 = pd.read_excel('CA_anthropometric.xlsx')
x1 = x1.drop(columns = ['Subject-ID', 'Disease duration (years)', 'SARA-total', 'Diagnosis'])
x1 = x1.values

index = 0
for ind in range(0,len(x1)):
    if x1[ind,index] == 'M':
        x1[ind,index] = 0
    else:
        x1[ind,index] = 1
        
x1 = pd.DataFrame(x1)

x2 = pd.read_excel('HSP_anthropometric.xlsx')
x2 = x2.drop(columns = ['Subject-ID','Disease duration (years)', 'SPRS total', 'Diagnosis'])
x2 = x2.values

index = 0
for ind in range(0,len(x2)):
    if x2[ind,index] == 'M':
        x2[ind,index] = 0
    else:
        x2[ind,index] = 1
        
x2 = pd.DataFrame(x2)

x3 = pd.read_excel('PD_anthropometric.xlsx')
x3 = x3.drop(columns = ['Subject-ID','Disease duration (years)', 'UPDRS III', 'Diagnosis'])
x3 = x3.values

index = 0
for ind in range(0,len(x3)):
    if x3[ind,index] == 'M':
        x3[ind,index] = 0
    else:
        x3[ind,index] = 1
        
x3 = pd.DataFrame(x3)

x4 = pd.read_excel('HC_anthropometric.xlsx')
x4 = x4.drop(columns = ['Subject-ID'])
x4 = x4.values
#x4 = x4[:30,:]

index = 0
for ind in range(0,len(x4)):
    if x4[ind,index] == 'M':
        x4[ind,index] = 0
    else:
        x4[ind,index] = 1
        
x4 = pd.DataFrame(x4)

X2 = np.concatenate((x1,x1,x1,x2,x2,x3,x3,x4))
X2 = pd.DataFrame(X2, columns=['Gender', 'Age', 'Weight (kg)', 'Height (m)'])

X = np.concatenate((X2, X1), axis = 1)
X = pd.DataFrame(X)

#scaler = StandardScaler()
#X = scaler.fit_transform(X)

X = normalize(X, norm='l2', axis = 1)
pca = PCA(n_components= 8)
X = pca.fit_transform(X)
var = pca.explained_variance_ratio_
tot_var = sum(var)
print('Total variance covered by PCs', tot_var)
print('\n')

Y = np.concatenate((np.zeros(57),np.ones(52),2*np.ones(64),3*np.ones(65)))
#Y = to_categorical(Y)
X, Y = shuffle(X, Y, random_state=0)


#fig = plt.figure()
#ax = plt.axes(projection='3d')
#plt.title('Plot of Top 3 PCs of each disease group',  fontsize=50, ha='center')
#
#for i in range(len(X)):
#    if Y[i] == 0:
#        l1 = ax.scatter3D(X[i,0], X[i,1], X[i,2], c='r', marker='o', s=40)
#    if Y[i] == 1:
#        l2 = ax.scatter3D(X[i,0], X[i,1], X[i,2], c='b', marker='x', s=40)
#    if Y[i] == 2:
#        l3 = ax.scatter3D(X[i,0], X[i,1], X[i,2], c='g', marker='*', s=40)
#    if Y[i] == 3:
#        l4 = ax.scatter3D(X[i,0], X[i,1], X[i,2], c='m', marker='D', s=40)
#        
#ax.set_xlabel('PC1')
#ax.set_ylabel('PC2')
#ax.set_xlabel('PC3')     
#legend = ax.legend((l1, l2, l3, l4), ('CA', 'HSP', 'PD', 'HC'), prop={'size': 18} , borderpad=2)
#plt.show()
#
#
#fig = plt.figure()
#plt.title('Plot of Top 2 PCs of each disease group')
#
#for i in range(len(X)):
#    if Y[i] == 0:
#        plt.plot(X[i,0], X[i,1], marker = 'o', color='r', markersize = 8)
#    if Y[i] == 1:
#        plt.plot(X[i,0], X[i,1], marker = 'x', color='b', markersize = 8)
#    if Y[i] == 2:
#        plt.plot(X[i,0], X[i,1], marker = '*', color='g', markersize = 8)
#    if Y[i] == 3:
#        plt.plot(X[i,0], X[i,1], marker = 'D', color='m', markersize = 8)
#
#plt.xlabel('PC1')
#plt.ylabel('PC2')
#legend = plt.legend((l1, l2, l3, l4), ('CA', 'HSP', 'PD', 'HC'), prop={'size': 18} , borderpad=2)
#plt.show()
