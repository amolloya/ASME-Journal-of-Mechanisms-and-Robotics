import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

# Importing gait parameters of 4 groups
x1 = pd.read_excel('Dataset/CA.xlsx')
x1 = x1.iloc[:,1:]

x2 = pd.read_excel('Dataset/HSP.xlsx')
x2 = x2.iloc[:,1:]

x3 = pd.read_excel('Dataset/PD.xlsx')
x3 = x3.iloc[:,1:]

x4 = pd.read_excel('Dataset/HC.xlsx')
x4 = x4.iloc[:,1:]

# Taking care of class imbalance problem by resampling the minimum frequency of classes
X1 = np.concatenate((x1,x1,x1,x2,x2,x3,x3,x4))

# Importing body parameters of 4 groups
x1 = pd.read_excel('Dataset/CA_anthropometric.xlsx')
x1 = x1.drop(columns = ['Subject-ID', 'Disease duration (years)', 'SARA-total', 'Diagnosis'])
x1 = x1.values

# Assigning 0 and 1 label to Male and Female group respectively
index = 0
for ind in range(0,len(x1)):
    if x1[ind,index] == 'M':
        x1[ind,index] = 0
    else:
        x1[ind,index] = 1       
x1 = pd.DataFrame(x1)

x2 = pd.read_excel('Dataset/HSP_anthropometric.xlsx')
x2 = x2.drop(columns = ['Subject-ID','Disease duration (years)', 'SPRS total', 'Diagnosis'])
x2 = x2.values

index = 0
for ind in range(0,len(x2)):
    if x2[ind,index] == 'M':
        x2[ind,index] = 0
    else:
        x2[ind,index] = 1       
x2 = pd.DataFrame(x2)

x3 = pd.read_excel('Dataset/PD_anthropometric.xlsx')
x3 = x3.drop(columns = ['Subject-ID','Disease duration (years)', 'UPDRS III', 'Diagnosis'])
x3 = x3.values

for ind in range(0,len(x3)):
    if x3[ind,index] == 'M':
        x3[ind,index] = 0
    else:
        x3[ind,index] = 1      
x3 = pd.DataFrame(x3)

x4 = pd.read_excel('Dataset/HC_anthropometric.xlsx')
x4 = x4.drop(columns = ['Subject-ID'])
x4 = x4.values

for ind in range(0,len(x4)):
    if x4[ind,index] == 'M':
        x4[ind,index] = 0
    else:
        x4[ind,index] = 1     
x4 = pd.DataFrame(x4)

# Taking care of class imbalance problem by resampling the minimum frequency of classes
X2 = np.concatenate((x1,x1,x1,x2,x2,x3,x3,x4))
X2 = pd.DataFrame(X2, columns=['Gender', 'Age', 'Weight (kg)', 'Height (m)'])

# Concatenating the body and gait parameters into a single array, which acts as an input 
X = np.concatenate((X2, X1), axis = 1)
X = pd.DataFrame(X)

# Normalize the input data with L-2 norm
X = normalize(X, norm='l2', axis = 1)

# Fitting PCA to the input data and choosing top 8 PCs
pca = PCA(n_components= 8)
X = pca.fit_transform(X)
var = pca.explained_variance_ratio_
tot_var = sum(var)
print('Total variance covered by PCs', tot_var)

# Creating labels for the classes
Y = np.concatenate((np.zeros(57),np.ones(52),2*np.ones(64),3*np.ones(65)))

# Shuffling the data randomly
X, Y = shuffle(X, Y, random_state=0)
