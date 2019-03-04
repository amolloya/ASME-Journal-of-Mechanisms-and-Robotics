import numpy as np
import pandas as pd
from sklearn import impute
from sklearn.preprocessing import StandardScaler

# Fills in the nan or missing values
imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')

x = pd.read_excel('Body parameters.xls', header = None)
x = imputer.fit_transform(x)
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_trunc = x[:,:4]

y = [[[] for i in range(14)] for j in range(77)]

for i in range(1,109):
    filename = 'Sheet%s' % str(i)
    df = pd.read_excel('Kinematic parameters.xls', filename , header = None)
    df = imputer.fit_transform(df)
    y = np.dstack((y,df))

#title_str = ['Pelvis X disp', 'Pelvis Y disp', 'Pelvis Z disp', 'Pelvis Rotation', 'R. Hip adduction', 'R. Hip extension', 'R. Hip medial rotation', 'R. Knee Flexion', 'R. Ankle P.flex', 'L. Hip abduction', 'L. Hip extension', 'L. Hip lateral rotation', 'L. Knee flexion', 'L. Ankle P.flex']
#y_axis_str = ['mm', 'mm', 'mm', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg']
#
sub_list = range(108)
sub_list = np.transpose(sub_list)
N = len(sub_list)

time_frame = 77
    
X = [[[] for i in range(15)] for j in range(77)]
    
for i in range(N):
        
    xb = [[] for i in range(15)]
    xb = np.transpose(xb)
        
    for j in range(1,time_frame+1):
            
        xa = x[i,:]
        xa = np.append(xa, j)
           
        xb = np.row_stack((xb,xa))
        scaler = StandardScaler()
        xb = scaler.fit_transform(xb)
    X = np.dstack((X,xb))
        
        
Y = [[[] for i in range(108)] for j in range(77)]

for M in range(14):
   
    yz =  [[] for i in range(77)]

    for i in range(N):
        
        yy = y[:,M,i]
        yz = np.column_stack((yz, yy))

    Y = np.dstack((Y,yz))
    
    
    
X_trunc = [[[] for i in range(5)] for j in range(77)]
    
for i in range(N):
        
    xb_trunc = [[] for i in range(5)]
    xb_trunc = np.transpose(xb_trunc)
        
    for j in range(1,time_frame+1):
            
        xa_trunc = x_trunc[i,:]
        xa_trunc = np.append(xa_trunc, j)
           
        xb_trunc = np.row_stack((xb_trunc,xa_trunc))
        scaler = StandardScaler()
        xb = scaler.fit_transform(xb_trunc)
    X_trunc = np.dstack((X_trunc,xb_trunc))
