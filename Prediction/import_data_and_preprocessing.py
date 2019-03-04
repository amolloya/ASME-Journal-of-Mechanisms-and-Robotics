import numpy as np
import pandas as pd
from sklearn import impute
from sklearn.preprocessing import StandardScaler

N = 108

# Fills in the nan or missing values
imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')

# Importing body parameters
x = pd.read_excel('Body parameters.xls', header = None)

# Fit the imputer function to fill in missing values in the data
x = imputer.fit_transform(x)

# Normalizing the data
scaler = StandardScaler()
x = scaler.fit_transform(x)

y = [[[] for i in range(14)] for j in range(77)]

# Importing the joint angle trajectories
for i in range(1,109):
    filename = 'Sheet%s' % str(i)
    df = pd.read_excel('Kinematic parameters.xls', filename , header = None)
    df = imputer.fit_transform(df)
    y = np.dstack((y,df))

#title_str = ['Pelvis X disp', 'Pelvis Y disp', 'Pelvis Z disp', 'Pelvis Rotation', 'R. Hip adduction', 'R. Hip extension', 'R. Hip medial rotation', 'R. Knee Flexion', 'R. Ankle P.flex', 'L. Hip abduction', 'L. Hip extension', 'L. Hip lateral rotation', 'L. Knee flexion', 'L. Ankle P.flex']
#y_axis_str = ['mm', 'mm', 'mm', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg']

Y = [[[] for i in range(108)] for j in range(77)]

# Rearranging the joint angle trajectories of same joint together
for M in range(14):
    yz =  [[] for i in range(77)]
    for i in range(N):
        yy = y[:,M,i]
        yz = np.column_stack((yz, yy))
        
    Y = np.dstack((Y,yz))
