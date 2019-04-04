import numpy as np
import pandas as pd
import scipy.io

X = scipy.io.loadmat('Indian_pines_corrected.mat')
y = scipy.io.loadmat('Indian_pines_gt.mat')

X=X["indian_pines_corrected"]
X=np.reshape(X,(21025,200))
X=X.astype(int)
 
y=y["indian_pines_gt"]
y=y.flatten()
y=y.astype(int)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X = sc.transform(X)

X=np.column_stack((X,y))

d=0
for j in range(0,18):
    c=0
    for row in X:
        if row[200]==j:
            c=c+1
    print(d,c)
    d=d+1

Z = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200] in (2,3,5,6,8,10,11,12,14):
        Z = np.insert(Z, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Z)
df.to_csv("./processed_data.csv", sep=',',index=False)

"""
Alfalfa_1 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==1:
        Alfalfa_1 = np.insert(Alfalfa_1, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Alfalfa_1)
df.to_csv("./data_Alfalfa_1.csv", sep=',',index=False)

Corn_notil_2 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==2:
        Corn_notil_2 = np.insert(Corn_notil_2, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Corn_notil_2)
df.to_csv("./data_Corn_notil_2.csv", sep=',',index=False)

Corn_min_3 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==3:
        Corn_min_3 = np.insert(Corn_min_3, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Corn_min_3)
df.to_csv("./data_Corn_min_3.csv", sep=',',index=False)

Corn_4 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==4:
        Corn_4 = np.insert(Corn_4, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Corn_4)
df.to_csv("./data_Corn_4.csv", sep=',',index=False)

Grass_5 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==5:
        Grass_5 = np.insert(Grass_5, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Grass_5)
df.to_csv("./data_Grass_5.csv", sep=',',index=False)

Grass_trees_6 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==6:
        Grass_trees_6 = np.insert(Grass_trees_6, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Grass_trees_6)
df.to_csv("./data_Grass_trees_6.csv", sep=',',index=False)

Gress_pesture_Mowed_7 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==7:
        Gress_pesture_Mowed_7 = np.insert(Gress_pesture_Mowed_7, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Gress_pesture_Mowed_7)
df.to_csv("./data_Gress_pesture_Mowed_7.csv", sep=',',index=False)

Hey_Windrowed_8 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==8:
        Hey_Windrowed_8 = np.insert(Hey_Windrowed_8, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Hey_Windrowed_8)
df.to_csv("./data_Hey_Windrowed_8.csv", sep=',',index=False)

Oats_9 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==9:
        Oats_9 = np.insert(Oats_9, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Oats_9)
df.to_csv("./data_Oats_9.csv", sep=',',index=False)

Soybeans_notil_10 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==10:
        Soybeans_notil_10 = np.insert(Soybeans_notil_10, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Soybeans_notil_10)
df.to_csv("./data_Soybeans_notil_10.csv", sep=',',index=False)

Soybeans_min_11 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==11:
        Soybeans_min_11 = np.insert(Soybeans_min_11, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Soybeans_min_11)
df.to_csv("./data_Soybeans_min_11.csv", sep=',',index=False)

Soybeans_clean_12 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==12:
        Soybeans_clean_12 = np.insert(Soybeans_clean_12, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Soybeans_clean_12)
df.to_csv("./data_Soybeans_clean_12.csv", sep=',',index=False)

Wheat_13 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==13:
        Wheat_13 = np.insert(Wheat_13, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Wheat_13)
df.to_csv("./data_Wheat_13.csv", sep=',',index=False)

Wood_14 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==14:
        Wood_14 = np.insert(Wood_14, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Wood_14)
df.to_csv("./data_Wood_14.csv", sep=',',index=False)

Buildings_15 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==15:
        Buildings_15 = np.insert(Buildings_15, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Buildings_15)
df.to_csv("./data_Buildings_15.csv", sep=',',index=False)

Stone_Steel_tower_16 = np.empty([0, 201])
i=0
j=0
for row in X:
    if row[200]==16:
        Stone_Steel_tower_16 = np.insert(Stone_Steel_tower_16, j,X[i], axis=0)  #row wise, axis=0
        j=j+1
    i=i+1
df=pd.DataFrame(Stone_Steel_tower_16)
df.to_csv("./data_Stone_Steel_tower_16.csv", sep=',',index=False)



