import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import time

data = pd.read_csv("processed_data.csv")
data=data.values
classes =data[: ,200:201 ]
classes=np.ravel(classes)
classes=classes.astype(int)
attribute=data[:,0:200]
'''
d=0
for j in range(0,18):
    c=0
    for row in data:
        if row[200]==j:
            c=c+1
    print(d,c)
    d=d+1
'''
X_train, X_test, Y_train, Y_test = train_test_split(attribute, classes, test_size = .2, random_state = 0)
#for i in range(100,200):
pcaobj=PCA(n_components=80)
principalComponents = pcaobj.fit_transform(X_train)
new_space=pcaobj.components_
new_space=new_space.transpose()          ##why this??? not the built in
new_X_train=np.matmul(X_train,new_space)
new_X_test=np.matmul(X_test,new_space)

fullStart = time.time()
kf = KFold(n_splits=10)
bestC = 0
bestG = 0
maxAcc = 0
for c in range(130,150,2):
    for g in range(4,6,1):
        start = time.time()
        avgacc = 0
        for train_index, test_index in kf.split(new_X_train):
           #print("TRAIN:", train_index, "TEST:", test_index)
           x_train, x_test = new_X_train[train_index], new_X_train[test_index]
           y_train, y_test = Y_train[train_index], Y_train[test_index]
           classifier = SVC(C=c,class_weight=None,gamma=g,random_state=0)
           classifier.fit(x_train, y_train)
           Y_pred = classifier.predict(x_test)
           accuracy = accuracy_score(y_test, Y_pred)
           avgacc = avgacc + accuracy
        avgacc = avgacc/10
        if avgacc > maxAcc:
            maxAcc = avgacc
            bestC = c
            bestG = g
        end = time.time()
        print("c=",c,",g=",g,",acc=",avgacc,",best acc so far = ",maxAcc,", time = ",(end-start)/60)
        
          
classifier = SVC(C=bestC,gamma=bestG,random_state=0)
classifier.fit(new_X_train, Y_train)
Y_pred = classifier.predict(new_X_test)
accuracy = accuracy_score(Y_test, Y_pred)
fullEnd = time.time()
#print("Accuracy = ",accuracy)
print("bestC=",bestC,",bestG=",bestG,",final accuracy = ",accuracy,", time took = ", (fullEnd-fullStart)/3600)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

'''
var=pcaobj.explained_variance_ratio_
print(var)
sum=0
j=1
for i in var:
    sum=sum+i
    print(j,sum)
    j=j+1



mrmr_features = pymrmr.mRMR(pca_attribute, 'MID', 35)  #MID or MIQ

df=pd.DataFrame(np.array(lst))
mrmr=[1,24,25,22,23,28,21,26,19,27,30,31,32,33,34]
pca_mrmr_features = df.iloc[:, mrmr].values
'''


'''
from sklearn.decomposition import PCA
pca = PCA().fit(attribute)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()
'''

''' kaj kore na
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
estimator = Lasso()
featureSelection = SelectFromModel(estimator)
featureSelection.fit(X, Y)
selectedFeatures = featureSelection.transform(X)
selectedFeatures
'''