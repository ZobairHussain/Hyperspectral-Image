import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = pd.read_csv("processed_data.csv")
X=data.iloc[:,0:200]
Y=data.iloc[:,200:201]

 #spliting the dataset into training & testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#applying LDA
lda=LDA()
X_train=lda.fit_transform(X_train,Y_train)
X_test=lda.transform(X_test)

#fitting Logistic Regression to the Training set
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

#predicting the test set results
Y_pred=classifier.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)  #83% ase normally...

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
'''
from matplotlib.colors import ListedColormap
X_set,Y_set=X_train,Y_train
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),
                    np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=.75,cmap=ListedColormap(('red','green','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('red','green','blue'))(i), lebel = j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()
'''