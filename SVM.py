import scipy.io
data = scipy.io.loadmat('data.mat')

import scipy.io
label = scipy.io.loadmat('class.mat')

import numpy as np 
X=data["data"]
y=label["class"]
y=np.ravel(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)

from sklearn.svm import LinearSVC
classifier = LinearSVC(random_state=0, tol=1e-5,multi_class="ovr")
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
zoba = accuracy_score(y_test, y_pred)
print(zoba)