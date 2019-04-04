import pandas as pd
import numpy as np

data = pd.read_csv("mrmr_processed_data.csv")
Y=data.iloc[:,200:201]
Y=np.ravel(Y)

X=data.iloc[:,0:100]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .2, random_state = 0)

from sklearn.svm import LinearSVC
classifier = LinearSVC(random_state=0, tol=1e-5,multi_class="ovr")
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)

from sklearn.metrics import confusion_matrix
cm=(confusion_matrix(y_test, y_pred))