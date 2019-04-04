import pandas as pd
import numpy as np

data = pd.read_csv("processed_data.csv")
mrmr = pd.read_csv("mRMR.csv")
data=data.values
mrmr=mrmr.values
data=np.transpose(data)
X=np.empty([0,20010])
j=0
for i in range(200):
    X = np.insert(X,j,data[mrmr[i]-1], axis=0)
    j=j+1
X = np.insert(X,j,data[200], axis=0)
X=np.transpose(X)
df=pd.DataFrame(X)
df.to_csv("./mrmr_processed_data.csv", sep=',',index=False)