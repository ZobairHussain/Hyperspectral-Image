import pandas as pd
import numpy as np
import pymrmr

load = pd.read_csv("processed_data.csv")
data=load.values
attribute=data[:,0:200]
sum=0.0
for i in attribute:
    for j in range(0,200,10):
        sum=(i[j]+i[j+1]+i[j+2]+i[j+3]+i[j+4]+i[j+5]+i[j+6]+i[j+7]+i[j+8]+i[j+9])/10
        i[j]=sum
        i[j+1]=sum
        i[j+2]=sum
        i[j+3]=sum
        i[j+4]=sum
        i[j+5]=sum
        i[j+6]=sum
        i[j+7]=sum
        i[j+8]=sum
        i[j+9]=sum
        sum=0
        
df=pd.DataFrame(attribute)
df.to_csv("./mRMR_attribute.csv", sep=',',index=False)

classes =data[: ,200:201]
classes=classes.astype(int)  #int na korle class alada korte pare na

# concatanate y,X
data = np.hstack((classes,attribute)) 

dataframe = pd.DataFrame(data)  #convert
dataframe.columns = dataframe.columns.astype(str)   #change the column name

mrmr_features = pymrmr.mRMR(dataframe, 'MID', 200) #powerfull line

df=pd.DataFrame(mrmr_features)
df.to_csv("./mRMR.csv", sep=',',index=False)
