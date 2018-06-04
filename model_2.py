import numpy as np
import pandas as pd
import pickle
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pickle_in=open("data_1.pickle","rb")
data_1=pickle.load(pickle_in)

corelation_value=data_1.corr()["SystolicBP"][0:-1]
print(corelation_value)
corelation_in_descend=corelation_value[abs(corelation_value)>0].sort_values(ascending=False)
#print(corelation_in_descend)
ser=Series(corelation_in_descend)
lst=list()

for i,j in zip(ser.index,ser.values):
    if j>0.300000:
        lst.append(i)
    elif j<-0.300000:
        lst.append(i)
    else:
        continue
print (lst)

X=data_1[lst]
#X.info()
Y=data_1["SystolicBP"]
#print(Y.head())

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=0)
linreg= LinearRegression()
model_2=linreg.fit(x_train,y_train)
predicted_values=model_2.predict(x_test)
#print(r2_score(y_test,predicted_values))
model_2_score=r2_score(y_test,predicted_values)
print(model_2_score)

pickle_out=open("model_2_score.pickle","wb")
pickle.dump(model_2_score,pickle_out)
pickle_out.close()
