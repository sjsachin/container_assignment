import numpy as np
import pandas as pd
import pickle
data_1=pd.read_csv("Ex03_SystolicBP_Regreesion.csv")
from sklearn.model_selection import train_test_split
X=data_1.iloc[:,0:-1]
Y=data_1["SystolicBP"]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=0)
print(x_train.info())
print(data_1)
pickle_out=open("x_train_file.pickle","wb")
pickle.dump(x_train,pickle_out)
pickle_out.close()
pickle_out=open("x_test_file.pickle","wb")
pickle.dump(x_test,pickle_out)
pickle_out.close()
pickle_out=open("y_train_file.pickle","wb")
pickle.dump(y_train,pickle_out)
pickle_out.close()
pickle_out=open("y_test_file.pickle","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()
