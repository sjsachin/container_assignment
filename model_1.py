import numpy as np
import pandas as  pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
pickle_in_1=open("x_train_file.pickle","rb")
x_train=pickle.load(pickle_in_1)
pickle_in_2=open("x_test_file.pickle","rb")
x_test=pickle.load(pickle_in_2)
pickle_in_3=open("y_train_file.pickle","rb")
y_train=pickle.load(pickle_in_3)
pickle_in_4=open("y_test_file.pickle","rb")
y_test=pickle.load(pickle_in_4)
linreg=LinearRegression()
model_1=linreg.fit(x_train,y_train)
predicted_value=model_1.predict(x_test)
print(r2_score(y_test,predicted_value))
model_1_score=r2_score(y_test,predicted_value)
pickle_out=open("model_1_score.pickle","wb")
pickle.dump(model_1_score,model_1_score.pickle)
pickle_out.close()
