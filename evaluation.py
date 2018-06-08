import pickle
import numpy as np
import pandas as pd

pickle_in=open("/result/model_1_score.pickle","rb")
model_1_score=pickle.load(pickle_in)

pickle_in=open("/result/model_2_score.pickle","rb")
model_2_score=pickle.load(pickle_in)

from datetime import datetime
datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')

file = open('dataOn_'+datestring+'.txt', "w")

if model_1_score>model_2_score:
   file.write("model_1 is better ")
   #file.write("model_1 score is")
   #file.write(model_1_score)
elif model_1_score < model_2_score:
    file.write("model_2 is better ")
    #file.write("model_2 score is")
    #file.write(model_2_score)
else:
    file.write("Both model have equal accuracy")

file.close()
