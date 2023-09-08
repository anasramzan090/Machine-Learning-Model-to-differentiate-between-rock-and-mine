import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sonar_data=pd.read_excel('D:\\ML Programs\\Projects\\rock_vs_mine\\sonar_data.xlsx', header=None)
print(sonar_data.head()) #first five value of the data
print(sonar_data.shape) #to fien the shape of the data frame
print(sonar_data.describe()) #for statistical measure of the data
print(sonar_data[60].value_counts()) #value difference in data
print(sonar_data[59].value_counts())
print(sonar_data.groupby(60).mean())
print(sonar_data.groupby(59).mean())
print(sonar_data.groupby(58).mean())
X=sonar_data.drop(columns=60,axis=1) #data values
Y=sonar_data[60] #data labels
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
model=LogisticRegression().fit(X_train,Y_train)
X_train_pridiction=model.predict(X_train)
training_Accuracy=accuracy_score(Y_train,X_train_pridiction)
print(training_Accuracy)
X_test_pridiction=model.predict(X_test)
test_Accuracy=accuracy_score(Y_test,X_test_pridiction)
print(test_Accuracy)
input_data=(0.0762,0.0666, 0.0481,	0.0394,	0.059,	0.0649,	0.1209	,0.2467,0.3564,	0.4459,	0.4152,	0.3952,	0.4256	,0.4135,0.4528,0.5326,0.7306,0.6193,0.2032,0.4636,0.4148,0.4292,0.573,0.5399,0.3161,0.2285,	0.6995,	1,	0.7262,	0.4724	,0.5103,	0.5459	,0.2881,0.0981,	0.1951,	0.4181,	0.4604,	0.3217,	0.2828	,0.243,	0.1979,	0.2444	,0.1847,	0.0841,	0.0692	,0.0528	,0.0357	,0.0085,	0.023	,0.0046	,0.0156	,0.0031	,0.0054,	0.0105	,0.011	,0.0015,0.0072,0.0048,0.0107,0.0094)
ipd_as_np_array=np.asarray(input_data)
data_reshape=ipd_as_np_array.reshape(1,-1)
prediction=model.predict(data_reshape)
print(prediction)
