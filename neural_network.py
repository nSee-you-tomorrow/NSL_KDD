# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

df = pd.read_csv('data.csv') 
print(df.shape)
#print(df.describe().transpose())

target_column = ['output','num_outbound_cmds']
#target_column = ['output', 'land', 'wrong_fragment', 'urgent', 'num_failed_logins', 'root_shell', 'su_attempted', 'num_shells', 'num_outbound_cmds', 'is_host_login', 'is_guest_login']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
print(set(list(df.columns)))
print(set(target_column))
X = df[predictors].values
y = df[['output']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,20,10), activation='relu', solver='adam', max_iter=1000, learning_rate_init=0.0001)
cfl = mlp.fit(X_train,y_train.reshape(len(y_train), 1))

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
print(cfl.score(X_test, predict_test))
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
