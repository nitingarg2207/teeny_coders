import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle


data = pd.read_csv('heart.csv')
data.head()

X = data.drop('target',axis=1)
Y = data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state = 0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
model.fit(X_train, Y_train)
Y_predicted = model.predict(X_test)
score = accuracy_score(Y_test, Y_predicted)
print("Accuracy:",score*100)
arr=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(1,-1)
print(model.predict(arr))

filename='prediction_model.pkl'
# pickle.dump(model,open(filename,'wb'))
with open('prediction_model.pkl', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)