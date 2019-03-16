#House Prices: Advanced Regression Techniques
#Very Simple and Basic Linear Regression code without any modeling
#Kaggle Score : 0.39792
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
features=["MSSubClass","LotFrontage","LotArea","PoolArea"]
X_train=pd.DataFrame(train[features])
y_train=train["SalePrice"]
X_test=pd.DataFrame(test[features])
for i in features:
    X_train[i].fillna(X_train[i].median(), inplace=True)
    X_test[i].fillna(X_test[i].median(), inplace=True)
regg = LinearRegression()
regg.fit(X_train, y_train)
y_predictions = regg.predict(X_test)
f=y_predictions
for i in range(len(f)):
    f[i]=int(f[i])
id=[]
for i in range(1461, 2920):
    id.append(i)
sub = pd.DataFrame()
sub['Id'] = id
sub['SalePrice'] = f
sub.to_csv('submission.csv',index=False)
