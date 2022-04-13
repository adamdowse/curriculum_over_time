import pandas as pd
import numpy as np
from sklearn import linear_model


model = linear_model.LinearRegression()

def func(data):
    n = 2
    x = np.array(range(n)).reshape((-1,1))
    y = data[-(n+1):-1].to_numpy() 
    model = linear_model.LinearRegression().fit(x,y)
    out = model.predict(np.array([n]).reshape((-1,1)))[0]
    data.iloc[-1] = out
    print(data)

    return data

s = pd.DataFrame(data = {'label':[1,2,3,2,1],'score':[np.nan]*5,'0':[5,2,4,1,8],'1':[1,2,3,4,5],'2':[np.nan]*5})
print(s)
#s = s.apply(lambda row : func(row),axis=1)
for i, row in s.iterrows():
    print(row.iloc[-1])
    if pd.isnull(row.iloc[-1]):
        print(row.iloc[-1])
print(s)


a = np.nan