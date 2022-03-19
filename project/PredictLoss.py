import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
from sklearn import linear_model


#AIM:
#Show that loss can be predicted with simple equations online
#Current assumptions:
# - By removing the image from the dataset the loss of this image is not effected


df = pd.read_csv('/com.docker.devenvironments.code/project/Data/rank/OutputLayers/rank_df',index_col="Unnamed: 0")

plt.plot(df.iloc[:,3:].mean(axis=0))
plt.plot(df.iloc[:,3:].min(axis=0))
plt.plot(df.iloc[:,3:].max(axis=0))
plt.plot(df.iloc[:,3:].quantile(0.05,axis=0))
plt.plot(df.iloc[:,3:].quantile(0.95,axis=0))
plt.savefig('allplot')
plt.close()

change_df = df.iloc[:,3:].rolling(2,axis=1).var()

def tonan(x):
    if x[0] == x[1]:
        return np.nan
    else:
        return x[1]
n = 70
single_line = df.iloc[n,3:]
single_line = single_line.rolling(2).apply(tonan)
single_line[0] = df.iloc[n,3]

#plt.scatter(single_line)
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.savefig('singleline')
#plt.close()

#predict with mean
def testpred(pred,true):
    #pred is the column of preditions
    #true is the column of true values
    diff = []
    for i,p in pred.iteritems():
        if true[i] != np.nan:
            if true[i] != p:
                diff.append(math.sqrt(abs(p**2-true[i]**2)))
            else:
                diff.append(np.nan)
        else:
            diff.append(np.nan)
    return diff
    
def nmeanpred(n,data):
    #rows of data up to prediction point
    #n is the number of lookbacks
    roll_df = data.rolling(n,axis=1).mean()
    return roll_df.iloc[:,-1]

def regressionpred(n,data,lookforward):
    #data is the dat up to prediction time -1 
    preds = []
    for i,row in data.iterrows():
        x = np.array(range(n)).reshape((-1,1))
        y = row[-n:].to_numpy()
        model = linear_model.LinearRegression().fit(x,y)
        preds.append(model.predict(np.array(n+lookforward).reshape(-1,1)))
    return pd.Series(data=preds)

def ransacpred(n,data,lookforward):
    #ransac attempts to be robust to outliers 
    preds = []
    for i,row in data.iterrows():
        x = np.array(range(n)).reshape((-1,1))
        y = row[-n:].to_numpy()
        model = linear_model.RANSACRegressor(min_samples=x.shape[1] + 1).fit(x,y)
        preds.append(model.predict(np.array(n+lookforward).reshape(-1,1)))
    return pd.Series(data=preds)

lookforward = 10

for n in [2,3,5,20,50]:
    for i in range(n,100-lookforward):
        #predict the new column
        pred_nmean = nmeanpred(n,df.iloc[:,3:3+i])
        pred_reg = regressionpred(n,df.iloc[:,3:3+i],lookforward)
        pred_ransac = ransacpred(n,df.iloc[:,3:3+i],lookforward)

        #setup df
        if i == n:
            #create the scores df
            scores_nmean = pd.DataFrame(data=testpred(pred_nmean,df.iloc[:,i+lookforward]),columns=[str(i)])
            scores_reg   = pd.DataFrame(data=testpred(pred_reg,df.iloc[:,i+lookforward]),columns=[str(i)])
            scores_ransac= pd.DataFrame(data=testpred(pred_ransac,df.iloc[:,i+lookforward]),columns=[str(i)])
        else:
            scores_nmean = pd.concat([scores_nmean,pd.DataFrame(data=testpred(pred_nmean,df.iloc[:,i+lookforward]),columns=[str(i)])],axis=1)
            scores_reg = pd.concat([scores_reg,pd.DataFrame(data=testpred(pred_reg,df.iloc[:,i+lookforward]),columns=[str(i)])],axis=1)
            scores_ransac = pd.concat([scores_ransac,pd.DataFrame(data=testpred(pred_ransac,df.iloc[:,i+lookforward]),columns=[str(i)])],axis=1)


    plt.plot(scores_nmean.mean(axis=0,skipna=True),':',label=str(n)+'mean nmean')
    plt.plot(scores_reg.mean(axis=0,skipna=True),'--',label=str(n)+'mean linear regression')
    plt.plot(scores_ransac.mean(axis=0,skipna=True),label=str(n)+'mean RANSAC')


plt.legend(bbox_to_anchor=(1.04,0.5),loc='center left')
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.yscale('log')
plt.xticks(range(0,100,10))
plt.tight_layout()
plt.savefig('predfuncs/curric_predicts_lf10')
plt.close()




#Test on the non curriculum dataset

df = pd.read_csv('/com.docker.devenvironments.code/project/Data/normal/normal_df',index_col="Unnamed: 0")
df = df.iloc[:,:103]
for n in [2,3,5,20,50]:
    for i in range(n,100-lookforward):
        #predict the new column
        pred_nmean = nmeanpred(n,df.iloc[:,3:3+i])
        pred_reg = regressionpred(n,df.iloc[:,3:3+i],lookforward)
        pred_ransac = ransacpred(n,df.iloc[:,3:3+i],lookforward)

        #setup df
        if i == n:
            #create the scores df
            scores_nmean = pd.DataFrame(data=testpred(pred_nmean,df.iloc[:,i+lookforward]),columns=[str(i)])
            scores_reg   = pd.DataFrame(data=testpred(pred_reg,df.iloc[:,i+lookforward]),columns=[str(i)])
            scores_ransac= pd.DataFrame(data=testpred(pred_ransac,df.iloc[:,i+lookforward]),columns=[str(i)])
        else:
            scores_nmean = pd.concat([scores_nmean,pd.DataFrame(data=testpred(pred_nmean,df.iloc[:,i+lookforward]),columns=[str(i)])],axis=1)
            scores_reg = pd.concat([scores_reg,pd.DataFrame(data=testpred(pred_reg,df.iloc[:,i+lookforward]),columns=[str(i)])],axis=1)
            scores_ransac = pd.concat([scores_ransac,pd.DataFrame(data=testpred(pred_ransac,df.iloc[:,i+lookforward]),columns=[str(i)])],axis=1)


    plt.plot(scores_nmean.mean(axis=0,skipna=True),':',label=str(n)+'mean nmean')
    plt.plot(scores_reg.mean(axis=0,skipna=True),'--',label=str(n)+'mean linear regression')
    plt.plot(scores_ransac.mean(axis=0,skipna=True),label=str(n)+'mean RANSAC')


plt.legend(bbox_to_anchor=(1.04,0.5),loc='center left')
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.yscale('log')
plt.xticks(range(0,100,10))
plt.tight_layout()
plt.savefig('predfuncs/nocurric_predicts_lf10')
plt.close()















#plt.plot(scores.min(axis=0),label='min')
#plt.plot(scores.quantile(0.05,axis=0),label='0.05%')
#plt.plot(scores.mean(axis=0),label='mean')
#plt.plot(scores.quantile(0.95,axis=0),label='0.95%')
#plt.plot(scores.max(axis=0),label='max')
#plt.legend()
#plt.xlabel('epoch')
#plt.ylabel('RMSE')
#plt.savefig('predfuncs/nmean')
#plt.close()








plt.plot(change_df.iloc[1,:])
plt.plot(change_df.iloc[80,:])
plt.savefig('changeplot')
plt.close()

plt.imshow(df.rolling(2,axis=1).apply(tonan),norm=colors.LogNorm())
plt.savefig('heatmap')
plt.close()

