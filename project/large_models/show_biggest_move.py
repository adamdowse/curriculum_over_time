import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def euq_dis (x):
    #calculate the difference between equlidian distances of 2 points 
    #convert str to array
    a = x.iloc[0].replace('[','').replace(']','').split(', ')
    b = x.iloc[1].replace('[','').replace(']','').split(', ')
    a = [float(i) for i in a]
    b = [float(i) for i in b]

    a = [i**2 for i in a]
    a = np.power(np.sum(a),(1/10))

    b = [i**2 for i in b]
    b = np.power(np.sum(b),(1/10))

    #compare distances
    dist = a-b
    return dist

#show the analysis of the prediction space (softmax errors)

#download the data (i|)
df = pd.read_csv('/com.docker.devenvironments.code/project/large_models/datasets/mnist/normal_loss_info.csv')
df = df.set_index('i').drop('score',axis=1)

#create a df with the indexes and labels to record the euq change
df_change = df.copy().iloc[:,0:1]
print(df.head())
print(df.iloc[1,:])
print(df_change.head())

#euclidian distance
for i in range(3,31):

    #calculate the difference between the 2 epochs and add to the df_change df
    df_change[str(i)]  = df.iloc[:,i-2:i].apply(euq_dis,axis=1)

print(df_change.head())

#print graphz
for c in range(10):
    plt.plot(df_change[df_change.label == c].iloc[:,2:].mean(axis=0),label=c,alpha=0.7)
plt.plot(df_change.iloc[:,2:].mean(axis=0),label='Avg',color='k')

plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Change in Euclidean Distance in\nSoftmax Error Space Between (t-1) - (t)\nFor Training Data')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('normal_biggest_pred_move',bbox_inches='tight')
plt.close()

#print simple graphz
for c in [1,7]:
    plt.plot(df_change[df_change.label == c].iloc[:,2:].mean(axis=0)-df_change.iloc[:,2:].mean(axis=0),label=c,alpha=0.7)
plt.plot(df_change.iloc[:,2:].mean(axis=0)-df_change.iloc[:,2:].mean(axis=0),label='Avg',color='k')

plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Change in Euclidean Distance in\nSoftmax Error Space Between (t-1) - (t)\nFor Training Data')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('normal_biggest_pred_move_simple',bbox_inches='tight')
plt.close()




