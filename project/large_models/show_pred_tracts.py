import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition


#pull data from df_train_losses to in [i|(10dim),(10dim)...] to np array
#choose the traces
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

df_train_losses = pd.read_csv('/com.docker.devenvironments.code/project/large_models/datasets/mnist/normal_loss_info.csv')
df_train_losses = df_train_losses.set_index('i')
#print(df_train_losses[df_train_losses.label == 2])
for i in [169,448,482,614,410,526,89,460,651,354,16,324]:
    print(df_train_losses)
    data = df_train_losses.iloc[:,2:] #((10dim), (10dim)..)
    data = data.loc[i,:]
    data = np.array(data.str.strip('[]').str.strip(' ').str.split(',').tolist(), dtype='float')
    ax.plot(data[:,0],data[:,1],data[:,2])
ax.set_xlabel('0')
ax.set_ylabel('1')
ax.set_zlabel('2')
fig.savefig('pred_cluster_trace_3dims')

#compress to 2 dims
pca = decomposition.PCA(n_components=2)
data = []
for i in range(len(df_train_losses.index)):
    a = np.array(df_train_losses.iloc[i,2:].str.strip('[]').str.strip(' ').str.split(',').tolist(), dtype='float')
    data.append(a)

data = np.array(data)
pca.fit(data.reshape((data.shape[0]*data.shape[1]),data.shape[2]))

fig = plt.figure()
ax = fig.add_subplot()
n = 10
labs = df_train_losses.iloc[:n,0].to_numpy()
for i in range(n):
    comp_data = pca.transform(data[i,:,:])
    ax.plot(comp_data[:,0],comp_data[:,1],label=str(labs[i]))
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
#ax.set_zlabel('PCA3')
ax.legend(loc='center left')
fig.savefig('pred_cluster_dim_red_2',bbox_inches = 'tight')