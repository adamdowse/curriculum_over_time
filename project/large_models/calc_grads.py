import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import supporting_functions as sf

df = pd.read_csv('datasets/mnist/normal_loss_info.csv',index_col = 'i')

def calc_grad(n,data):
    grads = []
    for i,row in data.iterrows():
        x = np.array(range(n)).reshape((-1,1))
        y = row[-n:].to_numpy()
        model = linear_model.LinearRegression().fit(x,y)
        grads.append(model.coef_[0])
    return pd.Series(data=grads)

n=3
dataused = len(df.index)
front = df.iloc[:dataused,:3]
df = df.iloc[:dataused,3:]
for i in range(n,len(df.columns)-3):
    rolling_grads = calc_grad(n,df.iloc[:,i-n:i])
    if i == n:
        #Create the df
        grads = pd.DataFrame(data=rolling_grads,columns=[str(i)])
    else:
        grads = pd.concat([grads,pd.DataFrame(data=rolling_grads ,columns=[str(i)])],axis=1)


grads = pd.concat([front,grads],axis=1)

fig1, ax =plt.subplots(2)
for lab in range(10):
    sub_df = grads[grads['label']==lab].copy()
    ax[0].plot(sub_df.iloc[:,3:].mean(axis=0),label=str(lab))
    ax[1].plot(sub_df.iloc[:,3:].mean(axis=0)-grads.iloc[:,3:].mean(axis=0),label=str(lab))

ax[0].plot(grads.iloc[:,3:].mean(axis=0),label='Avg',color='k')
ax[1].plot(grads.iloc[:,3:].mean(axis=0)-grads.iloc[:,3:].mean(axis=0),label='Avg',color='k')
ax[1].legend(loc='lower left',bbox_to_anchor=(1,0.5))
ax[1].set_xlabel('Epoch')
ax[0].set_ylabel('Average Gradient')
ax[1].set_ylabel('Average Gradient\nDifference From Mean')
#ax[0].set_ylim([-0.2,0.05])
#ax[1].set_ylim([-0.15,0.15])
fig1.savefig('label_avg_grad',bbox_inches='tight')
plt.close()

#calc correlation
correlation = np.zeros((10,10))
fig, ax = plt.subplots(10,10)
for i in range(10):
    #take all of a single label and use % change
    i_df = grads[grads['label']==i].copy()
    i_df = i_df.iloc[:,3:].mean(axis=0)-grads.iloc[:,3:].mean(axis=0)
    i_df = i_df.pct_change()
    for j in range(10):
        #take all of a single label and use % change
        j_df = grads[grads['label']==j].copy()
        j_df = j_df.iloc[:,3:].mean(axis=0)-grads.iloc[:,3:].mean(axis=0)
        j_df = j_df.pct_change()
        correlation[i,j] = i_df.corr(j_df)
        ax[i,j].scatter(i_df,j_df,s=0.25,label=str(i)+ ' - '+str(j))
        #ax[i,j].set_ylim([-5,1])
        #ax[i,j].set_xlim([-5,20])
        ax[i,j].set_xlabel(str(i)+str(j))
        if i != 10:
            ax[i,j].set_yticks([])
        if j != 0:
            ax[i,j].set_xticks([])
fig.savefig('corr_plot_diffmean',dpi=500,bbox_inches='tight')
plt.close()

plt.imshow(correlation,cmap='hot')
plt.xticks(range(10))
plt.yticks(range(10))
plt.colorbar()
#plt.clim(-1,1)
plt.title('Correlation Between Classes Based\nOn Percentage Change Of Gradient Difference\nFrom The Mean Over 3 Epochs')
plt.savefig('heatmap_3_frommean')
plt.close()

#measuring the correlation between all of the images:
'''
dataused= len(grads.index) #number of images to use
#maybe sort by the label here
img_corr = grads.iloc[:dataused,3:].pct_change(axis=1).T.corr()

plt.imshow(img_corr,cmap='hot')
plt.colorbar()
plt.savefig('img_heatmap')
'''
'''
#find the top and bottom n correlated images
#set the diagonal to 0
for i in range(dataused):
    img_corr.iloc[i,i] = 0

#find the highest correlations
idx_max = img_corr.idxmax(axis=1)[0] #the col idx with highest corr in it
idx_min = img_corr.idxmin(axis=1)[0]  

n_max = img_corr.iloc[:,idx_max].nlargest(5) #in terms of relative index need to convert back to original
n_max = n_max.reset_index()

n_min = img_corr.iloc[:,idx_min].nsmallest(5)
n_min = n_min.reset_index()

#using this show the correlated and uncorrelated images
img_data = pd.read_csv('datasets/mnist/imagedata.csv')
img_data = img_data.iloc[:dataused]

def img_dim_shift(img,info):
    #x is a vector of strings
    img = img.replace('[','')
    img = img.replace(']','')
    img = img.replace('\n','')
    img = img.split(' ')
    img = [y.replace(' ','') for y in img]
    img = [y for y in img if y!= '']
    img = [float(y) for y in img]
    img = [y/256 for y in img]
    img = np.array(img)
    img = img.reshape(info.img_shape)
    return img

class Info():
    img_shape = (28,28,1)
info = Info()

#print base image and then most correlated
fig , ax = plt.subplots(5,2)
ax[0,0].imshow(img_dim_shift(img_data.loc[idx_max,'img'],info))
ax[0,1].imshow(img_dim_shift(img_data.loc[idx_min,'img'],info))
ax[0,0].tick_params(color='red')
ax[0,1].tick_params(color='red')
for i in range(1,5):
    ax[i,0].imshow(img_dim_shift(img_data.loc[n_max.loc[i,'index'],'img'],info))
    ax[i,1].imshow(img_dim_shift(img_data.loc[n_min.loc[i,'index'],'img'],info))

for i in range(2):
    for j in range(5):
        ax[j,i].set_xticks([])
        ax[j,i].set_yticks([])

plt.savefig('most_corr')

'''









