import pandas as pd
import numpy as np
import math
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
dataused = 100#len(df.index)
front = df.iloc[:dataused,:3]
df = df.iloc[:dataused,3:]
for i in range(n,len(df.columns)-3):
    rolling_grads = calc_grad(n,df.iloc[:,i-n:i])
    if i == n:
        #Create the df
        grads = pd.DataFrame(data=rolling_grads,columns=[str(i)])
    else:
        grads = pd.concat([grads,pd.DataFrame(data=rolling_grads ,columns=[str(i)])],axis=1)

front = front.reset_index().drop(columns=['Unnamed: 0'])
grads = pd.concat([front,grads],axis=1)

'''
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
'''
'''
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
        correlation[i,j] = i_df.corr(j_df,method='spearman')
        ax[i,j].scatter(i_df,j_df,s=0.25,label=str(i)+ ' - '+str(j))
        #ax[i,j].set_ylim([-5,1])
        #ax[i,j].set_xlim([-5,20])
        ax[i,j].set_xlabel(str(i)+str(j))
        if i != 10:
            ax[i,j].set_yticks([])
        if j != 0:
            ax[i,j].set_xticks([])
fig.savefig('corr_plot_spear_5',dpi=500,bbox_inches='tight')
plt.close()

plt.imshow(correlation,cmap='hot')
plt.xticks(range(10))
plt.yticks(range(10))
plt.colorbar()
#plt.clim(-1,1)
plt.title('Spearman\'s Correlation Between Classes Based On Percentage Change Of\nGradient Difference From The Mean Over 5 Epochs')
plt.savefig('heatmap_5_frommean_spear',bbox_inches='tight')
plt.close()
'''

#measuring the correlation between all of the images:
dataused= 100 #number of images to use
#maybe sort by the label here
grads = grads.sort_values(['label'])
img_corr = (grads.iloc[:dataused,3:]-grads.iloc[:dataused,3:].mean(axis=0)).pct_change(axis=1).T.corr()
#img correlations are in relative indexes not img identifiers here
'''
plt.imshow(img_corr,cmap='hot')
plt.colorbar()
#calc number of each class
num_class = [len(grads[grads['label']==x].index) for x in range(10)]
per_class = []
mid_class = []
for i in range(10):
    if i == 0:
        per_class.append(num_class[i])
        mid_class.append(math.floor(num_class[i]/2))
    else:
        per_class.append(num_class[i] + per_class[i-1])
        mid_class.append(math.floor(num_class[i]/2) + per_class[i-1])
print(per_class)
print(mid_class)

per_class = per_class[:-1]
plt.vlines(per_class,0,100)
plt.hlines(per_class,0,100)
plt.xlim([0,100])
plt.ylim([100,0])
plt.xticks(mid_class,range(10))
plt.yticks(mid_class,range(10))

plt.savefig('img_heatmap')
'''

#find the top and bottom n correlated images
#set the diagonal to 0
for i in range(dataused):
    img_corr.iloc[i,i] = 0
#find the highest correlations
sorted_idx = np.dstack(np.unravel_index(np.argsort(img_corr.values.ravel()), (dataused, dataused)))
sorted_idx = sorted_idx[0][::2]
#convert to img_corr indexes
sorted_idx = [[img_corr.index[x[0]],img_corr.index[x[1]]] for x in sorted_idx]
#convert to og img indexes
sorted_idx = [[grads.loc[x[0],'i'],grads.loc[x[1],'i']] for x in sorted_idx]
#list the correlations
sorted_corr = np.sort(img_corr.values.ravel())
sorted_corr = sorted_corr[::2]
print(str(sorted_corr[-4])[:5])
#using this show the correlated and uncorrelated images
img_data = pd.read_csv('datasets/mnist/imagedata.csv')
#need to use the whole dataset as the indexes need to be referenced (should be able to shorten in some way)
img_data = img_data.drop(columns=['Unnamed: 0'])
img_data = img_data.set_index('i')

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
fig , ax = plt.subplots(5,4)

for i in range(5):
    ax[i,0].imshow(img_dim_shift(img_data.loc[sorted_idx[i][0],'img'],info))
    ax[i,1].imshow(img_dim_shift(img_data.loc[sorted_idx[i][1],'img'],info))
    ax[i,1].set_ylabel(str(sorted_corr[i])[:6],rotation=0,labelpad=25)

    ax[i,2].imshow(img_dim_shift(img_data.loc[sorted_idx[-(i+1)][0],'img'],info))
    ax[i,3].imshow(img_dim_shift(img_data.loc[sorted_idx[-(i+1)][1],'img'],info))
    ax[i,3].set_ylabel(str(sorted_corr[-(i+1)])[:5],rotation=0,labelpad=25)

for i in range(4):
    for j in range(5):
        ax[j,i].set_xticks([])
        ax[j,i].set_yticks([])
ax[0,2].annotate('',xy=(-0.5,1),xycoords='axes fraction',xytext=(-0.5,-5),arrowprops=dict(arrowstyle="-",color='k'))

plt.savefig('most_corr')











