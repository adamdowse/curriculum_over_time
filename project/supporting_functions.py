import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
import math

#interesting websites
#https://towardsdatascience.com/multiclass-logistic-regression-from-scratch-9cc0007da372#:~:text=Multiclass%20logistic%20regression%20is%20also,really%20know%20how%20it%20works.

def data_init(n,class_num,feature_num,stddev,random_state=1):
    '''
    Returns a dataframe of blobs with [x1, x2, ..., class_num]
    '''
    #create a simple 2d classification dataset with n points in 3 classes
    x, y = make_blobs(n_samples=n, centers=class_num, n_features=feature_num, random_state=random_state, cluster_std=stddev)

    #save to dataframe
    df = pd.DataFrame(x, columns=['x1','x2'])
    df['class'] = pd.DataFrame(y, columns=['class'])

    return df

def collect_data(df,info,test=False):
    '''
    Takes the dataframe and produces a tf.dataset and an updated df
    rank = neive ranking sorting df by last loss values and taking a percentage based on the pacing function
    '''
    if test == True:
        batch_size = 1
    else:
        batch_size = info.batch_size


    if info.scoring_func == 'normal' or test ==True:
        mod_df = df.copy()

    elif info.scoring_func == 'rank':
        mod_df = df.copy()
        if info.current_epoch>0:
            a = 0.8
            b = 0.2
            #exp
            data_count = info.max_data*b + (info.max_data*(1-b)*(math.exp((10*info.current_epoch)/(a*info.max_epoch))-1))/(math.exp(10) - 1)
            if data_count > info.max_data:
                data_count = info.max_data
            print('Data Count = ',data_count)
            
            df_x1 = mod_df.where(mod_df['class']==0).copy()
            df_x2 = mod_df.where(mod_df['class']==1).copy()

            #take the easiest points
            df_x1 = df_x1.sort_values(str(info.current_epoch-1))
            df_x2 = df_x2.sort_values(str(info.current_epoch-1))

            df_x1 = df_x1.head(int(data_count/2))
            df_x2 = df_x2.head(int(data_count/2))

            mod_df = pd.concat([df_x1,df_x2])
            mod_df = mod_df.sample(frac=1).reset_index(drop=True)
    
    elif info.scoring_func == 'meanlossgradient':
        #This loks at the average gradient direction of the loss and if the gradient is:
        # + then increase the images given
        # - then decrease the images given
        # 0 then keep it consistant
        # Also want the overall amount of images to increase to max (maybe TODO)
        mod_df = df.copy()
        
        #ensure there is enough loss info for calcs
        if info.current_epoch > 2:
            
            #calculate the pos and neg 95th percentiles of the gradients (TODO make more advanced here maybe function?)
            diff = lambda x: ((x[2]-x[1]) + (x[1]-x[0]))/2
            q95 = mod_df.iloc[:,3:].mean(axis=0).rolling(3).apply(diff,raw=True).quantile(q=0.95)
            q05 = mod_df.iloc[:,3:].mean(axis=0).rolling(3).apply(diff,raw=True).quantile(q=0.05)
            m = mod_df.iloc[:,3:].mean(axis=0).rolling(3).apply(diff,raw=True).ewm(com=0.5).mean().iloc[-1]
            med = mod_df.iloc[:,3:].mean(axis=0).rolling(3).apply(diff,raw=True).median()
            #move the sigma function to sit between the quantiles 
            #sigmoid = lambda x: (2 / (1 + math.exp(-x))) - 1
            #x is the loss
            #b increases gradient
            #f shifts in x
            f = m#(q95 + q05)/2
            b = 10#(math.log((0.95+1)) - math.log(1-0.95)) / (q95 - f)
            z = info.current_epoch/info.max_epoch
            sigmoid = lambda x: (2 / (1 + math.exp(-b*(x-f)))) - 1 + z

            #calc average gradient 
            avgLoss1 = mod_df.loc[:,str(info.current_epoch-1)].mean()
            avgLoss2 = mod_df.loc[:,str(info.current_epoch-2)].mean()
            avgLoss3 = mod_df.loc[:,str(info.current_epoch-3)].mean()

            avgGrad = diff([avgLoss3,avgLoss2,avgLoss1])
            
            print('0.05%, mean, 0.95%, med = ',q05,m,q95,med)
            print('avgGrad , sigmoid = ',avgGrad,sigmoid(avgGrad))
            print('vars (f,b) = ',f,b)

            #increase or decrease images seen
            #run grad through sigmoid funciton to limit from 0 to 1 (could stretch a little)
            if info.current_epoch == 3:
                prev_data_used = 10
            else:
                prev_data_used = info.dataused[-1]
            data_count = prev_data_used + (sigmoid(avgGrad) * info.alpha)
            print('Data change = ',(sigmoid(avgGrad) * info.alpha))
            #ensure data does not reach 0 or over max data avalible
            if data_count > info.max_data:
                data_count = info.max_data
            if data_count < (0.1*info.max_data):
                data_count = int(0.1*info.max_data)
            
            df_x1 = mod_df.where(mod_df['class']==0).copy()
            df_x2 = mod_df.where(mod_df['class']==1).copy()

            #take the easiest points
            df_x1 = df_x1.sort_values(str(info.current_epoch-1))
            df_x2 = df_x2.sort_values(str(info.current_epoch-1))

            df_x1 = df_x1.head(int(data_count/2))
            df_x2 = df_x2.head(int(data_count/2))

            mod_df = pd.concat([df_x1,df_x2])
            mod_df = mod_df.sample(frac=1).reset_index(drop=True)
                

    else:
        print("ERROR: Incorrect scoring function")

    #form dataset from datafram
    train_x = tf.convert_to_tensor(mod_df[['x1','x2']],dtype='float32')
    train_y = tf.convert_to_tensor(mod_df['class'],dtype='int32')
    train_y = tf.one_hot(train_y,2)
    index = tf.convert_to_tensor(mod_df.index,dtype='int32')

    train_data = tf.data.Dataset.from_tensor_slices((train_x,train_y,index))
    train_data = train_data.shuffle(len(mod_df.index)).batch(batch_size)
    return train_data, mod_df

def logistic_regression(x,w,b):
    return tf.nn.softmax(tf.linalg.matmul(x,w) + b)

def accuracy(y_pred,y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.cast(y_true, tf.int64))
    return tf.cast(correct_prediction, tf.float32)

def save_img_old(mod_df,name,model,e):
    #TODO change model into w and b
    #print(model.trainable_variables)
    w = model.trainable_variables[0].numpy()
    b = model.trainable_variables[1].numpy()

    print(w)
    print(b)

    w1,w2 = w
    c = -b/w2
    m = -w1/w2

    xmin, xmax = -20, 10
    ymin, ymax = -20, 20

    xd = np.array([xmin,xmax])
    yd = m*xd + c

    plt.plot(xd,yd, 'k', lw=1, ls='--')
    plt.fill_between(xd,yd,ymax,color='tab:blue',alpha=0.2)
    plt.fill_between(xd,yd,ymin,color='tab:orange',alpha=0.2)

    plt.scatter(mod_df['x1'].where(mod_df['class']==0),mod_df['x2'].where(mod_df['class']==0),alpha =0.5)
    plt.scatter(mod_df['x1'].where(mod_df['class']==1),mod_df['x2'].where(mod_df['class']==1),alpha =0.5)

    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.savefig(name + str(e) +'.jpg')
    plt.close()

def save_img(mod_df,name,model,e,min,max):
    #https://www.kaggle.com/arthurtok/decision-boundaries-visualised-via-python-plotly
    res = 500
    #take the 2 features
    #TODO add generalisability
    #print(mod_df.head())

    dx = np.linspace(min[0]-1,max[0]+1,res)
    dy = np.linspace(min[1]-1,max[1]+1,res)
    xx,yy = np.meshgrid(dx,dy)

    Z1 = model.predict(np.c_[xx.ravel(),yy.ravel()])[:,:1]
    Z1 = Z1.reshape(xx.shape)

    Z2 = model.predict(np.c_[xx.ravel(),yy.ravel()])[:,1:2]
    Z2 = Z2.reshape(xx.shape)

    fig, (ax1,ax2) = plt.subplots(1,2)

    #plot each heatmap
    ax1.pcolormesh(xx[0], dy,Z1,cmap='PRGn',alpha=0.7)
    ax1.scatter(mod_df[mod_df['class'] == 0]['x1'],mod_df[mod_df['class'] == 0]['x2'])
    ax1.scatter(mod_df[mod_df['class'] == 1]['x1'],mod_df[mod_df['class'] == 1]['x2'])
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Class 1')
 

    im2 = ax2.pcolormesh(xx[0], dy,Z2,cmap='PRGn',alpha=0.7)
    ax2.scatter(mod_df[mod_df['class'] == 0]['x1'],mod_df[mod_df['class'] == 0]['x2'])
    ax2.scatter(mod_df[mod_df['class'] == 1]['x1'],mod_df[mod_df['class'] == 1]['x2'])
    fig.colorbar(im2,ax=ax2)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Class 2')
    fig.tight_layout()

    plt.savefig(name + str(e)+'.jpg')
    plt.close()

    



def run_optimization(x,y,loss_func,optimizer,w,b,e):
    with tf.GradientTape() as tape:
        pred = logistic_regression(x,w,b)
        full_loss = loss_func(y,pred)
        loss = tf.reduce_mean(full_loss,1)
    grads = tape.gradient(loss,[w,b])

    if e > 0:
        optimizer.apply_gradients(zip(grads, [w,b]))
    return full_loss

@tf.function
def run_optimization_model(x,y,model,loss_func,optimizer):
    with tf.GradientTape() as tape:
        pred = model(x,training=True)
        full_loss = loss_func(y,pred)
        loss = tf.reduce_mean(full_loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer[0].apply_gradients(zip(grads, model.trainable_variables))

    return full_loss,loss


def update_loss_info(df,loss,index,epoch):
    for l, i in zip(loss.numpy(), index.numpy()):
        df.at[i,str(epoch)] = l
    return df

def build_model():
    model = tf.keras.Sequential([
        layers.Dense(2, activation= 'softmax')
    ])
    return model
