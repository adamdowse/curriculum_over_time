import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt

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

def collect_data(df,n,batch_size,e=1,max_e=1,type='normal'):
    '''
    Takes the dataframe and produces a tf.dataset and an updated df
    rank = neive ranking sorting df by last loss values and taking a percentage based on the pacing function
    '''
    if type == 'normal':
        mod_df = df.copy()

    elif type == 'rank':
        mod_df = df.copy()
        if e>0:
            epoch_percent = (e+1)/(max_e-10)
            print('epoch percent = ',epoch_percent)

            mod_df = mod_df.sort_values(str(e-1))
            mod_df = mod_df.head(int(epoch_percent*len(df.index)))

            '''
            df_x1 = df.where(df['class']==0).copy()
            df_x2 = df.where(df['class']==1).copy()

            #take the easiest points
            df_x1 = df_x1.sort_values('x1')
            df_x2 = df_x2.sort_values('x2')

            df_x1 = df_x1.head(int(epoch_percent*100))
            df_x2 = df_x2.head(int(epoch_percent*100))

            mod_df = pd.concat([df_x1,df_x2])
            '''
    else:
        print("ERROR: Incorrect scoring function")

    #form dataset from datafram
    train_x = tf.convert_to_tensor(mod_df[['x1','x2']],dtype='float32')
    train_y = tf.convert_to_tensor(mod_df['class'],dtype='int32')
    train_y = tf.one_hot(train_y,2)
    index = tf.convert_to_tensor(mod_df.index,dtype='int32')

    train_data = tf.data.Dataset.from_tensor_slices((train_x,train_y,index))
    train_data = train_data.shuffle(n).batch(batch_size)
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
