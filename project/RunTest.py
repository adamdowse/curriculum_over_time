import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import supporting_functions as sf


def run_optimization_model(x,y):
    with tf.GradientTape() as tape:
        pred = model(x,training=True)
        print(pred)
        print(y)
        full_loss = loss_func(y,pred)
        print(full_loss)
        loss = tf.reduce_mean(full_loss)
    grads = tape.gradient(loss, model.trainable_variables)
    print('grads='+ str(grads))
    optimizer[0].apply_gradients(zip(grads, model.trainable_variables))

    return full_loss,loss

#build a simple logistic regression model
num_classes = 2
num_features = 2
learning_rate = 0.01
batch_size = 8
n = 1000
test_n = 100
max_epoch = 100

#collect data
df = sf.data_init(n,num_classes,num_features,stddev=4,random_state=1)
test_df = sf.data_init(n,num_classes,num_features,stddev=4,random_state=1)
test_ds, test_df = sf.collect_data(test_df,test_n,batch_size=1)

#init loss func and optimizer
optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
loss_func = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
model = sf.build_model()

#run experiment
acc = []
dataused = []
train_losses = []

for e in range(max_epoch):
    print('epoch = '+str(e))
    
    #collect the training data and df based on the scoring and pacing funciton
    train_data, mod_df = sf.collect_data(df,n,batch_size,e=e,max_e=max_epoch,type='rank')
    #record how much data was used
    dataused.append(len(mod_df.index))

    #if the first epoch set new data col as nan otherwise take data from previous col
    if e >0:
        df[str(e)] = df[str(e-1)].copy()
    else:
        df[str(e)] = np.nan
    
    #training loop
    c = 0
    for step, (batch_x,batch_y,index) in enumerate(train_data):
        #increment the amount of data used
        c+=len(index)
        #train the model and return the losses of the batch
        full_loss,_ = run_optimization_model(batch_x,batch_y)
        df = sf.update_loss_info(df,full_loss,index,e)

    sf.save_img(mod_df,'rank_neive',model,e)
    print('data used = '+str(c))

    correct = 0
    total = 0
    #loop through test data in batches of 1
    for step ,(batch_x,batch_y,index) in enumerate(test_ds):
        pred = model.predict(batch_x)
        if tf.argmax(pred,1).numpy() == tf.argmax(batch_y,1).numpy():
            correct+=1
        total+=1
        
    if correct > 0:
        acc.append(correct/total)
        print(correct/total)
    else:
        acc.append(0)

    if e == 50:
        sf.save_img(test_df,'rank_test_50',model,e)

df.to_csv('rank_df')

acc_df = pd.DataFrame(data=acc)
acc_df.to_csv('rank_test_acc',header=None,index=False)

data_df = pd.DataFrame(data=dataused)
data_df.to_csv('rank_dataused', header=None,index=False)
