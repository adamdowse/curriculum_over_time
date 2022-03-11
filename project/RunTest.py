import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import supporting_functions as sf


#build a simple logistic regression model
num_classes = 2
num_features = 2
learning_rate = 0.01
batch_size = 16
n = 500
test_n = 50
max_epoch = 51
data_stddev = 2
data_random_state = 2
scoring_func = 'normal'
run_name = 'Data/normal/normal'

#collect data
df = sf.data_init(n,num_classes,num_features,stddev=data_stddev,random_state=data_random_state)
test_df = sf.data_init(n,num_classes,num_features,stddev=data_stddev,random_state=data_random_state)
test_ds, test_df = sf.collect_data(test_df,test_n,batch_size=1)

max_x = [df.iloc[:,x].max() for x in range(num_features)]
min_x = [df.iloc[:,x].min() for x in range(num_features)]

#init loss func and optimizer
optimizer = keras.optimizers.SGD(learning_rate=learning_rate),
loss_func = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
model = sf.build_model()

#run experiment
acc = []
dataused = []
train_losses = []

for e in range(max_epoch):
    print('epoch = '+str(e))
    
    #collect the training data and df based on the scoring and pacing funciton
    train_data, mod_df = sf.collect_data(df,n,batch_size,e=e,max_e=max_epoch,type=scoring_func)
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
        full_loss,_ = sf.run_optimization_model(batch_x,batch_y,model,loss_func,optimizer)
        df = sf.update_loss_info(df,full_loss,index,e)
    if e % 5 ==0:
        sf.save_img(mod_df,run_name,model,e,min_x,max_x)
    print('data used = '+str(c))

    correct = 0
    total = 0
    #TODO 
    #add batching to speed up
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
        sf.save_img(test_df,run_name+'_test_50',model,e)

df.to_csv(run_name+'_df')

acc_df = pd.DataFrame(data=acc)
acc_df.to_csv(run_name+'_test_acc',header=None,index=False)

data_df = pd.DataFrame(data=dataused)
data_df.to_csv(run_name+'_dataused', header=None,index=False)
