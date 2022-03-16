import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import supporting_functions as sf

class Info:
    #build a simple logistic regression model
    num_classes = 2
    num_features = 2
    learning_rate = 0.01
    batch_size = 16
    max_data = 100
    max_test_data = 10
    max_epoch = 200
    data_stddev = 2
    data_random_state = 2
    scoring_func = 'meanlossgradient'
    run_name = 'Data/' + scoring_func + '/OutputLayers/' + scoring_func
    acc = []
    dataused = []
    train_losses = []
    current_epoch = 0
    alpha = 10 #(could vary a lot)
    

info = Info()

#collect data
df = sf.data_init(info.max_data,info.num_classes,info.num_features,stddev=info.data_stddev,random_state=info.data_random_state)
test_df = sf.data_init(info.max_data,info.num_classes,info.num_features,stddev=info.data_stddev,random_state=info.data_random_state)
test_ds, test_df = sf.collect_data(test_df,info,test=True)

max_x = [df.iloc[:,x].max() for x in range(info.num_features)]
min_x = [df.iloc[:,x].min() for x in range(info.num_features)]

#init loss func and optimizer
optimizer = keras.optimizers.SGD(learning_rate=info.learning_rate),
loss_func = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
model = sf.build_model()

#run experiment
for e in range(info.max_epoch):
    print('epoch = '+str(e))
    info.current_epoch = e
    #collect the training data and df based on the scoring and pacing funciton
    train_data, mod_df = sf.collect_data(df,info)
    #record how much data was used
    info.dataused.append(len(mod_df.index))

    #if the first epoch set new data col as nan otherwise take data from previous col
    if e > 0:
        df = pd.concat([df,df[str(e-1)]],axis=1)
        df.columns.values[-1] = str(e)
    else:
        df[str(e)] = np.nan

    #training loop
    for step, (batch_x,batch_y,index) in enumerate(train_data):
        #train the model and return the losses of the batch
        full_loss,_ = sf.run_optimization_model(batch_x,batch_y,model,loss_func,optimizer)
        df = sf.update_loss_info(df,full_loss,index,e)
    if e % 5 == 0:
        sf.save_img(mod_df,info.run_name,model,e,min_x,max_x)
    print('data used = '+str(info.dataused[-1]))

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
        info.acc.append(correct/total)
        print(correct/total)
    else:
        info.acc.append(0)


df.to_csv(info.run_name+'_df')

acc_df = pd.DataFrame(data=info.acc)
acc_df.to_csv(info.run_name+'_test_acc',header=None,index=False)

data_df = pd.DataFrame(data=info.dataused)
data_df.to_csv(info.run_name+'_dataused', header=None,index=False)
