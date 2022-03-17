import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import supporting_functions as sf

#Run the logistic regression model on more high dim data (more features)
#train on a curriculum 
#predict next loss based on infomation availible ie curriculum
#record all of the losses but dont train on all
#compare the predicted to the true losses

class Info:
    #build a simple logistic regression model
    num_classes = 3
    num_features = 3
    learning_rate = 0.01
    batch_size = 16
    max_data = 100
    max_test_data = 10
    max_epoch = 100
    data_stddev = 2
    data_random_state = 2
    scoring_func = 'rank'
    run_name = 'Data/' + scoring_func + '/OutputLayers/' + scoring_func
    acc = []
    dataused = []
    train_losses = []
    current_epoch = 0
    alpha = 10 #(could vary a lot)
    lookback = 3
    

info = Info()

#collect data
#this is the dataframe to train with and record the losses of trained on data
df = sf.data_init(info)
#this is the datasframe to record all losses and not to be trained on
df_hidden = df.copy()
a = info.scoring_func
info.scoring_func = 'normal'
hidden_ds,_ = sf.collect_data(df_hidden,info)
info.scoring_func = a
#test data
test_df = sf.data_init(info)
test_ds, test_df = sf.collect_data(test_df,info,test=True)

max_x = [df.iloc[:,x].max() for x in range(info.num_features)]
min_x = [df.iloc[:,x].min() for x in range(info.num_features)]

#init loss func and optimizer
optimizer = keras.optimizers.SGD(learning_rate=info.learning_rate),
loss_func = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
model = sf.build_model(info.num_classes)

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

    #fake training loop to save the true losses
    for step, (batch_x,batch_y,index) in enumerate(hidden_ds):
        l = sf.run_optimization_model_noupdate(batch_x,batch_y,model,loss_func)
        df_hidden = sf.update_loss_info(df_hidden,l,index,e)

    #predict next loss
    if e == info.lookback:
        #create the prediction df
        pred_df = pd.DataFrame(data=sf.nmeanpred(df.iloc[:,-info.lookback:]),columns=[str(e+1)])
    elif e > info.lookback:
        pred_df = pd.concat([pred_df,pd.DataFrame(data=sf.nmeanpred(df.iloc[:,-info.lookback:]),columns=[str(e+1)])],axis=1)


    #real training loop
    for step, (batch_x,batch_y,index) in enumerate(train_data):
        #train the model and return the losses of the batch
        full_loss,_ = sf.run_optimization_model(batch_x,batch_y,model,loss_func,optimizer)
        df = sf.update_loss_info(df,full_loss,index,e)
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

pred_df.to_csv(info.run_name+'_preddf')
df_hidden.to_csv(info.run_name+'_fulldf')
df.to_csv(info.run_name+'_df')

acc_df = pd.DataFrame(data=info.acc)
acc_df.to_csv(info.run_name+'_test_acc',header=None,index=False)

data_df = pd.DataFrame(data=info.dataused)
data_df.to_csv(info.run_name+'_dataused', header=None,index=False)
