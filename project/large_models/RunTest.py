
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorboard as tb
import supporting_functions as sf
import supporting_models as sm
import datagen
import wandb
import os
import argparse
import time
import sqlite3
from sqlite3 import Error
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments from script')

    parser.add_argument('--max_epochs',type=int,default=30)
    parser.add_argument('--learning_rate',type=float,default=0.01)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--scoring_function',type=str,default='normal')
    parser.add_argument('--pacing_function',type=str,default='none')
    parser.add_argument('--fill_function',type=str,default='ffill')
    parser.add_argument('--dataset',type=str,default='mnist')
    parser.add_argument('--dataset_size',type=float,default=1)
    parser.add_argument('--dataset_similarity',type=int,default=random.randint(1,10000000))
    parser.add_argument('--data_path',type=str,default='none')
    parser.add_argument('--db_path',type=str,default='none')
    parser.add_argument('--save_model_path',type=str,default='none')
    parser.add_argument('--early_stopping', type=int,default=0)

    parser.add_argument('--group', type=str,default=None)
    parser.add_argument('--record_loss',type=str,default='sum')
    parser.add_argument('--batch_logs',type=str,default='False')

    parser.add_argument('--lam_zero',type=float,default=0.1)
    parser.add_argument('--lam_max',type=float,default=0.9)
    parser.add_argument('--lam_lookback',type=int,default=3)
    parser.add_argument('--lam_low_first',type=bool,default=True)
    parser.add_argument('--lam_data_multiplier',type=float,default=1)
    parser.add_argument('--lam_lower_bound',type=float,default=-0.5)
    parser.add_argument('--lam_upper_bound',type=float,default=0.5)

    parser.add_argument('--score_grav',type=float,default=0.1)
    parser.add_argument('--score_lookback',type=int,default=3)

    args = parser.parse_args()
    return args

def main(args):

    config = {
        'max_epochs':args.max_epochs,               #maximum epochs before termination                          
        'learning_rate':args.learning_rate,         #learning rate of updater
        'batch_size':args.batch_size,               #training batch size
        'scoring_function':args.scoring_function,   #scoring function used to rank data
        'pacing_function':args.pacing_function,     #pacing fucntion orders the ranked data or selects a certain amount of data
        'fill_function':args.fill_function,         #fill functions estimate the unseen data that has no current infomation due to not being used
        
        'record_loss':tf.convert_to_tensor(args.record_loss,tf.string), # if 'True' the loss for each image trained on will be recorded in the db
        'record_softmax_error':tf.convert_to_tensor(args.record_loss,tf.string), # if 'True' the softmax error [num_classes] for each image trained on will be recorded in the db
        'batch_logs':args.batch_logs,               # record stats after each training batch

        'early_stopping':args.early_stopping,       #number of epochs of non increasing test accuracy before termination (0 is off)

        #pacing functions
        'lam_zero':args.lam_zero,                    #initial amount of data to use
        'lam_max':args.lam_max,                      #epoch to use full data at
        'lam_lookback':args.lam_lookback,            #how many steps for the regression to lookback
        'lam_low_first':args.lam_low_first,          #False: use the high score values first /True: uses low values first
        'lam_data_multiplier':args.lam_data_multiplier, #multiplied by the gradient to add or remove data from the set
        'lam_lower_bound':args.lam_lower_bound,      #Limits for data useage
        'lam_upper_bound':args.lam_upper_bound,

        #scoring vars
        'score_grav':args.score_grav,               #gravity to reduce regression predictions by
        'score_lookback':args.score_lookback,       #lookback for regression

        'dataset_name':args.dataset,                #if datset name is a path use that path
        'dataset_size':args.dataset_size,           #proportion of the train dataset to use
        'seed':args.dataset_similarity, #random seed number of experiment unless specified    
        'data_path':args.data_path,                 #root of where data is to be stored
        'save_model_path':args.save_model_path,     #root to save trained models
        'db_path':args.db_path,                     #root to save the db from

        'group':args.group,
    }
    class Info_class:
        img_shape = 0
        batch_num = 0
        step = 0
        test_step = 0
        dataused = [] 
        num_classes = 0
        class_names = []
        early_stopping_counter = 0
        early_stopping_max = 0
        current_epoch = 0
        lam_data = 0 #amount of data used

    info = Info_class()


    @tf.function
    def train_step(imgs,labels):
        with tf.GradientTape() as tape:
            preds = model(imgs,training=True)
            batch_loss = loss_func(labels,preds)
            loss = tf.math.reduce_mean(batch_loss)
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer[0].apply_gradients(zip(grads,model.trainable_variables))
        train_loss(loss)
        train_acc_metric(labels,preds)
        return batch_loss, loss, preds

    

    @tf.function
    def test_step(imgs, labels):
        preds = model(imgs, training=False)
        t_loss = loss_func(labels,preds)
        m_loss = tf.math.reduce_mean(t_loss)
        test_loss(m_loss)
        test_acc_metric(labels, preds)
        return preds, t_loss, m_loss

    random.seed(config['seed'])

    #setup database

    #setup convertion functions for storing in db
    sqlite3.register_adapter(np.ndarray, sf.array_to_bin)# Converts np.array to TEXT when inserting
    sqlite3.register_converter("array", sf.bin_to_array) # Converts TEXT to np.array when selecting

    # create a database connection
    #TODO change this is be based on a given dataset
    database =r"/com.docker.devenvironments.code/project/large_models/DBs/mnist.db"
    conn = sf.DB_create_connection(database)

    #check table exists
    curr = conn.cursor()
    curr.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='imgs' ''')
    if curr.fetchone()[0]==1 :
        print('Table exists.')
    else:
        print('Table does not exist, building now...')
        sf.DB_create(conn) #create the db if it does not exist
        info = sf.DB_import_dataset(conn,conifg,info)#download the dataset and add it to the db

    #count amount of data avalible
    curr = conn.cursor()
    curr.execute('''SELECT COUNT(DISTINCT id) FROM imgs WHERE test = 1''')
    test_data_amount = curr.fetchone()[0]
    curr.execute('''SELECT COUNT(DISTINCT id) FROM imgs WHERE test = 0''')
    train_data_amount = curr.fetchone()[0]
    print('Total Stored Test and Train Data: ',test_data_amount,train_data_amount)

    #Perform housekeeping on the db and reset to original state
    #TODO ensure all else is done and okTHIS 
    cur = conn.cursor()
    for i in range(1,train_data_amount+test_data_amount+1):
        cur.execute('''UPDATE imgs SET used = (?) WHERE id = (?)''',(random.random(),i))
        
    cur.execute('''UPDATE imgs SET batch_num = 0''')
    conn.commit()

    #Limit the data for both train and test
    train_data_amount = int(train_data_amount*config['dataset_size'])
    test_data_amount = int(test_data_amount)#TODO ADD SMALLER TEST DATA SIZE
    sf.DB_set_used(conn,True,test_data_amount)
    sf.DB_set_used(conn,False,train_data_amount)

    curr.execute('''SELECT COUNT(DISTINCT id) FROM imgs WHERE test = 1 AND used = 1''')
    test_data_amount = curr.fetchone()[0]
    curr.execute('''SELECT COUNT(DISTINCT id) FROM imgs WHERE test = 0 AND used = 1''')
    train_data_amount = curr.fetchone()[0]
    print('Test and Train data to be used: ',test_data_amount,train_data_amount)
    conn.commit()

    #init the batch_num randomly for first epoch
    #TODO could add different ways of doing this eg. stats based on data
    #create an array of all batch_nums
    sf.DB_random_batches(conn,test=0,img_num=train_data_amount,batch_size=config['batch_size'])

    #Setup logs and records
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    os.environ['WANDB_DISABLED'] = 'true'
    #wandb.login()
    tf.keras.backend.clear_session()

    #Init the data generators
    #TODO make these based on the arguments
    train_data_gen = datagen.CustomDBDataGen(
        conn = conn,
        X_col = 'img',
        Y_col = 'label',
        batch_size = args.batch_size, 
        num_classes = 10,
        input_size = (28,28,1),
        test=0
    )

    test_data_gen = datagen.CustomDBDataGen(
        conn = conn,
        X_col = 'img',
        Y_col = 'label',
        batch_size = args.batch_size, 
        num_classes = 10,
        input_size = (28,28,1),
        test=1
    )
    
    class timer:
        def __init__(self):
            self.t = {}
            self.tic = time.perf_counter()            
        
        def click(self,name):
            toc = time.perf_counter()
            self.t[name] = toc - self.tic
            self.tic = toc

        def print(self):
            print(self.t)


    #build and load model, optimizer and loss functions
    model = sm.Simple_CNN(info.num_classes)
    optimizer = keras.optimizers.SGD(learning_rate=config['learning_rate']),
    loss_func = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)

    #setup metrics to record: [train loss, test loss, train acc, test acc]
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc_metric = keras.metrics.CategoricalAccuracy()

    print('MAIN: Started Training')
    info.step = 0
    info.test_step = 0
    for info.current_epoch in range(config['max_epochs']):
        #training step
        info.batch_num = 0
        for i, (X,Y) in enumerate(train_data_gen):

            #print batch num
            if i % 500 == 0: print("Batch ="+str(i))
            
            #collect losses and train model
            batch_loss, mean_loss, preds = train_step(X[1],Y)

            #count number of each class in the batch
            label_corrected = np.array([np.argmax(x) for x in Y])
            #TODO make based on args
            class_counts = [np.count_nonzero(label_corrected == x) for x in range(10)]

            #update the db with the retrieved info
            sf.DB_update(conn,info,info.current_epoch,X,Y,batch_loss,preds)

            #record the batch by batch logs
            if config['batch_logs'] == 'True':
                #Run the test dataset to assess the training batch update
                for X,Y in test_data_gen:
                    preds, batch_test_loss, t_loss = test_step(X[1],Y)
                    sf.DB_update(conn,info,info.test_step,X,Y,batch_loss,preds)
                
                info.test_step += 1
                #TODO func needs completing
                sf.log_batch_test()
                
                #reset the test metrics so no clashes
                test_loss.reset_states()
                test_acc_metric.reset_states()

            info.batch_num += 1
            info.step += 1

        #END OF EPOCH
        #log info based on the losses or outputs
        sf.log_loss_stats()

        #calculate the score via a function

        #take score stats

        #rank and trim data

        #take rank statistics

        #run end of epoch updating

        #test steps
        for X,Y in test_data_gen:
            test_step(X[1],Y)

        #log test epoch
        sf.log #TODO
        
        #Printing to screen
        print('Epoch ',info.current_epoch+1,', Loss: ',train_loss.result().numpy(),', Accuracy: ',train_acc_metric.result().numpy(),', Test Loss: ',test_loss.result().numpy(),', Test Accuracy: ',test_acc_metric.result().numpy())
        
        #early stopping
        #TODO this needs checking 
        if info.early_stopping > 0:
            #increment if test acc lower
            if test_acc_metric.result().numpy() < info.early_stopping_max:
                info.early_stopping_counter += 1
            else:
                info.early_stopping_max = test_acc_metric.result().numpy()

            if info.early_stopping_counter >= info.early_stopping:
                print('EARLY STOPPING REACHED: MAX TEST ACC = ',info.early_stopping_max)
                break

        #reset the metrics
        train_loss.reset_states()
        train_acc_metric.reset_states()
        test_loss.reset_states()
        test_acc_metric.reset_states()

        #save the model
        if info.current_epoch % 10 == 0:
            model.save(info.save_model_path)
            print('Checkpoint saved')

    #save the model and data
    model.save(info.save_model_path)
    df_train_losses.to_csv(info.data_path + info.dataset_name + '/normal_loss_info.csv')



if __name__ =='__main__':
    args = parse_arguments()
    main(args)

