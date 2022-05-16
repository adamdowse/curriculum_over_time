
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
    parser.add_argument('--dataset_similarity',type=str,default='True')
    parser.add_argument('--data_path',type=str,default='none')
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
    #TODO - Implement split data csv
    class Info_class :
        #TODO remove this and use wadnb config
        #variables for test
        max_epochs = args.max_epochs
        learning_rate = args.learning_rate
        batch_size = args.batch_size
        scoring_function = args.scoring_function
        pacing_function = args.pacing_function
        fill_function = args.fill_function
        
        record_loss = args.record_loss
        record_loss = tf.convert_to_tensor(record_loss,tf.string)
        batch_logs = args.batch_logs

        #early stopping
        early_stopping = args.early_stopping #number
        early_stopping_counter = 0
        early_stopping_max = 0
        current_epoch = 0

        #pacing vars
        lam_data = 0 #amount of data used
        lam_zero = args.lam_zero #initial amount of data
        lam_max = args.lam_max #epoch to use full data at
        lam_lookback = args.lam_lookback #regression lookback
        lam_low_first = args.lam_low_first #use the high score values first (True uses low values first)
        lam_data_multiplier = args.lam_data_multiplier #multiplied by the gradient to add or remove data from the set
        lam_lower_bound = args.lam_lower_bound
        lam_upper_bound = args.lam_upper_bound

        #scoring vars
        score_grav = args.score_grav #gravity to reduce regression predictions by
        score_lookback = args.score_lookback #lookback for regression

        #if datset name is a path use that path
        dataset_name = args.dataset
        dataset_size = args.dataset_size #proportion of dataset to use
        dataset_similarity = args.dataset_similarity #true uses the same section of the dataset each time, false randomly shuffles.
        
        data_path = args.data_path
        save_model_path = args.save_model_path
        
        img_shape = 0
        dataused = [] 
        class_names = []

    info = Info_class()
    #TODO update config file for new vars
    config = {
        'learning_rate':args.learning_rate,
        'batch_size':args.batch_size,
        'scoring_func':args.scoring_function,
        'fill_func':args.fill_function,
        'pacing_func':args.pacing_function,
        'dataset':args.dataset,
        'dataset_size':args.dataset_size,
        'dataset_similarity':args.dataset_similarity,
        'early_stopping':args.early_stopping,
        'lam_zero':args.lam_zero,
        'lam_max':args.lam_lookback,
        'lam_low_first':args.lam_low_first,
        'lam_data_multiplier':args.lam_data_multiplier,
        'lam_lower_bound':args.lam_lower_bound,
        'lam_upper_bound':args.lam_upper_bound,
        'score_grav':args.score_grav,
        'score_lookback':args.score_lookback,

    }


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
        if tf.equal(info.record_loss,tf.constant('sum',tf.string)):
            return batch_loss, loss
        else:
            return preds, loss
    

    @tf.function
    def test_step(imgs, labels):
        preds = model(imgs, training=False)
        t_loss = loss_func(labels,preds)
        m_loss = tf.math.reduce_mean(t_loss)
        test_loss(m_loss)
        test_acc_metric(labels, preds)
        return t_loss, m_loss


    #Setup logs and records
    os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
    #os.environ['WANDB_DISABLED'] = 'true'
    wandb.login()
          
    wandb.init(project='curriculum_over_time',entity='adamdowse',config=config,group=args.group)
    tf.keras.backend.clear_session()

    # initilise the dataframe to train on and the test dataframe
    df_train_losses,df_test_losses, train_df, test_df, info = sf.init_data(info)
    #df_train_losses is a df of just training set without images, (i|label,score) used to record losses
    #train_df is the df with images in it (i|img,label)
    #test_df is the df with images in it (i|img,label)

    #Sort class_name var into list
    info.class_names = sf.str_to_list(info.class_names[0])
    info.class_names = [y.replace(',','') for y in info.class_names]
    info.class_names = [y.replace("'",'') for y in info.class_names]
    print('class names: ',info.class_names)


    #Init the data generators
    train_data_gen = datagen.CustomDataGen(
        df = train_df,
        X_col = 'img',
        Y_col = 'label',
        batch_size = args.batch_size, 
        input_size = (28,28,1),
        test=False
    )
    test_data_gen = datagen.CustomDataGen(
        df = test_df,
        X_col = 'img',
        Y_col = 'label',
        batch_size = args.batch_size,
        input_size = (28,28,1),
        test=True
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

    tim = timer()


    #build and load model, optimizer and loss functions
    model = sm.Simple_CNN(info.num_classes)
    optimizer = keras.optimizers.SGD(learning_rate=info.learning_rate),
    loss_func = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)

    #setup metrics to record: [train loss, test loss, train acc, test acc]
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc_metric = keras.metrics.CategoricalAccuracy()

    print('MAIN: Started Training')
    batch_num = 0
    for info.current_epoch in range(info.max_epochs):
        #create the column for losses
        col = pd.DataFrame(columns=['i',str(info.current_epoch)])

        #training step
        for i, (X,Y) in enumerate(train_data_gen):
            #collect losses and train model
            if i % 500 == 0: print("Batch ="+str(i))
            batch_loss, mean_loss = train_step(X[1],Y)
            #create a dataframe of the single column
            col = sf.update_col(batch_loss,Y,col,X[0],info) # = (i,current_epoch)
            
            #record the batch by batch logs
            if info.batch_logs == 'True':
                t_col = pd.DataFrame(columns=['i',str(info.current_epoch)])
                for X,Y in test_data_gen:
                    batch_test_loss, test_loss = test_step(X[1],Y)
                    t_col = sf.update_col(batch_test_loss,Y,t_col,X[0],info)
                wandb.log({
                    'batch_train_loss':mean_loss, 
                    'batch_num':batch_num, 
                    'batch_test_loss':batch_test_loss,
                    'batch_test_acc':test_acc_metric.result().numpy()})
                batch_num += 1

        #add the col to the loss holder
        col = col.set_index('i') #col = (i|current_epoch)
        df_train_losses = pd.concat([df_train_losses,col],axis=1) # = (i|label,score,0,1,2,..,current_epoch)(nan where data not used)

        #grab statistics before nans are infilled
        #class_loss_avg = [df_train_losses[df_train_losses.label==x].iloc[:,-1].mean() for x in range(train_data_gen.num_classes)]
        #class_loss_var = [df_train_losses[df_train_losses.label==x].iloc[:,-1].var() for x in range(train_data_gen.num_classes)]

        #calculate the score via a function
        df_train_losses = sf.scoring_func(df_train_losses,info) # =(i|score,0,1,2...)

        #take score stats
        class_score_avg = [df_train_losses[df_train_losses.label==x].score.mean() for x in range(train_data_gen.num_classes)]
        class_score_var = [df_train_losses[df_train_losses.label==x].score.var() for x in range(train_data_gen.num_classes)]

        #rank and trim data
        df_train_losses = sf.pacing_func(df_train_losses,info) # change the score to a rank and nan for not used

        #take rank statistics
        class_rank_avg = [df_train_losses[df_train_losses.label==x].score.mean() for x in range(train_data_gen.num_classes)]
        class_rank_var = [df_train_losses[df_train_losses.label==x].score.var() for x in range(train_data_gen.num_classes)]

        #run end of epoch updating
        train_data_gen.on_epoch_end(df_train_losses)

        #test steps
        for X,Y in test_data_gen:
            test_step(X[1],Y)

        if info.batch_logs == 'True':
            basic = {}
        else:
            basic = {
                'Epoch':info.current_epoch,
                'Train-Loss':train_loss.result().numpy(),
                'Test-Loss':test_loss.result().numpy(),
                'Train-Acc':train_acc_metric.result().numpy(),
                'Test-Acc':test_acc_metric.result().numpy(),
                'Data-Used':train_data_gen.dataused}
        keys = [x for x in info.class_names]
        cla = {}
        clv = {}
        csa = {}
        csv = {}
        cra = {}
        crv = {}
        for i in range(len(keys)):
            #cla['Loss-Avg-'+keys[i]] = class_loss_avg[i]
            #clv['Loss-Var-'+keys[i]] = class_loss_var[i]
            csa['Score-Avg-'+keys[i]] = class_score_avg[i]
            csv['Score-Var-'+keys[i]] = class_score_var[i]
            cra['Rank-Avg-'+keys[i]] = class_rank_avg[i]
            crv['Rank-Var'+keys[i]] = class_rank_var[i]

        wandb.log({**basic,**cla,**clv,**csa,**csv,**cra,**crv})
        
        
        #Printing to screen
        print('Epoch ',info.current_epoch+1,', Loss: ',train_loss.result().numpy(),', Accuracy: ',train_acc_metric.result().numpy(),', Test Loss: ',test_loss.result().numpy(),', Test Accuracy: ',test_acc_metric.result().numpy())
        
        #early stopping
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
        tim.click('logging')
        #save the model
        if info.current_epoch % 10 == 0:
            model.save(info.save_model_path)
            print('Checkpoint saved')
        tim.print()

    #save the model and data
    #np.savetxt(info.data_path + info.dataset_name + '/dataused.csv',dataused)
    model.save(info.save_model_path)
    df_train_losses.to_csv(info.data_path + info.dataset_name + '/normal_loss_info.csv')



if __name__ =='__main__':
    args = parse_arguments()
    main(args)

