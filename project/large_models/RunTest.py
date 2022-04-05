import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorboard as tb
import supporting_functions as sf
import supporting_models as sm
import datagen
import wandb
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments from script')

    parser.add_argument('--max_epochs',type=int,default=30)
    parser.add_argument('--learning_rate',type=float,default=0.01)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--scoring_function',type=str,default='normal')
    parser.add_argument('--pacing_function',type=str,default='none')
    parser.add_argument('--dataset',type=str,default='mnist')
    parser.add_argument('--lam_zero',type=float,default=0.1)
    parser.add_argument('--lam_pace',type=float,default=0.9)

    args = parser.parse_args()
    return args

def main(args):
    #TODO - Add dataused each epoch and save the array
    #TODO - Implement split data csv
    class Info_class :
        #TODO remove this and use wadnb config
        #variables for test
        max_epochs = args.max_epochs
        learning_rate = args.learning_rate
        batch_size = args.batch_size
        scoring_function = args.scoring_function
        pacing_function = args.pacing_function
        current_epoch = 0

        lam_zero = args.lam_zero
        lam_pace = args.lam_pace

        #if datset name is a path use that path
        dataset_name = args.dataset
        data_path = '/com.docker.devenvironments.code/project/large_models/datasets/'
        save_model_path = '/com.docker.devenvironments.code/project/large_models/saved_models/'

        img_shape = 0
        dataused = [] 
        class_names = []

    info = Info_class()

    config = {
        'epochs':args.max_epochs,
        'learning_rate':args.learning_rate,
        'batch_size':args.batch_size,
        'scoring_func':args.scoring_function,
        'pacing_func':args.pacing_function,
        'dataset':args.dataset,
        'lam_zero':args.lam_zero,
        'lam_pace':args.lam_pace
    }
    wandb.login()
    wandb.init(project='curriculum_over_time',entity='adamdowse',config=config)
    tf.keras.backend.clear_session()

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
        return batch_loss

    @tf.function
    def test_step(imgs, labels):
        preds = model(imgs, training=False)
        t_loss = loss_func(labels,preds)
        m_loss = tf.math.reduce_mean(t_loss)
        test_loss(m_loss)
        test_acc_metric(labels, preds)

    # initilise the dataframe to train on and the test dataframe
    df_train_losses, train_df, test_df, info = sf.init_data(info)
    #df_train_losses is a df of just training set without images, (i|label,score) used to record losses
    #train_df is the df with images in it (i|img,label)
    #test_df is the df with images in it (i|img,label)

    #Init the data generators
    train_data_gen = datagen.CustomDataGen(
        df = train_df,
        X_col = 'img',
        Y_col = 'label',
        batch_size = args.batch_size, #change this
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
    for info.current_epoch in range(info.max_epochs):
        #create the column for losses
        col = pd.DataFrame(columns=['i',str(info.current_epoch)])

        #training step
        s = []
        for i, (X,Y) in enumerate(train_data_gen):
            #collect losses and train model
            if i % 100 == 0: print("Batch ="+str(i))
            batch_loss = train_step(X[1],Y)
            #create a dataframe of the single column
            col = sf.update_col(batch_loss,col,X[0],info) # = (i,current_epoch)
            s = [X,Y]
        print(s)
        prn()
        #add the col to the loss holder
        col = col.set_index('i') #col = (i|current_epoch)
        df_train_losses = pd.concat([df_train_losses,col],axis=1) # = (i|label,score,0,1,2,..,current_epoch)(nan where data not used)

        #TODO calc data to use next epoch NEED TO MAKE SURE THE INDEXING IS RIGHT AND TAKE THIS INTO THE GEN

        df_train_losses = sf.scoring_func(df_train_losses,info) # =(i|score,0,1,2...)
        df_train_losses = sf.pacing_func(df_train_losses,info) # change the score to a rank and nan for not used
        train_data_gen.on_epoch_end(df_train_losses)

        #test steps
        for X,Y in test_data_gen:
            test_step(X[1],Y)

        wandb.log({
            'Epoch':info.current_epoch,
            'Train-Loss':train_loss.result().numpy(),
            'Test-Loss':test_loss.result().numpy(),
            'Train-Acc':train_acc_metric.result().numpy(),
            'Test-Acc':test_acc_metric.result().numpy(),
            'Data-Used':train_data_gen.dataused,
            '0-Used':train_data_gen.class_used[0],
            '1-Used':train_data_gen.class_used[1],
            '2-Used':train_data_gen.class_used[2],
            '3-Used':train_data_gen.class_used[3],
            '4-Used':train_data_gen.class_used[4],
            '5-Used':train_data_gen.class_used[5],
            '6-Used':train_data_gen.class_used[6],
            '7-Used':train_data_gen.class_used[7],
            '8-Used':train_data_gen.class_used[8],
            '9-Used':train_data_gen.class_used[9],})
        
        
        #Printing to screen
        print('Epoch ',info.current_epoch+1,', Loss: ',train_loss.result().numpy(),', Accuracy: ',train_acc_metric.result().numpy(),', Test Loss: ',test_loss.result().numpy(),', Test Accuracy: ',test_acc_metric.result().numpy())
        
        #reset the metrics
        train_loss.reset_states()
        train_acc_metric.reset_states()
        test_loss.reset_states()
        test_acc_metric.reset_states()

        #save the model
        if info.current_epoch % 5 == 0: #TODO change back to 10 for full test
            model.save(info.save_model_path)
            print('Checkpoint saved')

    #save the model and data
    #np.savetxt(info.data_path + info.dataset_name + '/dataused.csv',dataused)
    model.save(info.save_model_path)
    df_train_losses.to_csv(info.data_path + info.dataset_name + '/normal_loss_info.csv')


if __name__ =='__main__':
    args = parse_arguments()
    main(args)