import pandas as pd
import supporting_functions as sf


#AIMS:
# Try the predictive loss method on cifar 10 dataset and compare the results
# Setup the test bed to do multiple tests
# make variables generalisable



class Info_class :
    #variables for test
    max_epochs = 100
    current_epoch = 0

    #if datset name is a path use that path
    dataset_name = 'mnist'
    data_path = '/'#TODO
    save_model_path = '/'#TODO
    log_path =  ''#TODO

    img_shape = 0
    dataused = [] 
    class_names = []

    

info = Info_class()

# initilise the dataframe to train on and the test dataframe
train_ds, test_ds,df_train_losses,info = sf.init_data(info)

#build and load model, optimizer and loss functions
#TODO start here
#setup metrics to record: [train loss, test loss, train acc, test acc, dataused]

#for epochs

    #collect train dataset from the dataset via a scoring function

    #training step
        #collect losses and train model
        #update the original dataframe

    #update tensorboard

    #test steps

    #update tensorboard

    #reset metrics

    #every n save model