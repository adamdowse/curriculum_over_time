import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn import linear_model
import os
import csv
import random

def init_data(info,bypass=False):
    '''
    Take a named dataset and if it already exists use the pre svaed data otherwise download it.
    '''
    if os.path.isdir(info.data_path + info.dataset_name) and bypass==False:
        print('INIT: Using found',info.dataset_name, 'data')
        #if the data exists import the csv and change data into a tf dataset
        # need to take the csvs and fix the image data dimentions and turn into a tfdataset
        df = pd.read_csv(info.data_path + info.dataset_name + '/imagedata.csv',index_col='Unnamed: 0')
        df_losses = pd.read_csv(info.data_path + info.dataset_name + '/lossdata.csv',index_col='Unnamed: 0')
        with open(info.data_path+info.dataset_name+'/metadata.csv',newline='') as f:
            reader = csv.reader(f)
            file_content = []
            for row in reader:
                file_content.append(row)
            info.num_classes = int(file_content[0][0])
            info.class_names = file_content[1]
            info.img_shape = file_content[2][0]
            info.img_shape = info.img_shape[1:-1]
            info.img_shape = info.img_shape.split(',')
            info.img_shape = [int(x) for x in info.img_shape]
    else:
        print('INIT: Cannot find ',info.dataset_name, ' data, downloading now...')
        #take the tfds dataset and produce a dataset and dataframe
        ds, ds_info = tfds.load(info.dataset_name,with_info=True,shuffle_files=True,as_supervised=True,split='all')
        df = pd.DataFrame(columns=['img','label','i','test'])
        df_losses = pd.DataFrame(columns=['label','i','test']) #for speed reasons

        #record ds metadata
        info.num_classes = ds_info.features['label'].num_classes
        info.class_names = ds_info.features['label'].names

        #Take the dataset and form a csv with the infomation in it
        i = 0
        for image,label in ds:
            if i == 0:
                info.img_shape = image.shape
            if random.random() > 0.8: test = True 
            else: test = False
            new_row = pd.DataFrame(data={'img':[image.numpy().flatten()], 'label':label.numpy(),'i':i,'test':test},index=[i])
            df = pd.concat([df,new_row],axis=0)
            n_row = pd.DataFrame(data={'label':label.numpy(),'i':i,'test':test},index=[i])
            df_losses = pd.concat([df_losses,n_row],axis=0)
            i += 1
        
        #make the required directory and save the data
        try:
            os.mkdir(info.data_path + info.dataset_name)
        except OSError as error:
            print(error)
        df.to_csv(info.data_path + info.dataset_name + '/imagedata.csv')
        df_losses.to_csv(info.data_path  + info.dataset_name + '/lossdata.csv')
        with open(info.data_path+info.dataset_name+'/metadata.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([info.num_classes])
            writer.writerow([info.class_names])
            writer.writerow([info.img_shape])

    df = df.set_index('i')
    train_df = df[df['test']==False]
    test_df = df[df['test']==True]

    train_df = train_df.drop('test').set_index('i')
    test_df = test_df.drop('test').set_index('i')

    print('INIT: Finished creating train dataset')

    df_train_losses = df_losses[df_losses['test']==False]
    df_train_losses = df_train_losses.drop('test').set_index('i')
    df_train_losses['score'] = np.nan

    return df_train_losses,train_df,test_df,info

#convert from a vector image to 3d representation again and build dataset form dataframe
def img_dim_shift(x,info):
    #x is a vector of strings
    new_x = []
    for img in x:
        img = img.replace('[','')
        img = img.replace(']','')
        img = img.replace('\n','')
        img = img.split(' ')
        img = [y.replace(' ','') for y in img]
        img = [y for y in img if y!= '']
        img = [float(y) for y in img]
        img = [y/256 for y in img]
        img = np.array(img)
        img = img.reshape(info.img_shape)
        new_x.append(img)
    return new_x

def label_oh(x,info):
    return tf.one_hot(x,info.num_classes)

def scoring_func(df,info):
    #df is train_data_losses = (i|0,1,2,3,4...)
    #take the training dataframe and apply the scoring functions to it
    #the output is a dataframe with the score of each index
    def calc_grad(n,data):
        grads = []
        for i,row in data.iterrows():
            x = np.array(range(n)).reshape((-1,1))
            y = row[-n:].to_numpy()
            model = linear_model.LinearRegression().fit(x,y)
            grads.append(model.coef_[0])
        return pd.Series(data=grads)

    #reset the scores
    df['score'] = np.nan

    if info.scoring_function == 'normal':
        #use the normal reshuffling technique
        i = random.shuffle(range(len(df.index)))
        df['score'] = i
        return df
        
    elif info.scoring_function == 'naive_grads':
        #calc gradients over last n losses
        #convert any nans into last loss val
        df_filled = df.copy()
        df_filled = df_filled.iloc[:,1:].fillna(method='ffill',axis=1)

        n = 3 #lookback for gradients
        if info.current_epoch == 0:
            #use all the data but shuffled
            i = random.shuffle(range(len(df.index)))
            df['score'] = i

        elif info.current_epoch == 1:
            #use the first epochs loss info
            df['score'] = df['0']

        elif info.current_epoch < n:
            #calc grad over as many as possible
            n = info.current_epoch
            df['score'] = calc_grad(n,df[:,np.r_[0,-n:]])

        elif info.current_epoch >= n:
            #needs to be n+1 epoch before it can use this
            df['score'] = calc_grad(n,df[:,-n:])


    else:
        print('COLLECT TRAIN DATA: ERROR no valid scoring function')

    return df

def pacing_func(df,info):
    if info.pacing_function == 'none':
        return df

    elif info.pacing_function == 'naive_linear_high_first':
        n = len(df.index)
        df = df.sort_values('score',ascending=True)
        #calc amount of data to use
        d = info.lam_zero + ((1-info.lam_zero)/(info.max_epochs * info.lam_pace))*info.current_epoch #ref from cl survey
        d = min([1,d])
        d = int(d*n)
        i = range(d)
        i = i + [np.nan]*(n-d)
        df['score'] = i
        return df

    elif info.pacing_function == 'naive_linear_low_first':
        n = len(df.index)
        df = df.sort_values('score',ascending=False)
        #calc amount of data to use
        d = info.lam_zero + ((1-info.lam_zero)/(info.max_epochs * info.lam_pace))*info.current_epoch #ref from cl survey
        d = min([1,d])
        d = int(d*n)
        i = range(d)
        i = i + [np.nan]*(n-d)
        df['score'] = i
        return df


def update_col(batch_loss, col, indexes,info):
    #take the batch_losses and update column 
    for i in range(len(batch_loss)):
        #add the index and loss to the column
        new_row = pd.DataFrame(data={'i':indexes[i].numpy(),str(info.current_epoch):batch_loss[i].numpy()},index=[0])
        col = pd.concat([col,new_row],axis=0)
    return col

