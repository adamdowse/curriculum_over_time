import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
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

    train_df = df[df['test']==False]
    print(train_df)
    train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(img_dim_shift(train_df['img'].values,info),'float32'),label_oh(train_df['label'].values,info),train_df['i'].values))
    print('INIT: Finished creating train dataset')
    test_df = df[df['test']==True]
    test_ds = tf.data.Dataset.from_tensor_slices((tf.cast(img_dim_shift(test_df['img'].values,info),'float32'),label_oh(test_df['label'].values,info),test_df['i'].values))
    print('INIT: Finished creating test dataset')

    df_train_losses = df_losses[df_losses['test']==False]

    return train_ds,test_ds,df_train_losses,info

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


def collect_train_data(train_df,info):
    #take the training dataframe and apply the scoring functions to it
    #the output is a tf dataset with an ordered crric 

    if info.scoring_function == 'normal':
            train_ds = tf.data.Dataset.from_tensor_slices((
                img_dim_shift(train_df['img'].values),
                train_df['label'].values,
                train_df['i'].values
            ))
    #add more here if needed
    else:
        print('COLLECT TRAIN DATA: ERROR no valid scoring function')

    return train_ds

def update_col(batch_loss, col, batch,info):
    #take the batch_losses and update column 
    imgs,labs,inds = batch
    for i in range(len(batch_loss)):
        #add the index and loss to the column
        new_row = pd.DataFrame(data={'i':inds[i].numpy(),str(info.current_epoch):batch_loss[i].numpy()},index=[0])
        col = pd.concat([col,new_row],axis=0)
    return col

def update_df(col,train_df):
    #add the column to the large dataframe
    #train_df = train_df.merge(col,left_on='i',right_on='i')
    train_df = pd.DataFrame.merge(train_df,col,on='i')
    print(train_df)
    #train_df = pd.concat([train_df,col],axis=1)
    return train_df