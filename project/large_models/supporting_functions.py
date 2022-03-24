import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylot as plt
import tensorflow_datasets as tfds
import os
import csv
import random



def init_data(info,bypass=False):
    '''
    Take a named dataset and if it already exists use the pre svaed data otherwise download it.
    '''
    if os.path.isdir(info.data_path + info.dataset_name) and bypass=False:
        print('INIT: Using found ',info.dataset_name, ' data')
        #if the data exists import the csv and change data into a tf dataset
        # need to take the csvs and fix the image data dimentions and turn into a tfdataset
        df = pd.read_csv(info.data_path + info.dataset_name + '/imagedata.csv')
        df_losses = pd.read_csv(info.data_path + info.dataset_name + '/lossdata.csv')
        with open(info.data_path+info.dataset_name+'/metadata.csv',newline=' ') as f:
            reader = csv.reader(f,delimiter=' ')
            info.num_classes = reader[0]
            info.class_names = reader[1]
            info.img_shape = reader[2]
        
    else:
        print('INIT: Cannot find ',info.dataset_name, ' data, downloading now...')
        #take the tfds dataset and produce a dataset and dataframe
        ds, ds_info = tfds.load(info.dataset_name,with_info=True,shuffle_files=True,as_supervised=True)
        df = pd.DataFrame(columns=['img','label','i','test'])
        df_losses = pd.DataFrame(columns=['label','i','test']) #for speed reasons

        #record ds metadata
        info.num_classes = ds_info.features['label'].num_classes
        info.class_names = ds_info.features['label'].names

        #Take the dataset and form a csv with the infomation in it
        i = 0
        for image, label in ds:
            if i == 0:
                info.img_shape = image.shape()
            if random.random() > 0.8: test = True else test = False
            df = pd.concat([df,[np.ravel(image.to_numpy()),label,i,test]],axis=0)
            df_losses = pd.concat([df_losses,[label,i,test]],axis=0)
            i += 1
        
        #make the required directory and save the data
        os.mkdir(info.data_path + info.dataset_name)
        df.to_csv(info.data_path + info.dataset_name + '/imagedata.csv')
        df_losses.to_csv(info.data_path  + info.dataset_name + '/lossdata.csv')
        with open(info.data_path+info.dataset_name+'/metadata.csv','w',newline=' ') as f:
            writer = csv.writer(f)
            writer.writerow(info.num_classes)
            writer.writerow(info.class_names)
            writer.writerow(info.img_shape)

    #convert from a vector image to 3d representation again and build dataset form dataframe
    def img_dim_shift(x):
        #x is a vector
        new_x = []
        for img in x:
            new_x.append(img.reshape(img_shape))
        return new_x

    train_df = df.where(df['test']==False)
    train_ds = tf.data.Dataset.from_tensor_slices((img_dim_shift(train_df['image'].values),train_df['label'].values,train_df['i'].values))
    print('INIT: Finished creating train dataset')
    test_df = df.where(df['test']==True)
    test_ds = tf.data.Dataset.from_tensor_slices((img_dim_shift(test_df['image'].values),test_df['label'].values,test_df['i'].values))
    print('INIT: Finished creating testrain dataset')

    df_train_losses = df_losses.where(df_losses['test']==False)
    return train_ds,test_ds,df_train_losses,info

