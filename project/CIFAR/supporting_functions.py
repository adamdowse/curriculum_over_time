import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylot as plt
import tensorflow_datasets as tfds



def init_data(info):
    if info.dataset_name.startwith('/'):
        #take the downloaded data and use that
        #TODO
        # need to take the csvs and fix the image data dimentions and turn into a tfdataset
        a =1####
    else:
        #take the tfds dataset and produce a dataset and dataframe
        ds, ds_info = tfds.load(info.dataset_name,with_info=True,shuffle_files=True,as_supervised=True)
        df = pd.DataFrame(columns=['img','label','i'])
        df_losses = pd.DataFrame(columns=['label','i']) #for speed reasons
        #record ds metadata
        info.num_classes = ds_info.features['label'].num_classes
        info.class_names = ds_info.features['label'].names

        #Take the dataset and form a csv with the infomation in it
        i = 0
        for image, label in ds:
            df = pd.concat([df,[np.ravel(image.to_numpy()),label,i]],axis=0)
            df_losses = pd.concat([df_losses,[label,i]],axis=0)
            i += 1

        df.to_csv(info.data_path + '/' + info.dataset_name + '/imagedata.csv')
        df_losses.to_csv(info.data_path + '/' + info.dataset_name + '/lossdata.csv')

    ds = tf.data.Dataset.from_tensor_slices((df['image'],df['label'],df['i']))
    #TODO add the conversion from ravel to 
    return 