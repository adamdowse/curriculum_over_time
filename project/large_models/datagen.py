import tensorflow as tf
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
import supporting_functions as sf
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, Y_col,
                 batch_size,
                 input_size=(28, 28, 1),
                 test=False):
        
        self.df = df.copy() #[i|img,label] use here is a random index
        self.df['score'] = np.nan #[i|img,label,score]
        self.df = self.df.sample(frac=1)
        self.X_col = X_col #img
        self.Y_col = Y_col #label
        self.batch_size = batch_size
        self.input_size = input_size
        self.test = test
        #add the other key vars here
        #also add stats holding info
        self.dataused = len(self.df.index)
        self.num_classes = df[Y_col].nunique()
        #holds the amount of data used on the next epoch 
        self.class_used = [len(self.df[self.df['label']==x].index) for x in range(self.num_classes)]
        self.class_score = [0]*self.num_classes
        self.class_loss_avg = [0]*self.num_classes
    
    def on_epoch_end(self,losses_df):
        #losses_df = (i|label,score,0,1,2,..) where use is from 0 to dataused and nan past dataused
        #df = (i|img,label,score)
        #update the score col in df
        self.df.score = np.nan
        self.df.update(losses_df) #this is a problem

        #sort the large dataframe by the use index and put nans last so not used
        self.df = self.df.sort_values('score',na_position='last')
 
        self.dataused = len(self.df.index)-self.df.score.isna().sum()

        #produce some statistics
        self.class_used = [len(self.df[self.df.label==x].index)-self.df.score[self.df.label==x].isna().sum() for x in range(self.num_classes)]
        print(self.class_used)



    def __get_input(self, img, img_shape):
        #take the individual img strings and convert them to tensors
        img = img.replace('[','')
        img = img.replace(']','')
        img = img.replace('\n','')
        img = img.split(' ')
        img = [y.replace(' ','') for y in img]
        img = [y for y in img if y!= '']
        img = [float(y) for y in img]
        img = [y/255 for y in img]
        img = np.array(img)
        img = img.reshape(img_shape)
        return tf.cast(img,'float32')
    
    def __get_output(self, label, num_classes):
        #convert the labels (as ints) to one hot encoding
        return tf.one_hot(label,num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        #batches is a dataframe where onely the indexes we want for this batch
        img_batch = batches[self.X_col]
        label_batch = batches[self.Y_col]
        index_batch = batches.index

        X_batch = np.asarray([self.__get_input(x, self.input_size) for x  in img_batch])
        Y_batch = np.asarray([self.__get_output(x, self.num_classes) for x in label_batch])

        return tuple([index_batch,X_batch]), Y_batch
    
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:min((index+1) * self.batch_size,len(self.df.index))]
        X, Y = self.__get_data(batches)   
        return X, Y
    
    def __len__(self):
        #This is the number of batches to use in an epoch
        if self.dataused % self.batch_size == 0:
            l = int(self.dataused//self.batch_size)
        else:
            l = int(self.dataused//self.batch_size) + 1
        print('Number of batches to use: ',l)
        return l

class CustomDBDataGen(tf.keras.utils.Sequence):

    def __init__(self, conn, X_col, Y_col,
                 batch_size,
                 num_classes,
                 input_size=(28, 28, 1),
                 test=False,
                 ):
        #do stuff
        #init db connection and init vars
        self.conn = conn
        self.test = test
        self.num_classes = num_classes


    def on_epoch_end(self):
        a = 0


    def __getitem__(self, index):
        #select only the batch where batch = index
        curr = self.conn.cursor()
        curr.execute('''SELECT id,data, label_num FROM imgs WHERE (test = (?) AND used = 1 AND batch_num = (?))''',(self.test,index))
        ids = []
        imgs = []
        labels = []
        for id,img, label in curr:
            ids.append(id)
            imgs.append(img)
            labels.append(int.from_bytes(label,'little'))

        #convert to tensors
        ids = np.array(ids)
        ids = tf.cast(ids, 'int16')

        imgs = np.array(imgs)
        imgs = tf.cast(imgs,'float32')

        labels = np.array(labels)
        labels = tf.one_hot(labels,self.num_classes)

        return tuple([ids,imgs]), labels





    def __len__(self):
        #calculates the number of batches to use
        #max of the DB.imgs.batch_num where train==True
        a = 1
    
