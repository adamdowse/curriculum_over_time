import tensorflow as tf
import pandas as pd
import numpy as np
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, Y_col,
                 batch_size,
                 input_size=(28, 28, 1),
                 test=False):
        
        self.df = df.copy() #[i|img,label,use] use here is a random index
        self.df = self.df.sample(frac=1)
        self.X_col = X_col #img
        self.Y_col = Y_col #label
        self.batch_size = batch_size
        self.input_size = input_size
        self.test = test
        #add the other key vars here
        #also add stats holding info
        self.dataused = len(self.df.index)
        self.n = len(self.df)
        self.num_classes = df[Y_col].nunique()
        #holds the amount of data used on the next epoch 
        self.class_used = [len(self.df[self.df['label']==x].index) for x in range(self.num_classes)]
        self.class_score = [0]*self.num_classes
    
    def on_epoch_end(self,use):
        #use = [i|use] where use is from 0 to dataused and nan past dataused
        #update the used col in df
        #TODO change / update this
        self.df['use'] = use
        #sort the large dataframe by the use index and put nans last so not used
        self.df = self.df.sort_values('use',na_position='last')
        self.dataused = self.df['use'].max()

        #produce some statistics
        self.class_used = [len(self.df[df.loc[:self.dataused,'label']==x].index) for x in range(self.num_classes)]

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
        batches = self.df[index * self.batch_size:(index+1) * self.batch_size]
        X, Y = self.__get_data(batches)      
        self.b += 1  
        return X, Y
    
    def __len__(self):
        return self.dataused // self.batch_size
