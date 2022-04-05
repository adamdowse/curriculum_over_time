import tensorflow as tf
import pandas as pd
import numpy as np
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
        self.class_score = [0]*self.num_classes #TODO implement this
        print('original',self.class_used)
        print('original',self.dataused)
    
    def on_epoch_end(self,losses_df):
        #losses_df = (i|label,score,0,1,2,..) where use is from 0 to dataused and nan past dataused
        #df = (i|img,label,score)
        #update the score col in df
        self.df.update(losses_df['score'],overwrite=True)
        #sort the large dataframe by the use index and put nans last so not used
        self.df = self.df.sort_values('score',na_position='last')
        print(self.df)
        self.dataused = len(self.df.index)-self.df.score.isna().sum()
        print(self.dataused)

        #produce some statistics
        #self.class_used = [len(self.df([:self.dataused,'label'==x].index) for x in range(self.num_classes)]
        #self.class_used = [len(self.df[(self.df.label==x) & (self.df.score !=np.nan)].index) for x in range(self.num_classes)]
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
        #print('_get')
        batches = self.df[index * self.batch_size:min((index+1) * self.batch_size,len(self.df.index))]
        X, Y = self.__get_data(batches)   
        #print(X,Y)  
        return X, Y
    
    def __len__(self):
        #TODO this is culpret
        #maybe need to add one but might be problem if exact size
        return int(self.dataused // self.batch_size)

#TODO THIS NEEDS TO BE BATCH SIZE AGNOSTIC SO IT CAN VARY IN SIZE