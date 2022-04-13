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
    if not os.path.isdir(info.data_path + info.dataset_name) or bypass==True:
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
        os.makedirs(info.data_path + info.dataset_name)
        df.to_csv(info.data_path + info.dataset_name + '/imagedata.csv')
        df_losses.to_csv(info.data_path  + info.dataset_name + '/lossdata.csv')
        with open(info.data_path+info.dataset_name+'/metadata.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([info.num_classes])
            writer.writerow([info.class_names])
            writer.writerow([info.img_shape])

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
        

    df = df.set_index('i')
    train_df = df[df['test']==False]
    test_df = df[df['test']==True]

    train_df = train_df.drop(columns='test')
    test_df = test_df.drop(columns='test')

    print('INIT: Finished creating train dataset')

    df_train_losses = df_losses[df_losses['test']==False]
    df_train_losses = df_train_losses.drop(columns='test').set_index('i')
    df_train_losses['score'] = np.nan
    print(df_train_losses.head())

    return df_train_losses,train_df,test_df,info

#convert from a vector image to 3d representation again and build dataset form dataframe
def img_dim_shift(x,info):
    #x is a vector of strings
    new_x = []
    for img in x:
        img = str_to_list(img)
        img = [float(y) for y in img]
        img = [y/256 for y in img]
        img = np.array(img)
        img = img.reshape(info.img_shape)
        new_x.append(img)
    return new_x

def str_to_list(img):
    img = img.replace('[','')
    img = img.replace(']','')
    img = img.replace('\n','')
    img = img.split(' ')
    img = [y.replace(' ','') for y in img]
    img = [y for y in img if y!= '']
    return img

def label_oh(x,info):
    return tf.one_hot(x,info.num_classes)


def calc_grad(n,data):
    grads = []
    for i,row in data.iterrows():
        x = np.array(range(n)).reshape((-1,1))
        y = row[-n:].to_numpy()
        model = linear_model.LinearRegression().fit(x,y)
        grads.append(model.coef_[0])
    return pd.Series(data=grads)

def fill_func(n,row,info):

    if info.fill_func == 'ffill':
        data.iloc[-2:] = data.iloc[:,-2:].fillna(method='ffill',axis=1)

    elif info.fill_func == 'reg_fill':
        #take the losses and fill the nan values in the last row with a regression output
        #input is a df (i|score,0,1,2,...,j) if j is np.nan predict it)
        if pd.isnull(data.iloc[-1]):
            if info.current_epoch == 0:
                print('error must use all data for current version of reg_fill on epoch 0')
            elif info.current_epoch == 1:
                #use the last epochs loss
                data.iloc[-1] = data.iloc[-2]
            elif info.current_epoch < n:
                #use a smaller lookback for regression
                n = info.current_epoch
                x = np.array(range(n)).reshape((-1,1))
                y = data[-(n+1):-1].to_numpy() 
                model = linear_model.LinearRegression().fit(x,y)
                data.iloc[-1] = model.predict(np.array([n]).reshape((-1,1)))[0]
            else:
                #use the specified lookback and predict next
                x = np.array(range(n)).reshape((-1,1))
                y = data[-(n+1):-1].to_numpy() 
                model = linear_model.LinearRegression().fit(x,y)
                data.iloc[-1] = model.predict(np.array([n]).reshape((-1,1)))[0]

    elif info.fill_func == 'reg_fill_grav':
        #take the losses and fill the nan values in the last row with a regression output
        #input is a row (i|score,0,1,2,...,j) if j is np.nan predict it
        if pd.isnull(data.iloc[-1]):
            if info.current_epoch == 0:
                print('error must use all data for current version of reg_fill on epoch 0')
            elif info.current_epoch == 1:
                #use the last epochs loss
                data.iloc[-1] = data.iloc[-2] - info.score_grav
            elif info.current_epoch < n:
                #use a smaller lookback for regression
                n = info.current_epoch
                x = np.array(range(n)).reshape((-1,1))
                y = data[-(n+1):-1].to_numpy() 
                model = linear_model.LinearRegression().fit(x,y)
                data.iloc[-1] = model.predict(np.array([n]).reshape((-1,1)))[0] - info.score_grav
            else:
                #use the specified lookback and predict next
                x = np.array(range(n)).reshape((-1,1))
                y = data[-(n+1):-1].to_numpy() 
                model = linear_model.LinearRegression().fit(x,y)
                data.iloc[-1] = model.predict(np.array([n]).reshape((-1,1)))[0] - info.score_grav
        
    return data

def scoring_func(df,info):
    #df is train_data_losses = (i|score,0,1,2,3,4...)
    #take the training dataframe and apply the scoring functions to it
    #the output is a dataframe with the score of each index

    #TODO Clean this all up 
    #reset the scores
    df['score'] = np.nan

    if info.scoring_function == 'normal':
        #use the normal reshuffling technique
        i = random.sample([x for x in range(len(df.index))],len(df.index))
        df['score'] = i
        return df

    elif info.scoring_function == 'grads':
        #calc gradients over last n losses
        #fill the missing losses with the predicted value from the regression

        n = info.score_lookback #lookback for gradients
        if info.current_epoch == 0:
            print('used 0 epoch')
            #use the first epochs loss info
            df['score'] = df['0']
        elif info.current_epoch == 1:
            #fill nas
            df = df.apply(lambda row : fill_func(n,row,info),axis=1)
            #df = reg_fill(n,df,info)
            df['score'] = df['1']
        elif info.current_epoch < n:
            print('use reduced grads')
            #fill the missing info with regression
            df = df.apply(lambda row : fill_func(n,row,info), axis=1)
            if np.nan in df.iloc[:,-1]:
                print('NAN found after fill proceduce')
            #calc grad over as many as possible
            df['score'] = calc_grad(info.current_epoch,df.iloc[:,-info.current_epoch:])

        elif info.current_epoch >= n:
            print('use full grads')
            #fill the missing info
            df = df.apply(lambda row : fill_func(n,row,info), axis=1)
            if np.nan in df.iloc[:,-1]:
                print('NAN found after fill proceduce')
            
            df['score'] = calc_grad(n,df.iloc[:,-n:])

    elif info.scoring_function == 'class_corr':

    else:
        print('COLLECT TRAIN DATA: ERROR no valid scoring function')
    
    print(df)
    return df

def pacing_func(df,info):
    '''
    functions:
        none
        naive_linear
        naive_grad
        stat_grad (not done)
    '''
    #df=(i|socore, 0,1,2,3...)


    def get_grad(n,row):
        #n is the lookback
        #data is a row
        x = np.array(range(n)).reshape((-1,1))
        y = row[-n:].to_numpy()
        model = linear_model.LinearRegression().fit(x,y)
        return model.coef_[0]


    if info.pacing_function == 'none':
        #shuffle all data
        df['score'] = random.sample([x for x in range(len(df.index))],len(df.index))
        return df

    elif info.pacing_function == 'naive_linear':
        n = len(df.index)
        df = df.sort_values('score',ascending=info.lam_high_first)
        #calc amount of data to use
        d = info.lam_zero + ((1-info.lam_zero)/(info.max_epochs * info.lam_max))*info.current_epoch #ref from cl survey
        d = min([1,d]) #percentage
        d = int(d*n)
        print('pacing dataused ',d)
        i = [x for x in range(d)] #[0 to dataused]
        if n-d != 0:
            i = i + [np.nan]*(n-d)
        df['score'] = i #the final rank
        return df
    
    elif info.pacing_function == 'naive_grad':
        #use the loss tracts as a guage of of well the model is doing
        #start with fixed small amount of data (COULD DO MORE WITH THIS, WHAT IS THE BEST STARTING SET?)
        #if average train loss:
            #<< add data
            #>> remove data
            #range around 0 keep fixed

        #current epoch is doing these calc for next epoch (epoch 0 has loss info for epoch 0 and rank will be used in epoch 1)
        total_data = len(df.index)
        if info.current_epoch == 0:
            info.lam_data = total_data * info.lam_zero
        else:
            #calc average grad
            if info.current_epoch < info.lam_lookback:
                n = info.current_epoch +1
            else:
                n = info.lam_lookback
            avg_grad = get_grad(n,df.iloc[:,-n:].mean())

            #modify amount of data used
            #TODO add a learning rate system maybe?
            if info.lam_low_first == True:
                if avg_grad < info.lam_lower_bound:
                    info.lam_data = info.lam_data + (avg_grad * info.lam_data_multiplier)
                elif avg_grad > info.lam_upper_bound:
                    info.lam_data = info.lam_data - (avg_grad * info.lam_data_multiplier)
        
        #clip to min and max data
        info.lam_data = min([total_data,info.lam_data])
        info.lam_data = max([info.lam_zero,info.lam_data])

        df = df.sort_values('score',ascending=info.lam_high_first) #can be high or low first
        df['score'] = range(info.lam_data) +[np.nan]*(total_data-info.lam_data)
    
    elif info.pacing_function == 'stat_grad':
        #use a statistical process to find the best addition reduction of data
        print('This has not been developed yet... ERROR')

def update_col(batch_loss, col, indexes,info):
    #take the batch_losses and update column 
    for i in range(len(batch_loss)):
        #add the index and loss to the column
        new_row = pd.DataFrame(data={'i':indexes[i],str(info.current_epoch):batch_loss[i].numpy()},index=[0])
        col = pd.concat([col,new_row],axis=0)
    return col

