import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn import metrics
from sklearn import linear_model
from sklearn import cluster
from sklearn import decomposition
import os
import csv
import random
import glob
import sqlite3
from sqlite3 import Error

def DB_create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def DB_add_img(conn, img):
    """
    Create a new img into the img table
    :param conn:
    :param img: (label_name,label_num,data,batch_num)
    """
    if conn is not None:
        sql = ''' INSERT INTO imgs(label_name,label_num,data,batch_num,test)
                VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, img)
        conn.commit()
    else:
        print('ERROR connecting to db')


def create_task(conn, task):
    """
    Create a new task
    :param conn:
    :param task:
    :return:
    """

    sql = ''' INSERT INTO tasks(name,priority,status_id,project_id,begin_date,end_date)
              VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()
    return cur.lastrowid

def DB_import_dataset(conn,info):
    '''
    Take the desired dataset and download it form tf_datasets and put the images into the db

    '''
    #downlaod the tfdataset
    print('INIT: Using ',info.dataset_name, ' data, downloading now...')
    #take the tfds dataset and produce a dataset and dataframe
    ds, ds_info = tfds.load(info.dataset_name,with_info=True,shuffle_files=False,as_supervised=True,split='all')
    
    #record ds metadata
    info.num_classes = ds_info.features['label'].num_classes
    info.class_names = ds_info.features['label'].names

    #Take the dataset and add info to db
    i = 0
    for image,label in ds:
        if i == 0:
            info.img_shape = image.shape
        if random.random() > 0.8: test = True 
        else: test = False
        #TODO DO ADDITION OF DATA HERE START HEREEEEEEEEEEEEEEEEEEEEEEE_________________________EEEE
        #are we adding test /train ids to db? YES
        i += 1


def DB_create(conn):
    '''
    Create the database if it does not exist for the current data
    param database: file loc of database to create
    '''

    sql_create_img_table = """ CREATE TABLE IF NOT EXISTS imgs (
                                        id integer PRIMARY KEY,
                                        label_name text NOT NULL,
                                        label_num integer,
                                        data text,
                                        score float,
                                        batch_num integer
                                        test bool
                                    ); """

    sql_create_loss_table = """CREATE TABLE IF NOT EXISTS losses (
                                    step integer PRIMARY KEY,
                                    img_id integer NOT NULL,
                                    loss float,
                                    FOREIGN KEY (img_id) REFERENCES imgs (id)
                                );"""
    
    #TODO ADD THE PREDS SECTION


    # create tables
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_img_table)

        # create tasks table
        create_table(conn, sql_create_loss_table)
    else:
        print("Error! cannot create the database connection.")

    







def split_save_df(df,fpath):
    #save the df as multiple smaller files
    num_rows = 10000
    i = 0
    while len(df.index) > num_rows:
        print('Saving chunk ',i)
        df.iloc[:num_rows,:].to_csv(fpath+str(i)+'.csv')
        df = df.drop(df.iloc[:num_rows,:].index)
        i += 1
    print('Saved DF in '+str(i+1)+' chunks')

def split_load_df(fpath):
    fns = glob.glob(fpath+'*.csv')
    #find first file
    fn = fns[0]
    df = pd.read_csv(fn,index_col='Unnamed: 0')
    for fn in fns[1:]:
        temp_df  =  pd.read_csv(fn,index_col='Unnamed: 0')
        df = pd.concat([df,temp_df])
    print('DF loaded ',len(df.index),'rows with ',len(fns),' parts')
    return df


def init_data(info,bypass=False):
    '''
    Take a named dataset and if it already exists use the pre svaed data otherwise download it.
    '''
    if not os.path.isdir(info.data_path + info.dataset_name) or bypass==True:
        print('INIT: Cannot find ',info.dataset_name, ' data, downloading now...')
        #take the tfds dataset and produce a dataset and dataframe
        ds, ds_info = tfds.load(info.dataset_name,with_info=True,shuffle_files=True,as_supervised=True,split='all')
        df = pd.DataFrame(columns=['img','label','i','test'])

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
            i += 1
        
        #make the required directory and save the data
        os.makedirs(info.data_path + info.dataset_name)
        split_save_df(df,info.data_path + info.dataset_name + '/imagedata')

        with open(info.data_path+info.dataset_name+'/metadata.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([info.num_classes])
            writer.writerow([info.class_names])
            writer.writerow([info.img_shape])

    print('INIT: Using found',info.dataset_name, 'data')

    df = split_load_df(info.data_path+info.dataset_name+'/imagedata')

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

    #reduce the dataset
    if info.dataset_size < 1 and info.dataset_size > 0:
        if info.dataset_similarity == 'True':
            d = int(info.dataset_size * len(train_df))
            print('amount of data used = ',d)
            train_df = train_df.iloc[:d,:]
            #df_train_losses = df_train_losses.iloc[:d,:]
        else:
            print('frac used = ',info.dataset_size)
            train_df = train_df.sample(frac=info.dataset_size,axis=0)
            #df_train_losses = df_train_losses.loc[train_df.index,:]

        print(train_df.label.value_counts())
    
    #create the loss info dfs
    df_train_losses = train_df.copy()
    df_train_losses = df_train_losses.drop('img',axis=1)
    df_train_losses['score'] = np.nan
    df_test_losses = test_df.copy()
    df_test_losses = df_test_losses.drop('img',axis=1)
    df_test_losses['loss'] = np.nan
    df_test_losses['pred'] = np.nan

    for c in range(info.num_classes):
        df_test_losses[str(c)] = np.nan
    
    print('INIT: Finished creating train dataset')

    return df_train_losses,df_test_losses,train_df,test_df,info

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

def fill_func(data,info):
    #data is a row
    if info.current_epoch == 0:
        #use the stat things here for initial preds
        a = 0
    else:
        if info.fill_function == 'ffill':
            data.iloc[-2:] = data.iloc[-2:].fillna(method='ffill')

        elif info.fill_function == 'reg_fill':
            #take the losses and fill the nan values in the last row with a regression output
            #input is a df (i|score,0,1,2,...,j) if j is np.nan predict it)
            if pd.isnull(data.iloc[-1]):
                n = info.score_lookback
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

        elif info.fill_function == 'reg_fill_grav':
            #take the losses and fill the nan values in the last row with a regression output
            #input is a row (i|score,0,1,2,...,j) if j is np.nan predict it
            if pd.isnull(data.iloc[-1]):
                n = info.score_lookback
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

    #reset the scores
    df['score'] = np.nan

    #fill the latest loss values so no np.nans
    df = df.apply(lambda row : fill_func(row,info),axis=1)

    #check if nans are in last column
    if np.nan in df.iloc[:,-1]:
        print('NAN found after fill proceduce')

    if info.scoring_function == 'normal':
        #use the normal reshuffling technique
        i = random.sample([x for x in range(len(df.index))],len(df.index))
        df['score'] = i
        return df

    elif info.scoring_function == 'last_loss':
        df['score'] = df.iloc[:,-1]
        return df

    elif info.scoring_function == 'grads':
        #calc gradients over last n losses
        #fill the missing losses with the predicted value from the regression

        n = info.score_lookback #lookback for gradients
        if info.current_epoch <= 1:
            #use the first or second epochs loss info
            df['score'] = df[str(info.current_epoch)]
        elif info.current_epoch < n:
            #calc grad over as many as possible
            df['score'] = calc_grad(info.current_epoch,df.iloc[:,-info.current_epoch:])
        elif info.current_epoch >= n:
            #fill the missing info
            df['score'] = calc_grad(n,df.iloc[:,-n:])

    elif info.scoring_function == 'class_corr':
        print('NOT DEVELOPED')


    elif info.scoring_function == 'loss_clusters':
        #convert to the array
        data = np.array([x for x in df.loc[:,str(info.current_epoch)].to_numpy()])
        print(data)

        #km cluster
        km = cluster.MiniBatchKMeans(n_clusters=info.batch_size)
        km = km.fit(data.reshape((-1,1)))
        cluster_data = km.predict(data.reshape((-1,1)))

        df['score'] = cluster_data
        df = df.sort_values('score',ascending=True)
        cluster_data = df['score'].to_numpy()

        #convert clusters to indexes
        output = []
        current = 0
        count = 0
        for i in range(len(df.index)):
            if cluster_data[i] != current:
                current = cluster_data[i]
                count = 0

            output.append(cluster_data[i] + (info.batch_size*count))
            count += 1

        df['score'] = output

    elif info.scoring_function == 'pred_clusters':
        #use the prediction landscape to cluster into batch size number of groups then randomly sample from each
        
        #convert to np array
        data = np.array([x for x in df.loc[:,str(info.current_epoch)].to_numpy()])

        #dim reduction if needed here

        #calc kmeans
        km = cluster.MiniBatchKMeans(n_clusters=info.batch_size)
        km = km.fit(data)
        cluster_data = km.predict(data)

        #visulise
        if False:
            #compress to 2 dims
            #pca = decomposition.PCA(n_components=3)
            #pca.fit(data)
            #comp_data = pca.transform(data)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            print(km.predict(data))
            print(data[0])
            ax.scatter(data[:,0],data[:,1],data[:,2],c=cluster_data)
            ax.set_xlabel('0')
            ax.set_ylabel('1')
            ax.set_zlabel('2')
            fig.savefig('pred_cluster_loss')

        #order by cluster
        df['score'] = cluster_data
        df = df.sort_values('score',ascending=True)
        cluster_data = df['score'].to_numpy()

        #convert clusters to indexes
        output = []
        current = 0
        count = 0
        for i in range(len(df.index)):
            if cluster_data[i] != current:
                current = cluster_data[i]
                count = 0

            output.append(cluster_data[i] + (info.batch_size*count))
            count += 1

        df['score'] = output

    elif info.scoring_function == 'pred_biggest_move':
        #euclidian distance
        #order by the biggest move towards 0 
        if info.current_epoch == 0:
            #randomly assign
            i = random.sample([x for x in range(len(df.index))],len(df.index))
            df['score'] = i
        else:
            #find the hyp distance of 1st points
            def euq_dis (x):
                #calculate the difference between equlidian distances of 2 points 
                a = [i**2 for i in x.iloc[0]]
                a = np.power(np.sum(a),(1/10))

                b = [i**2 for i in x.iloc[1]]
                b = np.power(np.sum(b),(1/10))

                #compare distances
                dist = a-b
                return dist

            n = info.current_epoch + 3
            df['score'] = df.iloc[:,n-2:n].apply(euq_dis,axis=1)


    elif info.scoring_function == 'pred_best_angle':
        #TODO needs checking
        #cosign distance from ~45degs
        def cos_dist(x):
            sim = metrics.pairwise.cosine_similarity([x,[1]*10])
            return sim[0,1]
        
        n = info.current_epoch + 2
        df['score'] = df.iloc[:,n].apply(cos_dist)

    elif info.scoring_function == 'pred_grad_cluster':
        prnt()
    else:
        print('SCORING FUNCTION: ERROR no valid scoring function')
    
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


    if info.pacing_function == 'shuffle':
        #shuffle all data
        df['score'] = random.sample([x for x in range(len(df.index))],len(df.index))
        return df
    
    elif info.pacing_function == 'ordered':
        #order the data by scoring func but dont reduce data
        df = df.sort_values('score',ascending=info.lam_low_first)
        df['score'] = [x for x in range(len(df.index))]
        return df
    
    elif info.pacing_function == 'mixed':
        df = df.sort_values('score',ascending=info.lam_low_first)
        a = [x for x in range(len(df.index))]
        #order from highest lowest second hgihest, second lowest ect
        i = []
        for x in range(len(df.index)):
            if x % 2 == 0 :
                i.append(a.pop(0))
            else:
                i.append(a.pop(-1))

        df['score'] = i

    elif info.pacing_function == 'naive_linear':
        n = len(df.index)
        df = df.sort_values('score',ascending=info.lam_low_first)
        #calc amount of data to use
        d = info.lam_zero + ((1-info.lam_zero)/(info.max_epochs * info.lam_max))*info.current_epoch #ref from cl survey
        d = min([1,d]) #dec percentage
        d = int(d*n)
        print('pacing dataused ',d)
        i = [x for x in range(d)] #[0 to dataused]
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
            info.lam_data = int(total_data * info.lam_zero)
        else:
            #calc average grad
            if info.current_epoch < info.lam_lookback-1:
                n = info.current_epoch
            else:
                n = info.lam_lookback

            avg_grad = get_grad(n,df.iloc[:,-n:].mean())

            #modify amount of data used
            #TODO add a learning rate system maybe?
            if avg_grad < info.lam_lower_bound:
                info.lam_data = int(info.lam_data + (avg_grad * info.lam_data_multiplier))
            elif avg_grad > info.lam_upper_bound:
                info.lam_data = int(info.lam_data - (avg_grad * info.lam_data_multiplier))

        #clip to min and max data
        info.lam_data = min([total_data,info.lam_data])
        info.lam_data = max([int(info.lam_zero*total_data),info.lam_data])
        print(info.lam_data)
        df = df.sort_values('score',ascending=info.lam_low_first) #can be high or low first
        df['score'] = [x for x in range(info.lam_data)] + [np.nan]*(total_data-info.lam_data)
    
    elif info.pacing_function == 'stat_grad':

        #use a statistical process to find the best addition reduction of data
        print('This has not been developed yet... ERROR')


    else:
        print('not correct pacing function')
    
    return df

def update_col(vals,labels, col, indexes,info):
    #This takes the list of losses from the batch, the labels

    if info.record_loss == 'sum':
        #take the batch_losses and update column 
        for i in range(len(vals)):
            #add the index and loss to the column
            new_row = pd.DataFrame(data={'i':indexes[i],str(info.current_epoch):vals[i].numpy()},index=[0])
            col = pd.concat([col,new_row],axis=0)
        return col
    else:
        #compress the predictions USE AVERAGE ERROR
        for i in range(len(vals)):
            #find the new loss vals
            label = labels[i]
            loss_vals = vals[i].numpy()
            loss_vals = [abs(label[j]-loss_vals[j]) for j in range(info.num_classes)]
            
            new_row = pd.DataFrame(data={'i':indexes[i],str(info.current_epoch):[loss_vals]},index=[0])

            col = pd.concat([col,new_row],axis=0)
        return col

def update_test_df(df,batch_loss,preds,indexes):
    #takes the df and 2 arrays of indexes and losses to update the df
    cols = [str(x) for x in range(10)]
    for i in range(len(batch_loss)):
        df.loc[indexes.values[i],'loss'] = batch_loss.numpy()[i]
        df.loc[indexes.values[i],cols] = preds.numpy()[i]
        
        #calc pred class
        df.loc[indexes.values[i],'pred'] = np.argmax(preds.numpy()[i])
    return df

def f1_score(df,target):
    #tp/tp+0.5(fp+fn)

    #count 
    tp = len(df[(df['label'] == target) & (df['label'] == df['pred'])].index)
    fp = len(df[(df['label'] != target) & (df['label'] == df['pred'])].index)
    tn = len(df[(df['label'] == target) & (df['label'] != df['pred'])].index)
    fn = len(df[(df['label'] != target) & (df['label'] != df['pred'])].index)

    return tp/(tp + 0.5*(fp+fn))
    


