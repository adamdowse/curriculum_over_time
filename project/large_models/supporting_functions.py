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
import io
import wandb
from dppy.finite_dpps import FiniteDPP

def DB_create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file,detect_types=sqlite3.PARSE_DECLTYPES)
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

def array_to_bin(arr):
    #converts an arry into binary representation
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def bin_to_array(bin):
    #converts bin to np array
    out = io.BytesIO(bin)
    out.seek(0)
    return np.load(out)

def DB_add_img(conn, img):
    """
    Create a new img into the img table
    :param conn:
    :param img: (label_name,label_num,data,batch_num)
    """
    if conn is not None:
        sql = ''' INSERT INTO imgs(label_name,label_num,data,score,rank,test,used)
                VALUES(?,?,?,?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, img)
    else:
        print('ERROR connecting to db')

def DB_import_dataset(conn,config,info):
    '''
    Take the desired dataset and download it form tf_datasets and put the images into the db

    '''
    #downlaod the tfdataset
    print('INIT: Using ',config['dataset_name'], ' data, downloading now...')
    #take the tfds dataset and produce a dataset and dataframe
    ds, ds_info = tfds.load(config['dataset_name'],with_info=True,shuffle_files=False,as_supervised=True,split='all')
    
    #record ds metadata
    info.num_classes = ds_info.features['label'].num_classes
    info.class_names = ds_info.features['label'].names

    #Take the dataset and add info to db
    i = 0

    for image,label in ds:
        if i % 5000 == 0: print('Images Complete = ',i)
        if i == 0:
            info.img_shape = image.shape
        if random.random() > 0.8: 
            test = 1 
        else: 
            test = 0
        #label_name, label_num, data,score,rank,test,used
        data_to_add = (str(label.numpy()),int(label.numpy()),image.numpy(),0,0,int(test),0,)
        DB_add_img(conn, data_to_add)
        i += 1
    conn.commit()
    return info

def DB_create(conn):
    '''
    Create the database if it does not exist for the current data
    param database: file loc of database to create
    '''

    sql_create_img_table = """ CREATE TABLE IF NOT EXISTS imgs (
                                        id INTEGER PRIMARY KEY,
                                        label_name text NOT NULL,
                                        label_num INTEGER,
                                        data array,
                                        score REAL,
                                        rank INTEGER,
                                        batch_num INTEGER,
                                        test INTEGER,
                                        used INTEGER
                                    ); """

    sql_create_loss_table = """CREATE TABLE IF NOT EXISTS losses (
                                    result_id INTEGER PRIMARY KEY,
                                    step INTEGER,
                                    img_id INTEGER NOT NULL,
                                    loss REAL,
                                    output array,
                                    batch_num INTEGER,
                                    FOREIGN KEY (img_id) REFERENCES imgs (id)
                                );"""
    


    # create tables
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_img_table)

        # create tasks table
        create_table(conn, sql_create_loss_table)

    else:
        print("Error! cannot create the database connection.")


def rand(a):
    return random.random()

def DB_set_used(conn,test_n,train_n):
    #randomly set the amount of data to use
    #test is the bool True or False
    #n is the number of samples
    #TODO convert to be able to set a seed in random (probably a for loop)

    cur = conn.cursor()
    cur.execute('''UPDATE imgs SET used = 0 WHERE test = 1''')
    cur.execute('''UPDATE imgs SET used = 1 WHERE test = 1 ORDER BY RANDOM() LIMIT (?) ''',(test_n,))
    cur.execute('''UPDATE imgs SET used = 0 WHERE test = 0''')
    cur.execute('''UPDATE imgs SET used = 1 WHERE test = 0 ORDER BY RANDOM() LIMIT (?) ''',(train_n,))
    conn.commit()


def DB_random_batches(conn,test,img_num,batch_size):
    if img_num / batch_size == 0:
        num_batches = int(img_num/batch_size)
    else:
        num_batches = 1 + int(img_num/batch_size)

    print('There are ',num_batches,' batches')
    arr = []
    for n in range(num_batches):
        to_add = [n]*batch_size
        arr = arr + to_add
    
    #limit
    arr = arr[:img_num]
    #shuffle
    random.shuffle(arr)

    #add to db
    curr = conn.cursor()
    #curr.execute(''' UPDATE imgs SET batch_num = (?) WHERE( id = (?) AND test = (?) AND used = 1)''',((arr[i],str(test))) 
    #get all the ids that are going to be updated
    curr.execute(''' SELECT id FROM imgs WHERE( test = (?) AND used = 1.0)''',(int(test),)) 
    ids = []
    for id in curr:
        ids.append(id[0])

    incurr = conn.cursor()
    for i,id in enumerate(ids):
        incurr.execute(''' UPDATE imgs SET batch_num = (?) WHERE id = (?)''',(int(arr[i]),int(id),))
    conn.commit()


def DB_init(conn):
    #set the losses and output tables to initial state and prune off columns that are not needed
    #imgs init
    curr = conn.cursor()
    curr.execute(''' UPDATE imgs SET score = (?) ''',(random.random(),))
    curr.execute(''' UPDATE imgs SET rank = (?) ''',('nan',))
    curr.execute(''' UPDATE imgs SET batch_num = (?) ''',('nan',))
    curr.execute(''' UPDATE imgs SET test = 1''')
    curr.execute(''' SELECT COUNT(DISTINCT id) FROM imgs ''')
    count = int(curr.fetchone()[0] * 0.8)
    curr.execute(''' UPDATE imgs SET test = 0 ORDER BY RANDOM() LIMIT (?)''',(count,)) #TODO this needs a seed
    curr.execute(''' UPDATE imgs SET used = (?) ''',(0,))

    #losses init
    curr.execute(''' DELETE FROM losses ''')

    conn.commit()
    

def DB_update(conn,info,step,X,Y,batch_loss,preds):
    #update the db from batch infomation
    curr = conn.cursor()
    for i, x in enumerate(X[0].numpy()): 
        #print(float(batch_loss.numpy()[i]),int(step),int(x),int(info.batch_num)) 
        curr.execute('''INSERT INTO losses(loss,step,output,img_id,batch_num) VALUES(?,?,?,?,?)''',
            (float(batch_loss[i]),
            int(step),
            np.array(preds[i]),
            int(x),
            int(info.batch_num),))

    conn.commit()


def log(conn,output_name,table,test,step,name,mean=False):
    #log an array with wandb that act as a point in the histogram over time
    curr = conn.cursor()
    #sql = ''' SELECT loss FROM losses WHERE img_id = SELECT id FROM imgs WHERE used = 1 AND test = (?)''' #step = (?) AND #(
    sql = '''   SELECT l.loss 
                FROM losses l
                inner join imgs i on l.img_id=i.id
                WHERE l.step=(?) AND i.used = 1 AND i.test = (?)''' 
    curr.execute(sql,(int(step),int(test),))

    results = curr.fetchall()
    results = np.array(results)
    results = np.squeeze(results)
    if mean:
        results = np.mean(results)
    wandb.log({name:results},step=step)



def log_acc(conn,test,step,name):
    #log an array with wandb that act as a point in the histogram over time
    curr = conn.cursor()
    sql = '''   SELECT l.output, i.label_num 
                FROM losses l
                inner join imgs i on l.img_id=i.id
                WHERE l.step=(?) AND i.used = 1 AND i.test = (?)'''
    curr.execute(sql,(step,test,))

    results = curr.fetchall()
    #print('result len ',len(results))
    cm = np.zeros((10,10)) #TODO change to args
    for output,label in results:
        output = np.argmax(output)
        cm[label,output] += 1
    
    #total accuracy
    acc = [cm[i,i] for i in range(10)]
    #can add class specific stuff here
    if np.sum(np.sum(cm)) == 0:
        print('zero acc..')
        acc = 0
    else:
        acc = np.sum(acc)/np.sum(np.sum(cm))
    wandb.log({name+'_acc':acc},step=step)

    #class accuracy
    keys = [x for x in range(10)] #TODO change to args
    class_accs = {}
    c_accs = [cm[i,i]/np.sum(cm[:,i]) if np.sum(cm[:,i]) != 0 else 0 for i in range(10)] #TODO change to args
    for i in range(len(keys)):
        class_accs[name+'_acc_'+str(keys[i])] = c_accs[i]
    wandb.log(class_accs,step=step)
    
    #TP,FN,FP,TNs
    TPs = [cm[i,i] for i in range(10)] #TODO args
    FNs = [np.sum(cm[i,:]) - cm[i,i] for i in range(10)] #TODO args
    FPs = [np.sum(cm[:,i]) - cm[i,i] for i in range(10)] #TODO args
    TNs = [np.sum(cm) - cm[:,i] - cm[i,:] + cm[i,i] for i in range(10)] #TODO args

    #class F1 score
    class_f1_scores = {}
    c_f1_scores = [TPs[i]/(TPs[i]+0.5*(FPs[i]+FNs[i])) if TPs[i]+0.5*(FPs[i]+FNs[i]) != 0 else 0 for i in range(10)] #TODO args
    for i in range(len(keys)):
        class_f1_scores[name+'_f1_'+str(keys[i])] = c_f1_scores[i]
    wandb.log(class_f1_scores,step=step)

    #total f1 score
    total_f1_score = [c_f1_scores[i]*np.sum(cm[i,:]) for i in range(10)] #TODO args
    total_f1_score = np.sum(total_f1_score)/np.sum([np.sum(cm[i,:]) for i in range(10)]) #TODO args
    wandb.log({name+'_weighted_f1':total_f1_score},step=step)





def setup_sql_funcs(conn):
    def scoring_function_random(max):
        return random.random()


    conn.create_function("scoring_function_random", 1, scoring_function_random)


def scoring_functions(conn,config,info):
    #convert the given varable into the scores
    #random - randomly assign scores

    curr = conn.cursor()

    if config['scoring_function'] == 'random':
        #randomly assign scores
        curr.execute(''' UPDATE imgs SET score = scoring_function_random(?) WHERE used =1''',(1,))

    
    if config['scoring_function'] == 'last_loss':
        #score is set as the last loss recored for each img
        #TODO this can be one sql call using join i believe
        curr.execute('''SELECT MAX(step) FROM losses''')
        step = curr.fetchall()
        print('Step is'+step)
        ''''''
        curr.execute('''SELECT img_id, loss FROM losses WHERE ( img_id = (SELECT id FROM imgs WHERE (used = 1 AND test = 0) AND step = (?))''',(int(step),))
        results = np.array(curr.fetchall())
        for i,loss in zip(results[:,0],results[:,1]):
            curr.execute('''UPDATE imgs SET score = (?) WHERE id = (?)''',(float(loss),int(i),))

    if config['scoring_function'] == 'loss_cluster':
        #cluster based on the loss 
        curr.execute('''SELECT MAX(step) FROM losses''')
        step = curr.fetchall()
        curr.execute('''SELECT img_id, loss FROM losses WHERE ( img_id = (SELECT id FROM imgs WHERE (used = 1 AND test = 0) AND step = (?))''',(int(step),))
        results = np.array(curr.fetchall())

        #km cluster
        km = cluster.MiniBatchKMeans(n_clusters=config['batch_size'])
        km = km.fit(results[1].reshape((-1,1)))
        cluster_data = km.predict(results[1].reshape((-1,1)))

        results = np.concatenate((results, cluster_data), axis=0) # [id,loss,(0,31 cluster)]
        output = results[0,2] #[id,(0,31 cluster)]

        #pick to create batches
        #create the sub arrays
        for i in range(max(output[1]+1)): #0,1,2,...31
            ind = np.argwhere(output[1]==i)
            if i == 0:
                cluster_split = np.expand_dims(np.squeeze(output[0,ind]), axis=0)
            else:
                cluster_split = np.append( cluster_split, np.expand_dims(np.squeeze(output[0,ind]), axis=0),axis=0)

        #cluster_split = [[ids where c = 0],[ids where c = 1]... ,[c=31]]
        
        i = 0
        mask = np.ones_like(cluster_split)
        out_ids = np.array([])
        while np.sum(np.sum(mask)) != 0:
            p = [x/sum(mask[i]) if x != 0 else 0 for x in mask[i] ]
            ch = np.random.choice(len(cluster_split[i]), 1, p = p)[0]
            out_ids = np.append(out_ids,cluster_split[i,ch])
            mask[i,ch] = 0
            if i == cluster_split.shape[0]-1:
                i = 0
            else:
                i += 1

        #out_ids = [img_id,img_id,...]

        for i in range(out_ids):
            curr.execute('''UPDATE imgs SET score = (?) WHERE id = (?)''',(float(i),int(out_ids[i]),))

    if config['scoring_function'] == 'loss_cluster_batches':
        #cluster so the batches used are clusters 
        #cluster based on the loss 
        curr.execute('''SELECT MAX(step) FROM losses''')
        step = curr.fetchall()
        curr.execute('''SELECT img_id, loss FROM losses WHERE ( img_id = (SELECT id FROM imgs WHERE (used = 1 AND test = 0) AND step = (?))''',(int(step),))
        results = np.array(curr.fetchall())

        #cluster number
        curr.execute('''SELECT COUNT(DISTINCT id) FROM imgs WHERE test = 0 AND used = 1''')
        data_amount = curr.fetchone()[0]
        if data_amount / config['batch_size'] == 0:
            num_batches = int(data_amount/config['batch_size'])
        else:
            num_batches = 1 + int(data_amount/config['batch_size'])

        #km cluster
        km = cluster.MiniBatchKMeans(n_clusters=num_batches)
        km = km.fit(results[1].reshape((-1,1)))
        cluster_data = km.predict(results[1].reshape((-1,1)))

        results = np.concatenate((results, cluster_data), axis=0) # [id,loss,(0,...,num batches)]
        output = results[0,2] #[id,(0,...,num batches)]

        out_ids = output [ :, output[1].argsort()] #ids sorted based on batch_num

        for i in range(out_ids[0]):
            curr.execute('''UPDATE imgs SET score = (?) WHERE id = (?)''',(float(i),int(out_ids[i]),))
    

    if config['scoring_function'] == 'pred_cluster':
        #cluster so the batches used are clusters 
        #cluster based on the softmax outputs
        curr.execute('''SELECT MAX(step) FROM preds''')
        step = curr.fetchall()
        sql = '''SELECT l.img_id, l.output, i.label_num
                FROM losses AS l
                INNER JOIN imgs AS i ON l.img_id=i.id
                WHERE l.step=(?) AND i.used = 1 AND i.test = 0'''
        curr.execute(sql,(int(step),))
        results = np.array(curr.fetchall())

        #seperate the output and ids 
        outputs = results[:,1]
        ids = results[:,0]
        labels = results[:,2]

        #turn the softmax output into softmax error
        for i,output in enumerate(outputs):
            output[labels[i]] = 1 - output[labels[i]]
            outputs[i] = output

        #km cluster
        km = cluster.MiniBatchKMeans(n_clusters=config['batch_size'])
        km = km.fit(outputs)
        cluster_data = km.predict(outputs)

        output = np.concatenate([ids,cluster_data],axis = 0) #reuslts = [[ids],[cluster num]]

        #pick to create batches
        #create the sub arrays
        for i in range(max(output[1]+1)): #0,1,2,...31
            ind = np.argwhere(output[1]==i)
            if i == 0:
                cluster_split = np.expand_dims(np.squeeze(output[0,ind]), axis=0)
            else:
                cluster_split = np.append( cluster_split, np.expand_dims(np.squeeze(output[0,ind]), axis=0),axis=0)

        #cluster_split = [[ids where c = 0],[ids where c = 1]... ,[c=31]]
        
        #randomly pick from each cluster
        i = 0
        mask = np.ones_like(cluster_split)
        out_ids = np.array([])
        while np.sum(np.sum(mask)) != 0:
            p = [x/sum(mask[i]) if x != 0 else 0 for x in mask[i] ]
            ch = np.random.choice(len(cluster_split[i]), 1, p = p)[0]
            out_ids = np.append(out_ids,cluster_split[i,ch])
            mask[i,ch] = 0
            if i == cluster_split.shape[0]-1:
                i = 0
            else:
                i += 1

        #out_ids = [img_id,img_id,...]

        for i in range(out_ids):
            curr.execute('''UPDATE imgs SET score = (?) WHERE id = (?)''',(float(i),int(out_ids[i]),))


    if config['scoring_function'] == 'pred_euq_distance':
        #score as the euclidiean distance from origin in sortmax error space
        curr.execute('''SELECT MAX(step) FROM preds''')
        step = curr.fetchall()
        sql = '''SELECT l.img_id, l.output, i.label_num
                FROM ouputs AS l
                INNER JOIN imgs AS i ON l.img_id=i.id
                WHERE l.step=(?) AND i.used = 1 AND i.test = 0'''
        curr.execute(sql,(int(step),))
        results = np.array(curr.fetchall())

        #seperate the output and ids 
        outputs = results[:,1]
        ids = results[:,0]
        labels = results[:,2]

        dist = []
        #turn the softmax output into softmax error
        for i,output in enumerate(outputs):
            output[labels[i]] = 1 - output[labels[i]]
            output = np.sqrt(np.sum(np.square(output)))
            dist.append(output)

        for i in range(ids):
            curr.execute('''UPDATE imgs SET score = (?) WHERE id = (?)''',(float(dist[i]),int(ids[i]),))
        

    if config['scoring_function'] == 'SE_kdpp_sampling':
        #kdpp k-determantal point process with features based on softmax space
        #THIS IS A SAMPLING BASED METHOD TO COMPARE TO THUS IT UPDATES THE BATCH NUMBERS to 0 or -1
        #this produces 
        #TODO add method that smaples multiple batches (need to adapt batch_num variable to do this)

        curr.execute('''SELECT MAX(step) FROM preds''')
        step = curr.fetchall()
        sql = '''SELECT l.img_id, l.output, i.label_num
                FROM losses AS l
                INNER JOIN imgs AS i ON l.img_id=i.id
                WHERE l.step=(?) AND i.used = 1 AND i.test = 0'''
        curr.execute(sql,(int(step),))
        results = np.array(curr.fetchall())

        #seperate the output and ids 
        outputs = results[:,1]
        ids = results[:,0]
        labels = results[:,2]

        #convert to softmax space
        for i,output in enumerate(outputs):
            output[labels[i]] = 1 - output[labels[i]]
            outputs[i] = output

        # TODO work out if this is the right input
        L = features.T.dot(features)
        DPP = FiniteDPP('likelihood', **{'L': L})

        DPP.sample_exact_k_dpp(size=batch_size)
        batch = DPP.list_of_samples  #TODO find out what it outputs

        curr.execute('''UPDATE imgs SET batch_num = -1 ''')
        for i in batch:
            curr.execute('''UPDATE imgs SET batch_num = 0 WHERE id = (?)''',(int(ids[i]),))

    #add data used to each of the above functions
    #TODO other distance measures
    #TODO pred biggest move
    #TODO compressed space representations f1 layers ect
    conn.commit()





def pacing_functions(conn,config):
    #take the score and order in a specific way updating the batch numbers, if batch_num is -1 its not used
    #ensure that if the socoring function does this already its not done twice!!
    curr = conn.cursor()

    if config['pacing_function'] == 'hl':
        #high to low no removing
        curr.execute('''SELECT id FROM imgs WHERE used = 1 AND test = 0 ORDER BY score DESC ''')
        ids = np.array(curr.fetchall())
        b = 0
        for i,ind in enumerate(ids):
            if i % config['batch_size'] == 0 and i != 0:
                b+=1
            curr.execute('''UPDATE imgs SET batch_num = (?) WHERE id = (?)''',(int(b),int(ind),))
    
    if config['pacing_function'] == 'lh':
        #low to high no removing
        curr.execute('''SELECT id FROM imgs WHERE used = 1 AND test = 0 ORDER BY score ASC ''')
        ids = np.array(curr.fetchall())
        b = 0
        for i,ind in enumerate(ids):
            if i % config['batch_size'] == 0 and i != 0:
                b+=1
            curr.execute('''UPDATE imgs SET batch_num = (?) WHERE id = (?)''',(int(b),int(ind),))
    
    if config['pacing_function'] == 'mixed':
        #high low high low high low ...
        curr.execute('''SELECT id FROM imgs WHERE used=1 AND test=0 ORDER BY score ASC ''')
        ids = np.array(curr.fetchall())
        ids_mixed = [ids.pop(0) if x % 2 == 0 else ids.pop(-1) for x in range(len(ids))]
        b = 0
        for i,ind in enumerate(ids):
            if i % config['batch_size'] == 0 and i != 0:
                b+=1
            curr.execute('''UPDATE imgs SET batch_num = (?) WHERE id = (?)''',(int(b),int(ind),))

    if config['pacing_function'] != 'none':
        print('max batches to use next = ',b)
    conn.commit()












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
    


