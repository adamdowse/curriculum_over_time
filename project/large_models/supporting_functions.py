import tensorflow as tf
from keras import backend as K
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
import scipy
from scipy.spatial.distance import cdist
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
                                    relu array,
                                    H REAL,
                                    epoch INTEGER,
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
 
    cur = conn.cursor()
    cur.execute('''UPDATE imgs SET used = 0 WHERE test = 1''')
    cur.execute('''UPDATE imgs SET used = 0 WHERE test = 0''')
    cur.execute('''SELECT id FROM imgs WHERE test = 1''')
    test_ids = np.array(cur.fetchall())
    random.shuffle(test_ids)
    test_input = test_ids[:test_n]
    test_input = [(int(x[0]),) for x in test_input]
    #print(test_input)
    cur.executemany('''UPDATE imgs SET used = 1 WHERE id = (?)''',test_input)

    cur.execute('''SELECT id FROM imgs WHERE test = 0''')
    train_ids = np.array(cur.fetchall())
    random.shuffle(train_ids)
    train_input = train_ids[:train_n]
    train_input = [(int(x[0]),) for x in train_input]
    cur.executemany('''UPDATE imgs SET used = 1 WHERE id = (?)''',train_input)
 
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
    
def calc_entropy(preds):
    #input is softmax predictions of the batch
    #this is how the paper did it although i am unsure

    H = preds #[[0.1,0.3,...],[...],[...]]
    H = -np.array([[y*y for y in x] for x in H])
    H = np.sum(H, axis=1)
    return H

def DB_update(conn,info,step,X,Y,batch_loss,preds,relu=None,H=None,t=0):
    #update the db from batch infomation
    curr = conn.cursor()
    for i, x in enumerate(X[0].numpy()): 
        if t == 0:
        #print(float(batch_loss.numpy()[i]),int(step),int(x),int(info.batch_num)) 
            curr.execute('''INSERT INTO losses(loss,step,output,img_id,batch_num,epoch,relu,H) VALUES(?,?,?,?,?,?,?,?)''',
                (float(batch_loss[i]),
                int(step),
                np.array(preds[i]),
                int(x),
                int(info.batch_num),
                int(info.current_epoch),
                np.array(relu[i]),
                float(H[i])))
        else:
            curr.execute('''INSERT INTO losses(loss,step,output,img_id,batch_num,epoch) VALUES(?,?,?,?,?,?)''',
                (float(batch_loss[i]),
                int(step),
                np.array(preds[i]),
                int(x),
                int(info.batch_num),
                int(info.current_epoch)))


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
    else:
        #bin_edges = [x for x in np.linspace(0, 10, num=40)]
        #hist = np.histogram(results,bins=bin_edges)
        #print('logging results =',results)
        if len(results) > 1:
            results = np.delete(results, results > 3) #THIS IS USED TO keep hist bins from being to big
            wandb.log({name:wandb.Histogram(results)},step=step)


def log_acc(conn,test,step,batch_num=0,name='none'):
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


    #batch mutual infomation
    def batch_MI(imgs):
        #calculate the mutual infomation between all images in 'imgs' and sum for final result
        #imgs = [[img1.ravel],[imgs2.ravel]]

        def mutual_information(hgram):
            """ Mutual information for joint histogram
            """
            # Convert bins counts to probability values
            pxy = hgram / float(np.sum(hgram))
            px = np.sum(pxy, axis=1) # marginal for x over y
            py = np.sum(pxy, axis=0) # marginal for y over x
            px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
            # Now we can do the calculation using the pxy, px_py 2D arrays
            nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
            return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


        MI_matrix = np.zeros((len(imgs),len(imgs)))

        for i in range(1,len(imgs)):
            for j in range(i):
                hist_2d, x_edges, y_edges = np.histogram2d(imgs[i],imgs[j],bins=40)
                MI_matrix[i,j] = mutual_information(hist_2d) 

        return MI_matrix, np.sum(np.sum(MI_matrix))/(len(MI_matrix)*(len(MI_matrix)-1)/2)
    
    if test == 0:
        sql = '''   SELECT data 
                FROM imgs
                WHERE batch_num = (?) AND used = 1 AND test = 0'''
        curr.execute(sql,(batch_num,))
        imgs = curr.fetchall()
        imgs = [np.array(x).squeeze().ravel() for x in imgs]
        mat, t = batch_MI(imgs)
        wandb.log({'train_batch_MI_normed':t},step=step)


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
        curr.execute('''SELECT MAX(epoch) FROM losses''')
        epoch = curr.fetchone()[0]
        #print('Epoch is',epoch)
        #TODO CHANGE THIS AS IT GETS REALLY SLOW
        sql = '''   UPDATE imgs
                SET score = (SELECT loss FROM losses WHERE img_id = imgs.id AND epoch=(?))
                WHERE used = 1 AND test = 0 AND EXISTS (SELECT loss FROM losses WHERE img_id = imgs.id AND epoch=(?))
                '''

        curr.execute(sql,(epoch,epoch,))

    if config['scoring_function'] == 'loss_cluster':
        #cluster based on the loss 
        curr.execute('''SELECT MAX(epoch) FROM losses''')
        epoch = curr.fetchone()[0]
        curr.execute('''SELECT img_id, loss FROM losses WHERE img_id IN (SELECT id FROM imgs WHERE used = 1 AND test = 0) AND epoch = (?)''',(int(epoch),))
        results = np.array(curr.fetchall(),dtype=object)
        
        #km cluster
        km = cluster.MiniBatchKMeans(n_clusters=config['batch_size'])
        km = km.fit(results[:,1].reshape((-1,1)))
        cluster_data = km.predict(results[:,1].reshape((-1,1)))
        output = np.array([[results[i,0],cluster_data[i]] for i in range(len(cluster_data))]) #[[id,(0,31 cluster)],[id, clsuter]...]

        #pick to create batches
        #create the sub arrays
        for i in range(int(max(output[:,1])+1)): #0,1,2,...31
            ind = np.squeeze(np.argwhere(output[:,1]==i))
            if i == 0:
                cluster_split = {i:np.array(output[ind,0])}
            
            else:
                cluster_split[i] = np.atleast_1d(np.array(output[ind,0]))

        #cluster_split = {i:[ids]}

        #shuffle into batches. If one cluster runs out take from highest amount class
        #print(cluster_split)
        count = np.sum([len(cluster_split[i])  for i in range(len(cluster_split))])
        out_ids = []
        cluster_count = 0
        while count >0:
            #length of all dictionary values
            cluster_lens = [len(cluster_split[i]) for i in range(len(cluster_split))]
            if cluster_lens[cluster_count] != 0:
                #choose random value in the clsuter
                inds = cluster_split[cluster_count]
                r = random.randint(0,cluster_lens[cluster_count]-1)
                #add index to output and update dictionary values
                out_ids.append(inds[r])
                cluster_split[cluster_count] = np.delete(inds,r)
            else:
                #select highest cluster clount
                temp_cluster_count = np.argmax(cluster_lens)
                inds = cluster_split[temp_cluster_count]
                r = random.randint(0,cluster_lens[temp_cluster_count]-1)
                #add index to output and update dictionary values
                out_ids.append(inds[r])
                cluster_split[temp_cluster_count] = np.delete(inds,r)
            count -= 1
            cluster_count += 1
            if cluster_count > 31:
                cluster_count = 0

        #out_ids = [id, id ,id]  in batch order

        for i,index in enumerate(out_ids):
            curr.execute('''UPDATE imgs SET score = (?) WHERE id = (?)''',(float(i),int(index),))
        conn.commit()

    if config['scoring_function'] == 'loss_cluster_batches':
        #cluster so the batches used are clusters 
        #cluster based on the loss 
        #cluster based on the loss 
        curr.execute('''SELECT MAX(epoch) FROM losses''')
        epoch = curr.fetchone()[0]
        curr.execute('''SELECT img_id, loss FROM losses WHERE img_id IN (SELECT id FROM imgs WHERE used = 1 AND test = 0) AND epoch = (?)''',(int(epoch),))
        results = np.array(curr.fetchall(),dtype=object)

        #cluster number
        curr.execute('''SELECT COUNT(DISTINCT id) FROM imgs WHERE test = 0 AND used = 1''')
        data_amount = curr.fetchone()[0]
        if data_amount / config['batch_size'] == 0:
            num_batches = int(data_amount/config['batch_size'])
        else:
            num_batches = 1 + int(data_amount/config['batch_size'])

        #km cluster
        km = cluster.MiniBatchKMeans(n_clusters=num_batches)
        km = km.fit(results[:,1].reshape((-1,1)))
        cluster_data = km.predict(results[:,1].reshape((-1,1)))
        output = np.array([[results[i,0],cluster_data[i]] for i in range(len(cluster_data))]) #[[id,(0,31 cluster)],[id, clsuter]...]

        #pick to create batches
        #create the sub arrays
        for i in range(int(max(output[:,1])+1)): #0,1,2,...31
            ind = np.squeeze(np.argwhere(output[:,1]==i))
            if i == 0:
                cluster_split = {i:np.array(output[ind,0])}
            
            else:
                cluster_split[i] = np.atleast_1d(np.array(output[ind,0]))

        #cluster_split = {i:[ids]}

        #shuffle into batches. If one cluster runs out take from highest amount class
        count = np.sum([len(cluster_split[i])  for i in range(len(cluster_split))])
        out_ids = []
        cluster_count = 0
        
        while count >0:
            #length of all dictionary values
            cluster_lens = [len(cluster_split[i]) for i in range(len(cluster_split))]
            #print(cluster_lens,cluster_count)
            if cluster_lens[cluster_count] != 0:
                #choose random value in the clsuter
                inds = cluster_split[cluster_count]
                r = random.randint(0,cluster_lens[cluster_count]-1)
                #add index to output and update dictionary values
                out_ids.append(inds[r])
                cluster_split[cluster_count] = np.delete(inds,r)
            else:
                #select highest cluster clount
                temp_cluster_count = np.argmax(cluster_lens)
                inds = cluster_split[temp_cluster_count]
                r = random.randint(0,cluster_lens[temp_cluster_count]-1)
                #add index to output and update dictionary values
                out_ids.append(inds[r])
                cluster_split[temp_cluster_count] = np.delete(inds,r)
            count -= 1
            cluster_count += 1
            if cluster_count > num_batches-1:
                cluster_count = 0

        #out_ids = [id, id ,id]  in batch order

        for i,index in enumerate(out_ids):
            curr.execute('''UPDATE imgs SET score = (?) WHERE id = (?)''',(float(i),int(index),))
        conn.commit()

    if config['scoring_function'] == 'pred_cluster':
        #cluster so the batches used are clusters 
        #cluster based on the softmax outputs
        #cluster based on the loss 
        curr.execute('''SELECT MAX(epoch) FROM losses''')
        epoch = curr.fetchone()[0]
        curr.execute('''SELECT img_id, output FROM losses WHERE img_id IN (SELECT id FROM imgs WHERE used = 1 AND test = 0) AND epoch = (?)''',(int(epoch),))
        results = np.array(curr.fetchall(),dtype=object) #updated?
        results_1 = [x[1] for x in results]
        
        #km cluster
        km = cluster.MiniBatchKMeans(n_clusters=config['batch_size'])
        km = km.fit(results_1)
        cluster_data = km.predict(results_1)
        output = np.array([[results[i,0],cluster_data[i]] for i in range(len(cluster_data))]) #[[id,(0,31 cluster)],[id, clsuter]...]

        #pick to create batches
        #create the sub arrays
        for i in range(int(max(output[:,1])+1)): #0,1,2,...31
            ind = np.squeeze(np.argwhere(output[:,1]==i))
            if i == 0:
                cluster_split = {i:np.array(output[ind,0])}
            else:
                cluster_split[i] = np.atleast_1d(np.array(output[ind,0]))

        #cluster_split = {i:[ids]}

        #shuffle into batches. If one cluster runs out take from highest amount class
        count = np.sum([len(cluster_split[i])  for i in range(len(cluster_split))])
        out_ids = []
        cluster_count = 0
        while count >0:
            #length of all dictionary values
            cluster_lens = [len(cluster_split[i]) for i in range(len(cluster_split))]
            if cluster_lens[cluster_count] != 0:
                #choose random value in the clsuter
                inds = cluster_split[cluster_count]
                r = random.randint(0,cluster_lens[cluster_count]-1)
                #add index to output and update dictionary values
                out_ids.append(inds[r])
                cluster_split[cluster_count] = np.delete(inds,r)
            else:
                #select highest cluster clount
                temp_cluster_count = np.argmax(cluster_lens)
                inds = cluster_split[temp_cluster_count]
                r = random.randint(0,cluster_lens[temp_cluster_count]-1)
                #add index to output and update dictionary values
                out_ids.append(inds[r])
                cluster_split[temp_cluster_count] = np.delete(inds,r)
            count -= 1
            cluster_count += 1
            if cluster_count > 31:
                cluster_count = 0

        #out_ids = [id, id ,id]  in batch order

        for i,index in enumerate(out_ids):
            curr.execute('''UPDATE imgs SET score = (?) WHERE id = (?)''',(float(i),int(index),))
        conn.commit()

    if config['scoring_function'] == 'pred_euq_distance':
        #score as the euclidiean distance from origin in sortmax error space
        curr.execute('''SELECT MAX(epoch) FROM losses''')
        epoch = curr.fetchone()[0]
        sql = '''SELECT l.img_id, l.output, i.label_num
                FROM losses AS l
                INNER JOIN imgs AS i ON l.img_id=i.id
                WHERE l.epoch=(?) AND i.used = 1 AND i.test = 0'''
        curr.execute(sql,(int(epoch),))

        f_all = curr.fetchall()
        ids = np.array([x[0] for x in f_all])
        outputs = np.array([x[1] for x in f_all])
        labels = np.array([x[2] for x in f_all])

        dist = []
        #turn the softmax output into softmax error
        for i,output in enumerate(outputs):
            output[labels[i]] = 1 - output[labels[i]]
            output = np.sqrt(np.sum(np.square(output)))
            dist.append(output)

        for i,id in enumerate(ids):
            curr.execute('''UPDATE imgs SET score = (?) WHERE id = (?)''',(float(dist[i]),int(id),))
        
    if config['scoring_function'] == 'SE_kdpp_sampling':
        #kdpp k-determantal point process with features based on softmax space
        #THIS IS A SAMPLING BASED METHOD TO COMPARE TO THUS IT UPDATES THE BATCH NUMBERS to 0 or -1
        #this produces 
        #TODO add method that smaples multiple batches (need to adapt batch_num variable to do this)
        #TODO THIS DOSENT WORK AT THE MOMENT NEEDS A LOT OF WORK TO IMPLEMENT
        sql =   '''
                SELECT l.img_id, l.output, i.label_num
                FROM losses AS l
                INNER JOIN imgs AS i ON l.img_id=i.id
                WHERE i.used = 1 AND i.test = 0
                GROUP BY l.img_id
                HAVING l.step = max(l.step) 
                '''
        curr.execute(sql)
        f_all = curr.fetchall()

        #seperate the output and ids 
        ids = np.array([x[0] for x in f_all])
        outputs = np.array([x[1] for x in f_all])
        labels = np.array([x[2] for x in f_all])
        print(len(ids))
        print(len(outputs))
        print(len(labels))
        #convert to softmax space
        for i,output in enumerate(outputs):
            output[labels[i]] = 1 - output[labels[i]]
            outputs[i] = output
        
        outputs = np.array(outputs)
        #for 
        L = np.dot(outputs,outputs.transpose()) #distance matrix
        print(L)
        print(L.shape)
        #convert to K 
        K = np.identity(len(L)) - np.linalg.inv(L+np.identity(len(L)))
        print(K)
        print(K.shape)

        dpp_K = FiniteDPP('correlation', **{'K': K})
        dpp_K.sample_exact_k_dpp(size=config['batch_size'])
        batch = DPP.list_of_samples  #TODO find out what it outputs
        print(batch)

        pnt()

        #flatten low values to tolerance of 1e-8
        #L[L < 1e-8] = 1e-8
        #L = L*10000
        print(np.linalg.eig(L))
        DPP = FiniteDPP('likelihood', **{'L': L})
        print((L < 0).sum())
        DPP.sample_exact_k_dpp(size=config['batch_size'])
        batch = DPP.list_of_samples  #TODO find out what it outputs
        print(batch)
        curr.execute('''UPDATE imgs SET batch_num = -1 ''')
        for i in batch:
            curr.execute('''UPDATE imgs SET batch_num = 0 WHERE id = (?)''',(int(ids[i]),))

    if config['scoring_function'] == 'submodular_sampling':
        #based on a combination of criteria greedly add to a batch to maximize the submodular score
        #based on https://github.com/VamshiTeja/SMDL/blob/master/lib/samplers/submodular.py
        
        def compute_u_score(entropy,indexes,alpha=1):
            #Compute the Uncertainity Score: The point that makes the model most confused, should be preferred
            if len(indexes) == 0:
                return 0
            else:
                u_score = alpha*entropy[indexes]
                return u_score
        
        def compute_r_score(penultimate_activations, subset_indices, index_set, alpha=0.2, distance_metric='gaussian'):
            #redundancy score: The greater the value of minimum distance between points in a batch is better
            if len(subset_indices) == 0:
                return 0
            else:
                index_p_acts = penultimate_activations[np.array(index_set)] #vlaues out of the subset
                subset_p_acts = penultimate_activations[np.array(subset_indices)] #values in the current subset
                if(distance_metric=='gaussian'):
                    #distance measure
                    pdist = cdist(index_p_acts, subset_p_acts, metric='sqeuclidean')
                    #r_score = scipy.exp(-pdist / (0.5) ** 2)
                    r_score = alpha * np.min(pdist, axis=1)
                    return r_score
                #can add other metrics here
                else:
                    print('NO defined metric ERROR')
        
        def compute_md_score(penultimate_activations, index_set, class_mean, alpha=0.2, distance_metric='gaussian'):
            """
            Computes Mean Divergence score: The new datapoint should be close to the class mean
            :param penultimate_activations:
            :param index_set:
            :param class_mean:
            :param alpha:
            :return: list of scores for each index item
            """

            if(distance_metric=='gaussian'):
                #distance measure
                pen_act = penultimate_activations[np.array(index_set)]
                md_score = alpha * cdist(pen_act, np.array([np.array(class_mean)]), metric='sqeuclidean')
                #md_score = scipy.exp(-md_score / (0.5) ** 2)
                return md_score.squeeze()
            else:
                print('NO defined metric ERROR')
            
        def compute_coverage_score(normalised_penultimate_activations, subset_indices, index_set, alpha=0.5):
            """
            :param penultimate_activations:
            :param subset_indices:
            :param index_set:
            :return: g(mu(S))
            """
            if(len(subset_indices)==0):
                score_feature_wise = np.sqrt(normalised_penultimate_activations[index_set])
                scores = np.sum(score_feature_wise, axis=1)
                return alpha*scores
            else:
                penultimate_activations_index_set =  normalised_penultimate_activations[index_set]
                subset_indices_scores = np.sum(normalised_penultimate_activations[subset_indices],axis=0)
                sum_subset_index_set = subset_indices_scores + penultimate_activations_index_set
                score_feature_wise = np.sqrt(sum_subset_index_set)
                scores = np.sum(score_feature_wise,axis=1)
                return alpha*scores

        def normalise(A):
            return A/np.sum(A)

        def get_subset_indices(index_set_input, penultimate_activations, normalised_penultimate_activations, entropy,  subset_size, alpha_1, alpha_2, alpha_3, alpha_4):

            #print('reached subset selection')
            index_set = index_set_input
            subset_indices = []     # Subset of indices. Keeping track to improve computational performance.

            class_mean = np.mean(penultimate_activations, axis=0)
            #print(class_mean)

            subset_size = min(subset_size, len(index_set)) # this deals with the last selections of data
            for i in range(0, subset_size):

                u_scores = compute_u_score(entropy, list(index_set), alpha=alpha_1)
                r_scores = compute_r_score(penultimate_activations, list(subset_indices), list(index_set), alpha=alpha_2)
                md_scores = compute_md_score(penultimate_activations, list(index_set), class_mean, alpha=alpha_3)
                coverage_scores = compute_coverage_score(normalised_penultimate_activations, subset_indices, index_set, alpha=alpha_4)
                #if i > 0:
                #    print(u_scores.shape)
                #    print(r_scores.shape)
                #    print(md_scores.shape)
                #    print(coverage_scores.shape)

                scores = normalise(np.array(u_scores)) + normalise(np.array(r_scores)) + normalise(np.array(md_scores)) + normalise(np.array(coverage_scores))
                #print(scores.shape)
                best_item_index = np.argmax(scores)
                subset_indices.append(index_set[best_item_index])
                index_set = np.delete(index_set, best_item_index, axis=0)

                # log('Processed: {0}/{1} exemplars. Time taken is {2} sec.'.format(i, subset_size, time.time()-now))

            return subset_indices
            
        
        #local hyperparams

        a_1 = 0.2
        a_2 = 0.1
        a_3 = 0.5
        a_4 = 0.2

        #get the size of the avalible indexes
        #set_size = len(self.index_set)
        sql =   '''
            SELECT l.img_id, l.output, i.label_num, l.relu, l.H
            FROM losses AS l
            INNER JOIN imgs AS i ON l.img_id=i.id
            WHERE i.used = 1 AND i.test = 0
            GROUP BY l.img_id
            HAVING l.step = max(l.step) 
            '''
        curr.execute(sql)
        f_all = curr.fetchall()

        #seperate the output and ids 
        ids = np.array([x[0] for x in f_all])
        outputs = np.array([x[1] for x in f_all]) #softmax
        labels = np.array([x[2] for x in f_all])
        relu_outputs = np.array([x[3] for x in f_all]) #relu
        H = np.array([x[4] for x in f_all]) #entropy
        indices = [x for x in range(len(ids))]


        batch_num = 0
        while len(indices) > 0:
            subset_indices = get_subset_indices(indices, relu_outputs, outputs, H, config['batch_size'],a_1,a_2,a_3,a_4)

            #add the batch num
            for i in subset_indices:
                curr.execute('''UPDATE imgs SET batch_num = (?) WHERE id = (?)''',(int(batch_num),int(ids[i])))

            #Subset selection without replacement.
            for item in subset_indices:    
                indices.remove(item)
            
            batch_num += 1

        #if detailed_logging:
        #    log('The selected {0} indices (second level): {1}'.format(len(subset_indices), subset_indices))
        









    #add data used to each of the above functions
    #TODO other distance measures
    #TODO pred biggest move
    #TODO compressed space representations f1 layers ect
    conn.commit()

def k_dpp_sample(input_array,size):
    #not working
    L = input_array @ input_array.transpose()

    #need to write own version of this that includes the space reduction methods
    DPP = FiniteDPP('likelihood', **{'L': L})
    print((L < 0).sum())
    DPP.sample_exact_k_dpp(size=config['batch_size'])
    batch = DPP.list_of_samples 


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
    


