import pandas as pd
import numpy as np
#import supporting_functions as sf
import sqlite3
import supporting_functions as sf
from sqlite3 import Error
import random
from dppy.finite_dpps import FiniteDPP
import math
import matplotlib.pyplot as plt


a = [[1,3,2],[1,3,3],[5,1,1],[2,2,2]]

#there way
def SM(a):
    out = []
    for i_a in a:
        i_a = [math.exp(x) for x in i_a]
        s = sum(i_a)
        o = [x/s for x in i_a]
        out.append(o)
    return out

a_SM = SM(a)
H_1 = [x**2 for x in a_SM]
H_1 = -np.array(H1)
H_1 = np.sum(H_1, axis=1)
print(H_1)

H_2 = [[-math.log(y) for y in x] for x in a_SM]

print(H_2)


pnt()


def l2 (a,b):
    c = sum([(a[i] - b[i])**2 for i in range(len(a)) ])
    return np.sqrt(c)

print(np.dot(np.array([1,2,3]).transpose(),np.array([1,2,3])))


a = np.array([[1],[2],[3],[2]])
print(a.shape)
L = np.dot(a.transpose(),a)
print(L)
print(L.shape)


a = np.array([[1,2,3,2],[4,3,2,1],[5,6,4,2],[1,1,1,3],[5,5,3,4]])
print(a.shape)
L = np.dot(a.transpose(),a)
print(L)
print(L.shape)

DPP = FiniteDPP('likelihood', **{'L': L})
DPP.sample_exact_k_dpp(size=2)
batch = DPP.list_of_samples  #TODO find out what it outputs
print(batch)



ptn()


#image mutual information
conn = sqlite3.connect('/com.docker.devenvironments.code/project/large_models/DBs/mnist.db',detect_types=sqlite3.PARSE_DECLTYPES)
curr = conn.cursor()
#setup convertion functions for storing in db
sqlite3.register_adapter(np.ndarray, sf.array_to_bin)# Converts np.array to TEXT when inserting
sqlite3.register_converter("array", sf.bin_to_array) # Converts TEXT to np.array when selecting

sql =   '''
        SELECT l.img_id, l.output, i.label_num
        FROM losses AS l
        INNER JOIN imgs AS i ON l.img_id=i.id
        WHERE i.used = 1 AND i.test = 0
        GROUP BY l.img_id
        HAVING l.step = max(l.step) 
        '''
##WHERE filters the raw table rows, HAVING filters the resulting grouped rows
curr.execute(sql)
print(len(curr.fetchall()))


pnt()
imgs = []
z = [2,7,12,19,36,40,53]
z = [2,3,4,5,6,7,8]

curr.execute('''SELECT data FROM imgs WHERE batch_num = (?) AND used = 1 AND test = 0 ''',(1,)) #2
imgs = curr.fetchall()
imgs = [np.array(x).squeeze().ravel() for x in imgs]
print(imgs)

curr.execute('''SELECT data FROM imgs WHERE id = 2''') #0
b = np.array(curr.fetchone()[0]).squeeze()

curr.execute('''SELECT data FROM imgs WHERE id = 7''') #0
c = np.array(curr.fetchone()[0]).squeeze()



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


#imgs = [a.ravel(),b.ravel(),c.ravel()]
mat, t = batch_MI(imgs)

print(mat,t)