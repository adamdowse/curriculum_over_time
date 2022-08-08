#imports
import numpy as np
import tensorflow_datasets as tfds
import time
from multiprocessing import Process, Value, Array

#PLEASE BE CARFUL WITH THE HUGE ARRAYS YOU ARE ABOUT TO CREATE
#collect dataset in same way as run main does via database


def mutual_information_between_2(MI_mat,n_imgs,func_input):
    """ Mutual information for joint histogram
    MI_mat : the shared mutual information to write to 
    i,j : the location to write to
    img1,img2 : the raveled images as lists
    """
    #unpack
    img1 = func_input[0]
    img2 = func_input[1]
    i    = func_input[2]
    j    = func_input[3]
    print(i[2])

    #convert the images into histograms
    hgram, x_edges, y_edges = np.histogram2d(img1,img2,bins=40)

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    MI_mat[i*n_imgs+j] = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])) 

def mutual_information_old(img1,img2):
    """ Mutual information for joint histogram
    MI_mat : the shared mutual information to write to 
    i,j : the location to write to
    img1,img2 : the raveled images as lists
    """

    #convert the images into histograms
    hgram, x_edges, y_edges = np.histogram2d(img1,img2,bins=40)

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])) 

if __name__ == '__main__':
    #get data
    ds, ds_info = tfds.load('mnist',with_info=True,shuffle_files=False,as_supervised=True,split='train')

    #take the first n images to use as a proxi and test
    n_imgs = 10
    img_list = []
    c = 0
    for img,label in ds:
        img_list.append(img.numpy().ravel().tolist())
        c+=1
        if c == n_imgs:
            break
    #img_list = [[unraveled image],[...],...]
    

    #multiprocessing version__
    #create the list fof inputs
    process_input = []
    for i in range(n_imgs):
        for j in range(n_imgs):
            if i<j:
                process_input.append((img_list[i],img_list[j],i,j))
    #process_input = [([img_0],[img_1],i,j)]
    print(process_input[0][2])

    t = time.time()
    MI_mat = Array('i',[0 for x in range(n_imgs**2)])  #index via x*n_imgs+y

    p = Process(target=mutual_information_between_2,args=(MI_mat,n_imgs,process_input))
    p.start()
    p.join()
    print(time.time() - t)
    print(MI_mat[:])

    #no multiprocessing

    t = time.time()
    MI_matrix = np.zeros((n_imgs,n_imgs),dtype='int8')
    for i,img1 in enumerate(img_list):
        for j,img2 in enumerate(img_list):
            if i<j:
                MI_matrix[i,j] = mutual_information_old(img1,img2)

    print(time.time() - t)
    print(MI_matrix)
    


    #t = time.time()
    #calc massive matrix of mutual information between images
    #MI_matrix = np.zeros((n_imgs,n_imgs),dtype='int8')
    #print('Mat size: ',MI_matrix.nbytes/(1*10**9),'Gbs')
    
    #c = 0
    #for i,[img1,label1] in enumerate(ds):
    #    for j,[img2,label2] in enumerate(ds): 
    #        #check to see if the pair has been done before
    #        if i<j:
    #            MI_matrix[i,j] = mutual_information_between_2(img1.numpy().ravel(),img2.numpy().ravel())
    #            c += 1
    #        
    #        if c == 100:
    #            print((time.time() - t)/100)
   # 
   #     if i % 500 == 0:
   #         print('i = ',i)



    #save the MI matrix to a csv to avoid recompute
  
    #TODO change this to a full address and put it in /MI/ so its git ignored
    np.savetxt("MNIST_MI.csv", MI_matrix, delimiter=",")

    #BMI formula np.sum(np.sum(MI_matrix))/(len(MI_matrix)*(len(MI_matrix)-1)/2)