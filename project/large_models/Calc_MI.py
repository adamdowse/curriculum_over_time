#imports
import numpy as np
import tensorflow_datasets as tfds
import time
from multiprocessing import Pool, Process, Value, Array

#PLEASE BE CARFUL WITH THE HUGE ARRAYS YOU ARE ABOUT TO CREATE

def mutual_information(func_in):
    """ Mutual information for joint histogram
    MI_mat : the shared mutual information to write to 
    i,j : the location to write to
    img1,img2 : the raveled images as lists
    """
    #unpack vars
    img1,img2,i,j = func_in
    

    #convert the images into histograms
    hgram, x_edges, y_edges = np.histogram2d(img1,img2,bins=40)

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return (np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])),i,j )

if __name__ == '__main__':
    #get data
    ds, ds_info = tfds.load('mnist',with_info=True,shuffle_files=False,as_supervised=True,split='train')

    #take the first n images to use as a proxi and test
    n_imgs = 6000
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

    #pool multiprocessing__
    t = time.time()
    with Pool(3) as p:
        map_out = p.map(mutual_information,process_input)
    MI_mat = np.zeros((n_imgs,n_imgs))
    for mi,i,j in map_out:
        MI_mat[j,i] = mi
    print(time.time()-t)


    #save the MI matrix to a csv to avoid recompute
  
    #TODO change this to a full address and put it in /MI/ so its git ignored
    np.savetxt("MI/MNIST_6000.csv", MI_mat, delimiter=",")

    #BMI formula np.sum(np.sum(MI_matrix))/(len(MI_matrix)*(len(MI_matrix)-1)/2)