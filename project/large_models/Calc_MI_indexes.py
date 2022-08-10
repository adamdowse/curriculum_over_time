#given the huge MI matrix that has been stored as a csv generate a list of indexes 
#that tries to set different BMI levels across an epoch.

#Imports
import numpy as np
import scipy


def merge_csvs(csv_path,n_files):
    '''
    csv_path : the main path without the number and .csv at the end
    '''
    #loop through the files
    print('Building large CSV from small files...')
    csv = np.loadtxt(open(csv_path+'0.csv', "rb"), delimiter=",")
    for i in range(1,n_files):
        temp_csv = np.loadtxt(open(csv_path+str(i)+'.csv', "rb"), delimiter=",")
        csv = np.vstack((csv, temp_csv))
        print(csv.shape)
    
    return csv
        



if __name__ == '__main__':
    #Current thouhgts on this is 
    #   - to use a greedy swapping method
    #   - aim to get max and min avg BMI first to see how much variance can be achieved
    #   - need to not use the same batches every epoch


    #collect the MI data
    MI_mat = merge_csvs('/com.docker.devenvironments.code/project/large_models/MI_small/MNIST_6000_',12)

    #finding the max



