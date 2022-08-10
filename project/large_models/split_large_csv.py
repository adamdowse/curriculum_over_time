#take a large csv and split it into chucks that are below the git limit

import numpy as np
import pandas as pd


#file loc to split
file_name = '/com.docker.devenvironments.code/project/large_models/MI/MNIST_6000.csv'
destination_name = '/com.docker.devenvironments.code/project/large_models/MI_small/MNIST_6000'

#import the large csv file
large_df = pd.read_csv(file_name,sep=',',header=None)

print(large_df.head())

split_size = 500
c = 0
i = 0
tot_rows = 6000
while c < tot_rows:
    if c + split_size > tot_rows:
        large_df.iloc[c:].to_csv(destination_name+'_'+str(i)+'.csv',sep=',',index=False,header=False)
    else:
        large_df.iloc[c:c+split_size].to_csv(destination_name+'_'+str(i)+'.csv',sep=',',index=False,header=False)
    i+=1
    c+=split_size
    print(c)