#take a large csv and split it into chucks that are below the git limit

import numpy as np
import pandas as pd


#file loc to split
file_name = '/com.docker.devenvironments.code/project/large_models/MI/MNIST_6000.csv'

#import the large csv file
large_df = pd.read_csv(file_name,sep=',',header=None)

print(large_df.head())

