import pandas as pd
import numpy as np

a = {}
keys = ['1','2','3','4']
for i in range(len(keys)):
    a['h'+keys[i]] = i+10

print(a)