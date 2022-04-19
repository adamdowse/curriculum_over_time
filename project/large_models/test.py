import pandas as pd
import numpy as np
from sklearn import linear_model


n = 8
#print([x for x in range(n)])
#print([x % 2 for x in range(n)])
a = [x for x in range(n)]
print(a)

x = []
for i in range(n): #[0,1,2,3,4,5,6,7]
    print( i % 2)
    if i % 2 == 0:
        x.append(a.pop(0))
    else:
        x.append(a.pop(-1))
print(x)