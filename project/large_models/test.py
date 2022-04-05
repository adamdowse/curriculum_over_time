import pandas as pd
import numpy as np

df = pd.DataFrame(data={'i':[1,2,3,4,5],'val':[10,11,12,np.nan,14]})
print(df)

df_2 = pd.DataFrame(data={'i':[6,5,4,3],'new_val':[58,56,59,50]})
print(df_2)
df_2 = df_2.sort_values('new_val')
print(df_2)

df.update(df_2['i'],overwrite=True)
print(df)

print(df.fillna(method='ffill',axis=1))
#isna().sum() 
