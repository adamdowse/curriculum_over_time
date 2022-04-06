import pandas as pd
import numpy as np

df = pd.DataFrame(data={'i':[1,2,3,4,5],'val':[10,11,12,np.nan,14]})
print(df)

df = df.sort_values('val')
print(df)

new_df = pd.DataFrame(data={'val':[1,2,3,4]})
print(df)

df['val'] = [1,2,3,4,np.nan]
print(df)