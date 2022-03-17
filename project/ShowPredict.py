import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('/com.docker.devenvironments.code/project/Data/rank/OutputLayers/rank_df',index_col='Unnamed: 0')
full_df = pd.read_csv('/com.docker.devenvironments.code/project/Data/rank/OutputLayers/rank_fulldf',index_col='Unnamed: 0')
pred_df = pd.read_csv('/com.docker.devenvironments.code/project/Data/rank/OutputLayers/rank_preddf',index_col='Unnamed: 0')

print(df)
print(full_df)
print(pred_df)

plt.plot(df.iloc[:,4:].mean(axis=0),label='df')
plt.plot(full_df.iloc[:,4:].mean(axis=0),label='full_df')
plt.plot(pred_df.iloc[:,4:].mean(axis=0),label='pred_df')
plt.legend()
plt.savefig('t1')
plt.close()


