import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df_1 = pd.read_csv('/com.docker.devenvironments.code/project/Data/modeldepth/1l_3f3c_df')
#df_2 = pd.read_csv('/com.docker.devenvironments.code/project/Data/modeldepth/2l5s_10f3c_df',index_col='Unnamed: 0')
df_3 = pd.read_csv('/com.docker.devenvironments.code/project/Data/modeldepth/3l5s_3f3c_df')
#df_5 = pd.read_csv('/com.docker.devenvironments.code/project/Data/modeldepth/5l5s_10f3c_df',index_col='Unnamed: 0')
df_10 = pd.read_csv('/com.docker.devenvironments.code/project/Data/modeldepth/10l5s_3f3c_df')
df_20 = pd.read_csv('/com.docker.devenvironments.code/project/Data/modeldepth/20l8s_3f3c_df')

plt.plot(df_1.iloc[:,4:].rolling(2).var().mean(),label='1 layer')
#plt.plot(df_2.iloc[:,11:].var(axis=0),label='2 layers')
plt.plot(df_3.iloc[:,4:].rolling(2).var().mean(),label='3 layers')
#plt.plot(df_5.iloc[:,11:].var(axis=0),label='5 layers')
plt.plot(df_10.iloc[:,4:].rolling(2).var().mean(),label='10 layers')
plt.plot(df_20.iloc[:,4:].rolling(2).var().mean(),label='20 layers')
plt.xlabel('epoch')
plt.ylabel('Rolling Var Loss')
plt.yscale('log')
plt.legend()
plt.savefig('Data/modeldepth/rollingvarloss3')
plt.close()


