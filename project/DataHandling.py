import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/com.docker.devenvironments.code/project/Data/meanlossgradient/OutputLayers/meanlossgradient_df',index_col="Unnamed: 0")

def diff (x):
    print(x)
    return((x[2]-x[1]) + (x[1]-x[0]))/2

#diff = lambda x: ((x[0]-x[1]) + (x[1]-x[2]))/2
diffs = df.iloc[:,3:].mean(axis=0).rolling(3).apply(diff,raw=True)

plt.hist(diffs)
plt.xlabel('avg gradient')
plt.ylabel('freq')
plt.savefig('hist')
plt.close()


plt.plot(diffs)
plt.savefig('gradplot')

#q05 = mod_df.iloc[:,3:].mean(axis=0).rolling(3).apply(diff,raw=True).quantile(q=0.05)
#m = mod_df.iloc[:,3:].mean(axis=0).rolling(3).apply(diff,raw=True).mean()
#med = mod_df.iloc[:,3:].mean(axis=0).rolling(3).apply(diff,raw=True).median()
            