import pandas as pd
import wandb
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
#os.environ['WANDB_DISABLED'] = 'true'
wandb.login()
wandb.init(job_type="analysis")
api = wandb.Api()

#collect the run names
run_df = pd.read_csv("0.01_mnist.csv")
print(run_df.head())

path = 'adamdowse/new_COT/'
names = run_df['ID'].to_numpy()
names = [path+x for x in names]
print(names)

#collect multiple runs
batches_per_epoch = 88
t_epochs = 10 #should be 10

#for 0.01 
#bpe = 9
#te = 50
#for a total of 450 steps

dic = {}
for run_name in names:
    run = api.run(run_name)
    history = run.scan_history()
    MI = [row["train_batch_MI_normed"] for row in history]
    print(len(MI))
    MI = MI[:(600)] #TEMP NEED TO REDO BROKEN RUNS
    #print(MI)
    dic[run_name] = MI

pnt()
df = pd.DataFrame(dic)
df.columns = run_df['ID']
print(df.head())

mean_df = {}
#loop through the types of scoring funciton and build a mean column
for sf in pd.unique(run_df['scoring_function']):
    ids = run_df[run_df['scoring_function'] == sf].copy()
    ids = ids['ID'].to_numpy().copy()
    print(ids)
    mean_df[sf] = df[ids].mean(axis=1).to_numpy().copy()
    #print(df[ids].mean(axis=1))


mean_df = pd.DataFrame(mean_df)
print(mean_df.head())
print(mean_df.mean(axis=0))

#figure section
fig, ax = plt.subplots()#figsize=(10, 8)

lw = 1
a = 0.5
r = 10

plt.plot(mean_df['submodular_sampling'].rolling(r).mean(),label='SM',alpha=a,color='purple',linewidth=lw)
plt.plot(mean_df['random'].rolling(r).mean(),label='Random',alpha=a,color='green',linewidth=lw)
plt.plot(mean_df['loss_cluster'].rolling(r).mean(),label='Loss Clustering',alpha=a,color='orange',linewidth=lw)
plt.plot(mean_df['pred_cluster'].rolling(r).mean(),label='SMC',alpha=a,color='blue',linewidth=lw)
plt.plot(mean_df['last_loss'].rolling(r).mean(),label='Last Loss',alpha=a,color='grey',linewidth=lw)
plt.plot(mean_df['loss_cluster_batches'].rolling(r).mean(),label='Loss CLustering Batches',alpha=a,color='red',linewidth=lw)
e = np.arange(0,t_epochs+1) * batches_per_epoch
plt.xticks(e,np.arange(0,t_epochs+1))
ax.set_xlim([0,(t_epochs+1)*batches_per_epoch])
ax.set_xlabel('Epoch')
ax.set_ylabel('Normalized Mutual Information Per Batch')
plt.legend()
ax.grid(which='major', color='#CCCCCC', linestyle='--')
plt.savefig('MI_01_mnist',bbox_inches='tight')


