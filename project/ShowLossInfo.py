import pandas as pd
import matplotlib.pyplot as plt

name = 'meanlossgradient'
root_dir = '/com.docker.devenvironments.code/project/Data/'+name+'/OutputLayers/'

#Collect all the run infomation
df_dataused = pd.read_csv(root_dir+name+'_dataused',header=None)
df_dataused.columns = ['data_used']
df_loss = pd.read_csv(root_dir+name+'_df',index_col="Unnamed: 0")
df_testacc = pd.read_csv(root_dir+name+'_test_acc',header=None)
df_testacc.columns = ['test_acc']

#print(df_dataused.head())
#print(df_loss.head())
print(df_testacc.head())

max_epochs = len(df_loss.iloc[1,3:])


#PLot the infomation to 3 graphs
plt.plot(range(max_epochs),df_testacc.test_acc)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.savefig(root_dir+'Test_Acc.jpg')
plt.close()

plt.plot(range(max_epochs),df_loss.iloc[:,3:].mean(axis=0),label='Mean')
plt.plot(range(max_epochs),df_loss.iloc[:,3:].max(axis=0),label='Max')
plt.plot(range(max_epochs),df_loss.iloc[:,3:].min(axis=0),label='Min')
plt.plot(range(max_epochs),df_loss.iloc[3,3:],label='Point 1')
#plt.plot(range(max_epochs),df_loss.iloc[10,3:],label='Point 2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig(root_dir+'Loss.jpg')
plt.close()


#rolling means or variances TODO
plt.plot(range(max_epochs),df_loss.iloc[:,3:].rolling(3,axis=1).var().mean(axis=0),label='Mean')
plt.plot(range(max_epochs),df_loss.iloc[:,3:].rolling(3,axis=1).var().max(axis=0),label='Max')
plt.plot(range(max_epochs),df_loss.iloc[:,3:].rolling(3,axis=1).var().min(axis=0),label='Min')
plt.plot(range(max_epochs),df_loss.iloc[3,3:].rolling(3).var(),label='Point 1')
#plt.plot(range(max_epochs),df_loss.iloc[10,3:].rolling(3).var(),label='Point 2')
plt.xlabel('Epoch')
plt.ylabel('Rolling Loss Variance')
plt.yscale('log')
plt.legend()
plt.savefig(root_dir+'Loss_rolling_var.jpg')
plt.close()

plt.plot(range(max_epochs),df_loss.iloc[:,3:].var(axis=0),label='Variance')
plt.xlabel('Epoch')
plt.ylabel('Loss Variance Across Images')
plt.yscale('log')
plt.savefig(root_dir+'Loss_var.jpg')
plt.close()

plt.plot(range(max_epochs),df_dataused.data_used,label='DataUsed')
plt.xlabel('Epoch')
plt.ylabel('Data Used')
plt.savefig(root_dir+'DataUsed.jpg')
plt.close()
