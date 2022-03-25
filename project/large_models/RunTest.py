import pandas as pd
import tensorflow as tf
import supporting_functions as sf
import supporting_models as sm


#AIMS:
# Try the predictive loss method on cifar 10 dataset and compare the results
# Setup the test bed to do multiple tests
# make variables generalisable

@tf.function
def train_step(imgs,labels):
    with tf.GradientTape() as tape:
        preds = model(imgs,training=True)
        batch_loss = loss_func(labels,preds)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    m_loss = tf.math.reduce_mean(batch_loss)
    train_loss(m_loss)
    train_acc_metric(labels,preds)
    return batch_loss

@tf.function
def test_step(imgs, labels):
    preds = model(imgs, training=False)
    t_loss = loss_func(labels,preds)
    m_loss = tf.math.reduce_mean()

    test_loss(t_loss)
    test_acc_metric(labels, preds)

class Info_class :
    #variables for test
    max_epochs = 100
    current_epoch = 0
    learning_rate = 0.01
    batch_size = 16
    scoring_function = 'normal'

    #if datset name is a path use that path
    dataset_name = 'mnist'
    data_path = 'datasets/'
    save_model_path = 'saved_models/'
    log_path =  'logs/'

    img_shape = 0
    dataused = [] 
    class_names = []

info = Info_class()

# initilise the dataframe to train on and the test dataframe
_, test_ds, df_train_losses, info = sf.init_data(info)

#build and load model, optimizer and loss functions
model = sm.Simple_CNN(info.num_classes)
optimizer = keras.optimizers.SGD(learning_rate=info.learning_rate),
loss_func = keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)

#setup metrics to record: [train loss, test loss, train acc, test acc, dataused]
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = keras.metrics.CategoricalAccuracy()
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc_metric = keras.metrics.CategoricalAccuracy()
dataused = []

#Tensorboard Setup
info = '_E' +str(max_epochs)+'_B'+str(batch_size)
train_log_dir = info.log_path + name + info + '/train'
test_log_dir = info.log_path + name + info + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

print('Setup Complete, Starting training:')

for info.current_epoch in range(info.max_epochs): #this may not work.
    #collect train dataset from the dataset via a scoring function
    train_ds = sf.collect_train_data(train_df,info)

    #create the column for losses
    col = pd.DataFrame(columns=['i',str(info.current_epoch)])

    #training step
    for i, batch in enumerate(train_ds.batch(info.batch_size)):
        #collect losses and train model
        if i % 100 == 0: print("Batch ="+str(i)+"        ",end='\r')
        batch_loss = sf.train_step(batch[0],batch[1])
        #create a dataframe of the single column
        col = update_col(batch_loss,col,batch)

    #add the dataframe to the 
    df_train_losses = update_df(col,df_train_losses)

    #update tensorboard
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=info.current_epoch)
        tf.summary.scalar('accuracy',train_acc_metric.result(), step=info.current_epoch)
    
    #test steps
    for batch in test_ds.batch(info.batch_size):
        test_step(batch[0],batch[1])

    #update tensorboard
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step =info.current_epoch)
        tf.summary.scalar('accuracy', test_acc_metric.result(), step=info.current_epoch)

    #Printing to screen
    print('Epoch ',epoch+1,', Loss: ',train_loss.result().numpy(),', Accuracy: ',train_acc_metric.result().numpy(),', Test Loss: ',test_loss.result().numpy(),', Test Accuracy: ',test_acc_metric.result().numpy())
    
    #reset the metrics
    train_loss.reset_states()
    train_acc_metric.reset_states()
    test_loss.reset_states()
    test_acc_metric.reset_states()

    #save the model
    if epoch % 10 == 0:
        model.save(info.saved_model_path)
        print('Checkpoint saved')

#save the model and data
#np.savetxt('imagecounts/'+name,dataused)

model.save(saved_model_path)
df_train_losses.to_csv(info.data_path + info.dataset_name + '/normal_loss_info.csv')


