import os

#File used to test running the experiments

epochs = str(10)
lr = str(0.01)
batch_size = str(32)
#'random' 'grads' pred_clusters pred_biggest_move pred_best_angle
scoring_function = 'random'
#'shuffle' 'hl' 'lh' 'mixed' 'naive_linear' 'naive_grad'
pacing_function = 'hl'
fill_function = 'ffill'
data = 'mnist'
data_amount = str(0.01)
dataset_similarity = str(0) 
l_0 = str(0.1)
l_max = str(0.9)
es = str(5)
lower_bound = str(0)
upper_bound = str(0)
#group = scoring_function+'_'+pacing_function+'_'+data_amount
group = 'testing'
record_loss = 'do'
batch_logs = 'True'
data_path = '/com.docker.devenvironments.code/project/large_models/datasets/'
db_path = "/com.docker.devenvironments.code/project/large_models/DBs/"
save_model_path = '/com.docker.devenvironments.code/project/large_models/saved_models/'


command = [
    'python RunTest.py',
    ' --max_epochs '+epochs,
    ' --learning_rate '+lr,
    ' --batch_size '+batch_size,
    ' --scoring_function '+scoring_function,
    ' --pacing_function '+pacing_function,
    ' --fill_function '+fill_function,
    ' --dataset '+data,
    ' --dataset_size '+data_amount,
    ' --dataset_similarity '+dataset_similarity,
    ' --data_path '+data_path,
    ' --db_path '+db_path,
    ' --save_model_path '+save_model_path,
    ' --early_stopping '+es,

    ' --group '+group,
    ' --record_loss '+record_loss,
    ' --batch_logs '+batch_logs,

    ' --lam_zero '+l_0,
    ' --lam_max '+l_max,
    
    ' --lam_lower_bound '+lower_bound,
    ' --lam_upper_bound '+upper_bound
    ]

os.system(''.join(command))