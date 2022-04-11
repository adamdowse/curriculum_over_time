import os

#File used to test running the experiments

epochs = str(30)
lr = str(0.01)
batch_size = str(32)
#'normal' 'naive_grads' 'pred_grads' 'pred_grads_gravity'
scoring_function = 'pred_grads'
#'none' 'naive_linear' 'naive_grad'
pacing_function = 'naive_linear'
data = 'mnist'
l_0 = str(0.1)
l_max = str(0.9)



command = [
    'python RunTest.py ',
    '--max_epochs '+epochs,
    ' --learning_rate '+lr,
    ' --batch_size '+batch_size,
    ' --scoring_function '+scoring_function,
    ' --pacing_function '+pacing_function,
    ' --dataset '+data,
    ' --lam_zero '+l_0,
    ' --lam_max '+l_max]

os.system(''.join(command))