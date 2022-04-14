import os

#File used to test running the experiments

epochs = str(30)
lr = str(0.01)
batch_size = str(32)
#'normal' 'grads' 
scoring_function = 'grads'
#'none' 'naive_linear' 'naive_grad'
pacing_function = 'naive_grad'
fill_function = 'ffill'
data = 'mnist'
l_0 = str(0.1)
l_max = str(0.9)
es = str(5)
lower_bound = str(0)
upper_bound = str(0)

command = [
    'python RunTest.py ',
    '--max_epochs '+epochs,
    ' --learning_rate '+lr,
    ' --batch_size '+batch_size,
    ' --scoring_function '+scoring_function,
    ' --pacing_function '+pacing_function,
    ' --fill_function '+fill_function,
    ' --dataset '+data,
    ' --lam_zero '+l_0,
    ' --lam_max '+l_max,
    ' --lam_lower_bound '+lower_bound,
    ' --lam_upper_bound '+upper_bound,
    ' --early_stopping '+es]

os.system(''.join(command))