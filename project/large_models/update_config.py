import wandb
import os

os.environ['WANDB_API_KEY'] = 'fc2ea89618ca0e1b85a71faee35950a78dd59744'
api = wandb.Api()

ids=[
    '2gvgg1wj',
    '1a2cjbl9',
    '213zj1d1',
    '2fisfle3',
    '110eqlpb',
    '2fn4zzoo',
    '2jzjfaq1',
    '31n4cfjf',
    '1ayrdepp',
    '2j44qu5t',
    '3brtilbk',
    '1ln7n7sv',
    '1hrg68eb',
    '2b8tyhnk',
    '2rr6zcx5']

g = [
    'll_ordered',
    'll_mixed',
    'll_ordered',
    'll_mixed',
    'grads_ordered',
    'grads_ordered',
    'grads_mixed',
    'grads_mixed',
    'normal',
    'normal',
    'normal',
    'll_mixed',
    'grads_mixed',
    'll_ordered',
    'grads_ordered'
]
for i,j in zip(ids,g):
    run = api.run("adamdowse/curriculum_over_time/"+i)
    run.config["group"] = j
    run.update()