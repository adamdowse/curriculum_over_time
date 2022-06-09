import pandas as pd
import numpy as np
#import supporting_functions as sf
import sqlite3
from sqlite3 import Error
import random

database =r"/com.docker.devenvironments.code/project/large_models/DBs/testdb.db"
conn = sqlite3.connect(database,detect_types=sqlite3.PARSE_DECLTYPES)

def scoring_function_random(a):
    return random.randint(0,100)

def scoring_function_name_len(name):
    return len(name)

conn.create_function("scoring_function_random", 1, scoring_function_random)
conn.create_function("scoring_function_name_len", 1, scoring_function_name_len)

cur = conn.cursor()
cur.execute(''' SELECT id, name FROM tasks ''')

info = np.array(cur.fetchall())

for i,name in zip(info[:,0],info[:,1]):
    print(i,name)
    cur.execute('''UPDATE tasks SET priority = scoring_function_name_len(?) WHERE id = (?)''',(name,i,))


conn.commit()