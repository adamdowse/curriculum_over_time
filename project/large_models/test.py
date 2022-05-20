import pandas as pd
import numpy as np
import supporting_functions as sf
import sqlite3
from sqlite3 import Error

database =r"/com.docker.devenvironments.code/project/large_models/DBs/mnist.db"
conn = sf.DB_create_connection(database)

cur = conn.cursor()

sql = '''   UPDATE imgs SET used = 0'''

cur.execute(sql)
conn.commit()