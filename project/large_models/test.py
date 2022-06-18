import pandas as pd
import numpy as np
#import supporting_functions as sf
import sqlite3
from sqlite3 import Error
import random
from dppy.finite_dpps import FiniteDPP

features = np.random.rand(8,3)

L = features.T.dot(features)
DPP = FiniteDPP('likelihood', **{'L': L})

DPP.sample_exact_k_dpp(size=3)
batch = DPP.list_of_samples  # list of trajectories, here there's only one
print(batch)