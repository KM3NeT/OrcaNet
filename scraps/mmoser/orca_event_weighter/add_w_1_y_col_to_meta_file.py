import pandas as pd
import numpy as np
from oscillatedWeights_PIDoutput import *

fpath_meta = '/sps/km3net/users/mmoser/summarry_PID/04_18/pid_result_shiftedVertexEventSelection.meta'
fpath_columns = '/sps/km3net/users/mmoser/summarry_PID/04_18/pid_result_shiftedVertexEventSelection_column_names.txt'
outfilename = './pid_result_shiftedVertexEventSelection_w_osc_w_1_y.meta'

print('Reading the .meta file')
df = pd.read_csv(fpath_meta, sep=' ', header=None)
cols = np.loadtxt(fpath_columns, dtype='str')
df.columns = cols
print('Adding the oscillated weight column to the .meta file')
df.addOscillatedWeight()
print('Writing the new .meta file')
df.to_csv(outfilename, sep=' ', na_rep=-1234, header=False, index=None)
