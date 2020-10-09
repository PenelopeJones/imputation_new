import numpy as np
import pdb

fname = 'Excape_x1_train_fingerprints'

n = 10000
splits = 77
pdb.set_trace()
for i in range(splits):
    x1 = np.load('{}{}.npy'.format(fname, i), allow_pickle=True)
    if i == 0:
        x = x1
    else:
        x = np.concatenate((x, x1), axis=0)

pdb.set_trace()
np.save('{}.npy'.format(fname), x)

