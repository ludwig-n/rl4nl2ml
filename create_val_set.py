# Note: running this will not exactly reproduce the validation set used for this project.
# However, this is the same procedure that was used to create it, just with different random number generator states.

import collections
import pathlib
import pickle
import random


TRAIN_DATA_DIR = pathlib.Path('generator/data')
MIN_VAL_PORTION = 0.19
MAX_VAL_PORTION = 0.21
RANDOM_SEED = 27

random.seed(RANDOM_SEED)

with open(TRAIN_DATA_DIR / 'train_untrained_bert_tabular.p', 'rb') as fp:
    trainval_data = pickle.load(fp)
trainval_comp_counts = collections.Counter(sample['comp_name'] for sample in trainval_data).most_common()

val_comps = set()
cur_val_size = 0
min_val_size = int(MIN_VAL_PORTION * len(trainval_data))
max_val_size = int(MAX_VAL_PORTION * len(trainval_data))
while cur_val_size < min_val_size:
    comp, size = random.choice(trainval_comp_counts)
    while comp in val_comps or cur_val_size + size > max_val_size:
        comp, size = random.choice(trainval_comp_counts)
    val_comps.add(comp)
    cur_val_size += size

val_kernel_ids = {sample['kernel_id'] for sample in trainval_data if sample['comp_name'] in val_comps}
with open('val_kernel_ids_tabular.p', 'wb') as fp:
    pickle.dump(val_kernel_ids, fp)
