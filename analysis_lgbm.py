import numpy as np
import pandas as pd
import gc
import sys
import lightgbm as lgb
import itertools

v = int(sys.argv[1])

# function for deciding which folds to include a data point in based on subset_key
sets = [x for x in itertools.combinations(np.arange(0, 10), 5)]
def binarize_set(set):
    x = np.zeros((10), dtype=np.bool)
    for i in set:
        x[i] = True
    return x
sets = np.array(list(map(binarize_set, sets)))

np.random.seed(seed=1234)

# loading data in chunks to avoid memory issues
df = []
for chunk in pd.read_hdf("data/prior_all_stats.h5", "table", chunksize=10**6):
    filtered = chunk.loc[sets[chunk.subset_key,v],:].query('eval_set != "test" and validation_set != 10 and validation_set != @v')
    # filtered = chunk.query('eval_set != "test" and validation_set != 10 and validation_set != @v')
    filtered = filtered.drop(["eval_set", "validation_set", "order_id", "user_id", "extratrain", "subset_key"], axis=1)
    # filtered = filterextra(filtered)
    df.append(filtered)
    print('loaded_chunk')
    sys.stdout.flush()
for chunk in pd.read_hdf("data/prior_all_stats_extratrain.h5", "table", chunksize=10**6):
    filtered = chunk.loc[sets[chunk.subset_key,v],:].drop(["eval_set", "validation_set", "order_id", "user_id", "extratrain", "subset_key"], axis=1)
    df.append(filtered)
    print('loaded_chunk')
    sys.stdout.flush()
df = pd.concat(df, ignore_index=True)
gc.collect()
labels = df.reordered.values
df = df.drop('reordered', axis=1)
d_train = lgb.Dataset(df.values, label=labels, feature_name=df.columns.tolist(), categorical_feature=['product_id', 'aisle_id', 'department_id'])
print("made d_train")
sys.stdout.flush()

df = []
out1 = []
for chunk in pd.read_hdf("data/prior_all_stats.h5", "table", chunksize=10**6):
    filtered = chunk.query('eval_set == "test"')
    filtered_out = filtered.loc[:, ['eval_set', 'validation_set', 'order_id', 'product_id', 'user_id', 'reordered']]
    filtered = filtered.drop(["reordered", "eval_set", "validation_set", "order_id", "user_id", "extratrain", "subset_key"], axis=1)
    df.append(filtered)
    out1.append(filtered_out)
df = pd.concat(df, ignore_index=True)
out1 = pd.concat(out1, ignore_index=True)
df_test = df
print("made d_test")
sys.stdout.flush()

df = []
for chunk in pd.read_hdf("data/prior_all_stats.h5", "table", chunksize=10**6):
    filtered = chunk.query('validation_set == @v')
    filtered = filtered.drop(["eval_set", "validation_set", "order_id", "user_id", "extratrain", "subset_key"], axis=1)
    df.append(filtered)
df = pd.concat(df, ignore_index=True)
labels = df.reordered.values
df = df.drop('reordered', axis=1)
d_valid = lgb.Dataset(df.values, label=labels, feature_name=df.columns.tolist(), categorical_feature=['product_id', 'aisle_id', 'department_id'])
print("made d_valid")
sys.stdout.flush()

df = []
out2 = []
for chunk in pd.read_hdf("data/prior_all_stats.h5", "table", chunksize=10**6):
    filtered = chunk.query('validation_set == 10')
    filtered_out = filtered.loc[:, ['eval_set', 'validation_set', 'order_id', 'product_id', 'user_id', 'reordered']]
    filtered = filtered.drop(["eval_set", "validation_set", "order_id", "user_id", "extratrain", "subset_key"], axis=1)
    df.append(filtered)
    out2.append(filtered_out)
df = pd.concat(df, ignore_index=True)
out2 = pd.concat(out2, ignore_index=True)
labels = df.reordered.values
df_valid_2 = df.drop('reordered', axis=1)
print("made d_valid_2")
sys.stdout.flush()

raw_output = pd.concat([out1, out2], ignore_index=True)
raw_output['prediction'] = 0
print("made raw_output")
sys.stdout.flush()
del out1
del out2
del filtered
del filtered_out
del df
gc.collect()

# Set our parameters for lgbm
params = {}
params['objective'] = 'binary'
params['metric'] = 'auc'
params['learning_rate'] = 0.02
params['min_data_in_leaf'] = 50
params['num_leaves'] = 200
params['nthread'] = 16
params['feature_fraction'] = .5

print("fold:", v)
bst = lgb.train(params, d_train, 2500, valid_sets=[d_train, d_valid], valid_names=['train', 'valid'], early_stopping_rounds=50, verbose_eval=10)

bst.save_model('lgb' + str(v) + '.model')

raw_output.loc[raw_output.eval_set == "test", 'prediction'] = bst.predict(df_test.values)
raw_output.loc[raw_output.validation_set == 10, 'prediction'] = bst.predict(df_valid_2.values)

raw_output.to_csv("rawpredictions/lgb" + str(v) + ".csv", index=False)
