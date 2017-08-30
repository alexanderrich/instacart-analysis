import numpy as np
import pandas as pd
import gc
import sys
import xgboost as xgb
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

# load data set in chunks from hdf and filter desired rows for each dmatrix.
# Chunked loading is necessary to avoid memory issues (for 64GB RAM machine at
# least)
df = []
for chunk in pd.read_hdf("data/prior_all_stats.h5", "table", chunksize=10**6):
    filtered = chunk.loc[sets[chunk.subset_key,v],:].query('eval_set != "test" and validation_set != 10 and validation_set != @v')
    filtered = filtered.drop(["eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain", "subset_key"], axis=1)
    df.append(filtered)
    print('loaded_chunk')
    sys.stdout.flush()
for chunk in pd.read_hdf("data/prior_all_stats_extratrain.h5", "table", chunksize=10**6):
    filtered = chunk.loc[sets[chunk.subset_key,v],:].drop(["eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain", "subset_key"], axis=1)
    df.append(filtered)
    print('loaded_chunk')
    sys.stdout.flush()
df = pd.concat(df, ignore_index=True)
gc.collect()
labels = df.reordered.values
df = df.drop('reordered', axis=1)
d_train = xgb.DMatrix(df.values, feature_names=df.columns.values.tolist(), label=labels)
print("made d_train")
sys.stdout.flush()

df = []
out1 = []
for chunk in pd.read_hdf("data/prior_all_stats.h5", "table", chunksize=10**6):
    filtered = chunk.query('eval_set == "test"')
    filtered_out = filtered.loc[:, ['eval_set', 'validation_set', 'order_id', 'product_id', 'user_id', 'reordered']]
    filtered = filtered.drop(["reordered", "eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain", "subset_key"], axis=1)
    df.append(filtered)
    out1.append(filtered_out)
df = pd.concat(df, ignore_index=True)
out1 = pd.concat(out1, ignore_index=True)
d_test = xgb.DMatrix(df.values, feature_names=df.columns.values.tolist())
print("made d_test")
sys.stdout.flush()

df = []
for chunk in pd.read_hdf("data/prior_all_stats.h5", "table", chunksize=10**6):
    filtered = chunk.query('validation_set == @v')
    filtered = filtered.drop(["eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain", "subset_key"], axis=1)
    df.append(filtered)
df = pd.concat(df, ignore_index=True)
labels = df.reordered.values
df = df.drop('reordered', axis=1)
d_valid = xgb.DMatrix(df.values, feature_names=df.columns.values.tolist(), label=labels)
print("made d_valid")
sys.stdout.flush()


df = []
out2 = []
for chunk in pd.read_hdf("data/prior_all_stats.h5", "table", chunksize=10**6):
    filtered = chunk.query('validation_set == 10')
    filtered_out = filtered.loc[:, ['eval_set', 'validation_set', 'order_id', 'product_id', 'user_id', 'reordered']]
    filtered = filtered.drop(["eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain", "subset_key"], axis=1)
    df.append(filtered)
    out2.append(filtered_out)
df = pd.concat(df, ignore_index=True)
out2 = pd.concat(out2, ignore_index=True)
labels = df.reordered.values
df = df.drop('reordered', axis=1)
d_valid_2 = xgb.DMatrix(df.values, feature_names=df.columns.values.tolist(), label=labels)
print("made d_valid_2")
sys.stdout.flush()


# create df to hold eventual predictions
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

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'auc'
params['eta'] = 0.05
params['min_child_weight'] = 20
params['max_depth'] = 8
params['nthread'] = 16
params['subsample'] = 1
params['colsample_bytree'] = .5

watchlist = [(d_train, 'train'), (d_valid, 'valid')]


print("fold:", v)
sys.stdout.flush()
bst = xgb.train(params, d_train, 1500, watchlist, early_stopping_rounds=50, verbose_eval=10)

bst.save_model('xgb' + str(v) + '.model')

raw_output.loc[raw_output.eval_set == "test", 'prediction'] = bst.predict(d_test)
raw_output.loc[raw_output.validation_set == 10, 'prediction'] = bst.predict(d_valid_2)

raw_output.to_csv("rawpredictions/xgb" + str(v) + ".csv", index=False)
