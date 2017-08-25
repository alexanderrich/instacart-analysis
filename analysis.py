
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import gc
import sys
import xgboost as xgb

import itertools

v = int(sys.argv[1])

sets = [x for x in itertools.combinations(np.arange(0, 10), 5)]
def binarize_set(set):
    x = np.zeros((10), dtype=np.bool)
    for i in set:
        x[i] = True
    return x
sets = np.array(list(map(binarize_set, sets)))

np.random.seed(seed=1234)

# for v in range(10):

# def filterextra(filtered):
#     filtered = filtered.drop(["dim"+str(x) for x in range(10, 30)], axis=1)
#     filtered = filtered.drop(["user_dim"+str(x) for x in range(10,15)], axis=1)
#     filtered = filtered.drop(["indprod_hod_sin", "indprod_hod_cos", 'indprod_dow_sin', 'indprod_dow_cos',
#                               "product_hod_sin", "product_hod_cos", 'product_dow_sin', 'product_dow_cos',
#                               "user_hod_sin", "user_hod_cos", 'user_dow_sin', 'user_dow_cos',
#                               "order_hod_sin", "order_hod_cos", 'order_dow_sin', 'order_dow_cos'], axis=1)
#     filtered = filtered.drop([x for x in filtered.columns.values if "_reg" in x], axis=1)
#     return filtered


df = []
for chunk in pd.read_hdf("data/prior_all_stats.h5", "table", chunksize=10**6):
    filtered = chunk.loc[sets[chunk.subset_key,v],:].query('eval_set != "test" and validation_set != 10 and validation_set != @v')
    # filtered = chunk.query('eval_set != "test" and validation_set != 10 and validation_set != @v')
    filtered = filtered.drop(["eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain", "subset_key"], axis=1)
    # filtered = filterextra(filtered)
    df.append(filtered)
    print('loaded_chunk')
    sys.stdout.flush()
for chunk in pd.read_hdf("data/prior_all_stats_extratrain.h5", "table", chunksize=10**6):
    filtered = chunk.loc[sets[chunk.subset_key,v],:].drop(["eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain", "subset_key"], axis=1)
    # filtered = chunk.drop(["eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain", "subset_key"], axis=1)
    # filtered = filterextra(filtered)
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
    # filtered = filterextra(filtered)
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
    # filtered = filterextra(filtered)
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
    # filtered = filterextra(filtered)
    df.append(filtered)
    out2.append(filtered_out)
df = pd.concat(df, ignore_index=True)
out2 = pd.concat(out2, ignore_index=True)
labels = df.reordered.values
df = df.drop('reordered', axis=1)
d_valid_2 = xgb.DMatrix(df.values, feature_names=df.columns.values.tolist(), label=labels)
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

###############################

# prior_all_stats_extratrain = pd.read_hdf("data/prior_all_stats_extratrain.h5", "table")
# r = extratrain_rands[:prior_all_stats_extratrain.shape[0]]
# prior_all_stats_extratrain = prior_all_stats_extratrain.loc[sets[r,v],:]
# print('loaded extratrain')

# prior_all_stats_train = pd.read_hdf("data/prior_all_stats.h5", "table").query('eval_set != "test"')
# r = train_rands[:prior_all_stats_train.shape[0]]
# prior_all_stats_train = prior_all_stats_train.loc[sets[r,v],:]
# prior_all_stats_train = prior_all_stats_train.query('and validation_set == 10 and validation_set == @v')

# df = pd.concat([prior_all_stats_extratrain, prior_all_stats_train])
# del prior_all_stats_extratrain
# del prior_all_stats_train
# print(gc.collect())
# sys.stdout.flush()
# labels = df.reordered.values
# df = df.drop(["reordered", "eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain"], axis=1)
# d_train = xgb.DMatrix(df.values, label=labels)
# print("made train")
# sys.stdout.flush()

# df = pd.read_hdf("data/prior_all_stats.h5", "table").query('eval_set == "test"')
# out1 = df.loc[:, ['eval_set', 'validation_set', 'order_id', 'product_id', 'user_id', 'reordered']]
# df = df.drop(["reordered", "eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain"], axis=1)
# d_test = xgb.DMatrix(df.values)
# print("made test")
# sys.stdout.flush()

# df = pd.read_hdf("data/prior_all_stats.h5", "table").query('validation_set == @v')
# labels = df.reordered.values
# df = df.drop(["reordered", "eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain"], axis=1)
# d_valid = xgb.DMatrix(df.values)
# print("made valid")
# sys.stdout.flush()


# df = pd.read_hdf("data/prior_all_stats.h5", "table").query('validation_set == 10')
# out2 = df.loc[:, ['eval_set', 'validation_set', 'order_id', 'product_id', 'user_id', 'reordered']]
# labels = df.reordered.values
# df = df.drop(["reordered", "eval_set", "validation_set", "order_id", "user_id", "product_id", "extratrain"], axis=1)
# d_valid_2 = xgb.DMatrix(df.values)
# print("made valid 2")
# sys.stdout.flush()

# raw_output = pd.concat([out1, out2], ignore_index=True)
# raw_output['prediction'] = 0
# print("made raw_output")
# sys.stdout.flush()
# del out1
# del out2
# del df
# del labels
# print(gc.collect())
# sys.stdout.flush()


# prior_all_stats_orig = pd.read_hdf("data/prior_all_stats.h5", "table")
# print('loaded orig')
# print(gc.collect())
# sys.stdout.flush()

# prior_all_stats_orig_test_valid = prior_all_stats_orig.query('eval_set == "test" or validation_set == 10 or validation_set == @v')
# print('queried test')
# print(gc.collect())
# sys.stdout.flush()
# prior_all_stats_orig_train = prior_all_stats_orig.query('eval_set != "test" and validation_set != 10 and validation_set != @v').sample(frac=.4)
# print('queried train')
# print(gc.collect())
# sys.stdout.flush()

# del prior_all_stats_orig
# print(gc.collect())
# sys.stdout.flush()

# prior_all_stats = pd.concat([prior_all_stats_extratrain,
#                              prior_all_stats_orig_test_valid,
#                              prior_all_stats_orig_train])
# print('concatenated')
# del prior_all_stats_extratrain
# del prior_all_stats_orig_test_valid
# del prior_all_stats_orig_train
# print(gc.collect())
# sys.stdout.flush()


# prior_test = prior_all_stats.loc[prior_all_stats.eval_set == "test", :]
# d_test = xgb.DMatrix(prior_test.drop(["eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).values,
#                             feature_names=prior_test.drop(["eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).columns.values.tolist())
# print('built test')
# del prior_test
# print(gc.collect())
# sys.stdout.flush()


# # for v in range(10):
# prior_train = prior_all_stats.loc[(prior_all_stats.eval_set != "test") & (prior_all_stats.validation_set != v) & (prior_all_stats.validation_set != 10), :]
# prior_valid = prior_all_stats.loc[prior_all_stats.validation_set == v, :]
# prior_valid_2 = prior_all_stats.loc[prior_all_stats.validation_set == 10, :]

# d_train = xgb.DMatrix(prior_train.drop(["eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).values,
#                       feature_names=prior_train.drop(["eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).columns.values.tolist(),
#                       label=prior_train.reordered.values)
# print('built train')
# del prior_train
# sys.stdout.flush()
# print(gc.collect())
# sys.stdout.flush()

# d_valid = xgb.DMatrix(prior_valid.drop(["eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).values,
#                       feature_names=prior_valid.drop(["eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).columns.values.tolist(),
#                       label=prior_valid.reordered.values)
# print('built valid')
# print(gc.collect())
# sys.stdout.flush()

# d_valid_2 = xgb.DMatrix(prior_valid_2.drop(["eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).values,
#                         feature_names=prior_valid_2.drop(["eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).columns.values.tolist(),
#                         label=prior_valid_2.reordered.values)
# print('built valid 2')
# print(gc.collect())
# sys.stdout.flush()

# raw_output = prior_all_stats.loc[:,['eval_set', 'validation_set', 'order_id', 'product_id', 'user_id', 'user_distinct_products', 'reordered']]
# raw_output['prediction'] = 0
# print('got valid and test')
# sys.stdout.flush()

# del prior_train
# del prior_test
# del prior_valid_2
# del prior_valid
# del prior_all_stats
# print(gc.collect())
# sys.stdout.flush()

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'auc'
params['eta'] = 0.05
# params['gamma'] = 1
params['min_child_weight'] = 20
params['max_depth'] = 8
params['nthread'] = 16
params['subsample'] = 1
params['colsample_bytree'] = .5

watchlist = [(d_train, 'train'), (d_valid, 'valid')]


print("fold:", v)
sys.stdout.flush()
bst = xgb.train(params, d_train, 1500, watchlist, early_stopping_rounds=50, verbose_eval=10)

bst.save_model('glove_testing_xgb' + str(v) + '.model')

# bst = xgb.Booster()
# bst.load_model('multi_xgb' + str(v) +  '.model')

raw_output.loc[raw_output.eval_set == "test", 'prediction'] = bst.predict(d_test)
# prior_all_stats.loc[(prior_all_stats.eval_set != "test") & (prior_all_stats.validation_set != v) & (prior_all_stats.validation_set != 10), 'prediction'] = bst.predict(d_train)
# prior_all_stats.loc[prior_all_stats.validation_set == v, 'prediction'] = bst.predict(d_valid)
raw_output.loc[raw_output.validation_set == 10, 'prediction'] = bst.predict(d_valid_2)

raw_output.to_csv("rawpredictions/glove_testing" + str(v) + ".csv", index=False)
