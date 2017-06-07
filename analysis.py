
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#% matplotlib inline

prior_all_stats = pd.read_hdf("data/prior_all_stats.h5", "table")


# In[32]:

# split into train, validation, and test sets
#prior_all_stats['validation_set'] = 0
#prior_all_stats['prediction'] = 0
#valid_users = prior_all_stats.loc[prior_all_stats.eval_set == "train", "user_id"].unique()
#valid_users = pd.Series(valid_users).sample(frac=.1, random_state=1234)


# In[36]:

prior_all_stats['prediction'] = 0
all_users = prior_all_stats.loc[prior_all_stats.eval_set == "train", "user_id"].unique()
np.random.seed(1234)
np.random.shuffle(all_users)


# In[53]:

valid_set = pd.DataFrame({'user_id': all_users, 'validation_set': np.arange(0, all_users.shape[0]) % 10})


# In[56]:

prior_all_stats = prior_all_stats.merge(valid_set, on='user_id', how='left')


# In[58]:

prior_all_stats.validation_set = prior_all_stats.validation_set.fillna(-1)

# all_predictions = []
# all_none = []

import xgboost as xgb
prior_test = prior_all_stats.loc[prior_all_stats.eval_set == "test"]
d_test = xgb.DMatrix(prior_test.drop(["prediction", "eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).as_matrix())


for v in range(3,10):
    prior_train = prior_all_stats.loc[(prior_all_stats.eval_set == "train") & (prior_all_stats.validation_set != v)]
    prior_valid = prior_all_stats.loc[prior_all_stats.validation_set == v]

    d_train = xgb.DMatrix(prior_train.drop(["prediction", "eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).as_matrix(), label=prior_train.reordered.as_matrix())
    d_valid = xgb.DMatrix(prior_valid.drop(["prediction", "eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).as_matrix(), label=prior_valid.reordered.as_matrix())

    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'
    params['eta'] = 0.1
    params['max_depth'] = 6
    params['nthread'] = 12

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # bst = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, verbose_eval=10)

    # bst.save_model('multi_xgb' + str(v) + '.model')

    bst = xgb.Booster()
    bst.load_model('multi_xgb' + str(v) +  '.model')


    y_predicted = bst.predict(d_valid)

    prior_valid = prior_valid.copy()

    threshold_df = prior_valid.loc[:, ['user_id', 'user_distinct_products', 'prediction', 'reordered']].copy()


    guess = [-.8, -.1, .4, -1]
    width = np.array([.2, .1, .2, .1])
    best_reorder_cutoff = (0, 0)
    best_none_cutoff = (0, 0)
    best_cutoff_f1 = 0
    for i in range(4):
        for reorder_cutoff in [(x,y) for x in np.arange(guess[0]-4*width[0], guess[0]+4*width[0], width[0]) for y in np.arange(guess[1]-4*width[1], guess[1]+4*width[1], width[1])]:
            threshold_df['reorder_cutoff'] = 1.0 / (1 + np.exp(-reorder_cutoff[0] - reorder_cutoff[1] * np.log(threshold_df.user_distinct_products)))
            threshold_df.loc[:,'prediction'] = 1 * (y_predicted > threshold_df.reorder_cutoff)
            threshold_df['p_not'] = 1 - y_predicted
            threshold_df['hit'] = (threshold_df.reordered * threshold_df.prediction)
            threshold_df_agg = threshold_df.groupby("user_id").agg({'reordered': np.sum,
                                                              'prediction': np.sum,
                                                              'hit': np.sum,
                                                                  'user_distinct_products': np.mean,
                                                                 'p_not': np.prod})
            for none_cutoff in [(x,y) for x in np.arange(guess[2]-4*width[2], guess[2]+4*width[2], width[2]) for y in np.arange(guess[3]-4*width[3], guess[3]+4*width[3], width[3])]:
                threshold_df_agg['none_cutoff'] = 1.0 / (1 + np.exp(-none_cutoff[0] - none_cutoff[1] * np.log(threshold_df_agg.user_distinct_products)))
                threshold_df_agg['putnone'] = (threshold_df_agg.p_not > threshold_df_agg.none_cutoff) | (threshold_df_agg.prediction == 0)
                threshold_df_agg['truenone'] = (threshold_df_agg.reordered == 0)
                threshold_df_agg['r'] = threshold_df_agg.reordered
                threshold_df_agg['p'] = threshold_df_agg.prediction
                threshold_df_agg['h'] = threshold_df_agg.hit
                threshold_df_agg.loc[threshold_df_agg.putnone & threshold_df_agg.truenone, "h"] = 1
                threshold_df_agg.loc[threshold_df_agg.putnone, 'p'] = threshold_df_agg.loc[threshold_df_agg.putnone, 'p'] + 1
                threshold_df_agg.loc[threshold_df_agg.truenone, 'r'] = threshold_df_agg.loc[threshold_df_agg.truenone, 'r'] + 1
                threshold_df_agg['precision'] = (threshold_df_agg['h']) / (threshold_df_agg['p'])
                threshold_df_agg['recall'] = (threshold_df_agg['h']) / (threshold_df_agg['r'])
                threshold_df_agg['f1'] = 2 * threshold_df_agg['precision'] * threshold_df_agg['recall'] / (threshold_df_agg['precision'] + threshold_df_agg['recall'] + .000001)
                if threshold_df_agg['f1'].mean() > best_cutoff_f1:
                    best_cutoff_f1 = threshold_df_agg['f1'].mean()
                    best_reorder_cutoff = reorder_cutoff
                    best_none_cutoff = none_cutoff
        guess = [best_reorder_cutoff[0], best_reorder_cutoff[1], best_none_cutoff[0], best_none_cutoff[1]]
        width = width / 4
    print("best reorder cutoff:", best_reorder_cutoff)
    print("best none cutoff:", best_none_cutoff)
    print("best f1:", best_cutoff_f1)

    y_test = bst.predict(d_test)
    # all_reorder_cutoffs.append(best_reorder_cutoff)
    # all_none_cutoffs.append(best_none_cutoff)
    prior_test = prior_test.copy()
    prior_test['prediction'] = 1 * (y_test > 1.0 / (1 + np.exp(-best_reorder_cutoff[0] - best_reorder_cutoff[1] * np.log(prior_test.user_distinct_products))))
    prior_test['p_not'] = 1 - y_test

    writenone_df = prior_test.groupby('order_id').agg({'p_not': np.prod,
                                                       'prediction': np.sum,
                                                       'user_distinct_products': np.mean}).reset_index()
    writenone_df['putnone'] = ((writenone_df.p_not > 1.0 / (1 + np.exp(-best_none_cutoff[0] - best_none_cutoff[1] * np.log(writenone_df.user_distinct_products)))) | (writenone_df.prediction == 0))
    writenone_df['nonestring'] = ''
    writenone_df.loc[writenone_df.putnone, 'nonestring'] = 'None'

    prediction_df = prior_test[prior_test['prediction'] == 1].copy()
    prediction_df = prediction_df[['order_id', 'product_id']]


    prediction_lists = prediction_df.groupby('order_id').agg(lambda x: " ".join(x.astype(str))).reset_index()
    prediction_lists = prediction_lists.merge(writenone_df[['order_id', 'nonestring']], on='order_id', how='right')
    prediction_lists['products'] = prediction_lists.product_id.fillna('')
    prediction_lists['products'] = prediction_lists.products + " " + prediction_lists.nonestring

    prediction_lists = prediction_lists[['order_id', 'products']]
    prediction_lists.to_csv("submissions/multi_xgb" + str(v) + ".csv", index=False)
