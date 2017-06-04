
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
np.random.shuffle(all_users)


# In[53]:

valid_set = pd.DataFrame({'user_id': all_users, 'validation_set': np.arange(0, all_users.shape[0]) % 10})


# In[56]:

prior_all_stats = prior_all_stats.merge(valid_set, on='user_id', how='left')


# In[58]:

prior_all_stats.validation_set = prior_all_stats.validation_set.fillna(-1)

all_predictions = []
all_none = []

import xgboost as xgb
prior_test = prior_all_stats.loc[prior_all_stats.eval_set == "test"]
d_test = xgb.DMatrix(prior_test.drop(["prediction", "eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"], axis=1).as_matrix())


for v in range(2):
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

    bst = xgb.train(params, d_train, 20, watchlist, early_stopping_rounds=50, verbose_eval=10)

    bst.save_model('multi_xgb' + str(v) + '.model')

    y_predicted = bst.predict(d_valid)

    prior_valid = prior_valid.copy()


    guess = [-1, -.1, -1, -.3]
    width = np.array([.2, .03, .2, .1])
    best_reorder_cutoff = (0, 0)
    best_none_cutoff = (0, 0)
    best_cutoff_f1 = 0
    for i in range(4):
        for reorder_cutoff in [(x,y) for x in np.arange(guess[0]-4*width[0], guess[0]+4*width[0], width[0]) for y in np.arange(guess[1]-4*width[1], guess[1]+4*width[1], width[1])]:
            prior_valid['reorder_cutoff'] = np.exp(reorder_cutoff[0] + reorder_cutoff[1] * np.log(prior_valid.user_distinct_products))
            prior_valid.loc[:,'prediction'] = 1 * (y_predicted > prior_valid.reorder_cutoff)
            prior_valid['p_not'] = 1 - y_predicted
            prior_valid['hit'] = (prior_valid.reordered * prior_valid.prediction)
            prior_valid_agg = prior_valid.groupby("user_id").agg({'reordered': np.sum, 
                                                              'prediction': np.sum, 
                                                              'hit': np.sum,
                                                                  'user_distinct_products': np.mean,
                                                                 'p_not': np.prod})
            for none_cutoff in [(x,y) for x in np.arange(guess[2]-4*width[2], guess[2]+4*width[2], width[2]) for y in np.arange(guess[3]-4*width[3], guess[3]+4*width[3], width[3])]:
                prior_valid_agg['none_cutoff'] = np.exp(none_cutoff[0] + none_cutoff[1] * np.log(prior_valid_agg.user_distinct_products))
                prior_valid_agg['putnone'] = (prior_valid_agg.p_not > prior_valid_agg.none_cutoff) | (prior_valid_agg.prediction == 0)
                prior_valid_agg['truenone'] = (prior_valid_agg.reordered == 0)
                prior_valid_agg['r'] = prior_valid_agg.reordered
                prior_valid_agg['p'] = prior_valid_agg.prediction
                prior_valid_agg['h'] = prior_valid_agg.hit
                prior_valid_agg.loc[prior_valid_agg.putnone & prior_valid_agg.truenone, "h"] = 1
                prior_valid_agg.loc[prior_valid_agg.putnone, 'p'] = prior_valid_agg.loc[prior_valid_agg.putnone, 'p'] + 1
                prior_valid_agg.loc[prior_valid_agg.truenone, 'r'] = prior_valid_agg.loc[prior_valid_agg.truenone, 'r'] + 1
                prior_valid_agg['precision'] = (prior_valid_agg['h']) / (prior_valid_agg['p'])
                prior_valid_agg['recall'] = (prior_valid_agg['h']) / (prior_valid_agg['r'])
                prior_valid_agg['f1'] = 2 * prior_valid_agg['precision'] * prior_valid_agg['recall'] / (prior_valid_agg['precision'] + prior_valid_agg['recall'] + .000001)
                if prior_valid_agg['f1'].mean() > best_cutoff_f1:
                    best_cutoff_f1 = prior_valid_agg['f1'].mean()
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
    prior_test['prediction'] = 1 * (y_test > np.exp(best_reorder_cutoff[0] + best_reorder_cutoff[1] * np.log(prior_test.user_distinct_products)))
    prior_test['p_not'] = 1 - y_test

    writenone_df = prior_test.groupby('order_id').agg({'p_not': np.prod, 
                                                       'prediction': np.sum, 
                                                       'user_distinct_products': np.mean}).reset_index()
    writenone_df['putnone'] = 1 * ((writenone_df.p_not > np.exp(best_none_cutoff[0] + best_none_cutoff[1] * np.log(writenone_df.user_distinct_products))) | (writenone_df.prediction == 0))


    all_predictions.append(prior_test.prediction.as_matrix())
    all_none.append(writenone_df.putnone.as_matrix())


# In[57]:

prediction = 1 * (np.array(all_predictions).mean(axis=0) >= .5)
is_none = 1 * (np.array(all_none).mean(axis=0) >= .5)

# prior_test = prior_test.copy()
# prior_test['prediction'] = 1 * (y_test > np.exp(best_reorder_cutoff[0] + best_reorder_cutoff[1] * np.log(prior_test.user_distinct_products)))
# prior_test['p_not'] = 1 - y_test


# In[58]:
prior_test['prediction'] = prediction

writenone_df = prior_test.groupby('order_id').agg({'p_not': np.prod, 
                                                   'prediction': np.sum, 
                                                   'user_distinct_products': np.mean}).reset_index()


# In[59]:

writenone_df['putnone'] = is_none
writenone_df['nonestring'] = ''
writenone_df.loc[writenone_df.putnone, 'nonestring'] = 'None'


# In[63]:

prediction_df = prior_test[prior_test['prediction'] == 1].copy()


# In[64]:

prediction_df = prediction_df[['order_id', 'product_id']]


# In[65]:

prediction_lists = prediction_df.groupby('order_id').agg(lambda x: " ".join(x.astype(str))).reset_index()


# In[66]:

prediction_lists = prediction_lists.merge(writenone_df[['order_id', 'nonestring']], on='order_id', how='right')


# In[67]:

prediction_lists['products'] = prediction_lists.product_id.fillna('')


# In[68]:

prediction_lists['products'] = prediction_lists.products + " " + prediction_lists.nonestring


# In[69]:

prediction_lists = prediction_lists[['order_id', 'products']]


# In[70]:

prediction_lists.to_csv("submissions/multi_xgb.csv", index=False)


# In[ ]:



