import numpy as np
import pandas as pd
import xgboost as xgb

none_df = pd.concat([pd.read_hdf("data/none_stats.h5", "table"),
                     pd.read_hdf("data/none_stats_extratrain.h5", "table")])

# create df to hold predictions from each fold
predictions_df = none_df[["eval_set", "validation_set", "order_id", "reordered", "user_id", "product_id"]].copy()

# create dmatrix for test data
prior_test = none_df.loc[none_df.eval_set == "test", :]
d_test = xgb.DMatrix(prior_test.drop(["prediction", "eval_set", "validation_set", "order_id",
                                      "reordered", "user_id", "product_id", "extratrain"], axis=1).as_matrix(),
                            feature_names=prior_test.drop(["prediction", "eval_set", "validation_set",
                                                           "order_id", "reordered", "user_id",
                                                           "product_id", "extratrain"], axis=1).columns.values.tolist(),
                     )

# run model 10 times on different data sets. Hold 1/11 of data out of all
# folds. For other 10/11, hold 1 part out of each fold and use to determine
# early stopping.
for v in range(10):
    predictions_df['prediction_'+str(v)] = 0
    # create dmatrix for train and two validation sets for test data
    prior_train = none_df.loc[(none_df.eval_set != "test") & (none_df.validation_set != v) & (none_df.validation_set != 10), :]
    prior_valid = none_df.loc[none_df.validation_set == v, :]
    prior_valid_2 = none_df.loc[none_df.validation_set == 10, :]
    d_train = xgb.DMatrix(prior_train.drop(["prediction", "eval_set", "validation_set",
                                            "order_id", "reordered", "user_id",
                                            "product_id", "extratrain"], axis=1).values,
                          feature_names=prior_train.drop(["prediction", "eval_set", "validation_set",
                                                          "order_id", "reordered", "user_id",
                                                          "product_id", "extratrain"], axis=1).columns.values.tolist(),
                          label=prior_train.reordered.as_matrix())
    d_valid = xgb.DMatrix(prior_valid.drop(["prediction", "eval_set", "validation_set",
                                            "order_id", "reordered", "user_id",
                                            "product_id", "extratrain"], axis=1).values,
                          feature_names=prior_valid.drop(["prediction", "eval_set", "validation_set",
                                                          "order_id", "reordered", "user_id",
                                                          "product_id", "extratrain"], axis=1).columns.values.tolist(),
                          label=prior_valid.reordered.as_matrix())
    d_valid_2 = xgb.DMatrix(prior_valid_2.drop(["prediction", "eval_set", "validation_set",
                                                "order_id", "reordered", "user_id",
                                                "product_id", "extratrain"], axis=1).values,
                            feature_names=prior_valid_2.drop(["prediction", "eval_set", "validation_set",
                                                              "order_id", "reordered", "user_id",
                                                              "product_id", "extratrain"], axis=1).columns.values.tolist(),
                            label=prior_valid_2.reordered.as_matrix())

    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'
    params['eta'] = 0.05
    params['max_depth'] = 4
    params['nthread'] = 12
    params['gamma'] = 0
    params['min_child_weight'] = 0
    params['colsample_bytree'] = .5

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    print("fold:", v)
    bst = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, verbose_eval=10)
    # add predictions to prediction_df
    predictions_df.loc[none_df.eval_set == "test", 'prediction_' + str(v)] = bst.predict(d_test, ntree_limit=bst.best_ntree_limit+50)
    predictions_df.loc[(none_df.eval_set != "test") & (none_df.validation_set != v) & (none_df.validation_set != 10),
                       'prediction_' + str(v)] = bst.predict(d_train, ntree_limit=bst.best_ntree_limit+50)
    predictions_df.loc[none_df.validation_set == v, 'prediction_' + str(v)] = bst.predict(d_valid, ntree_limit=bst.best_ntree_limit+50)
    predictions_df.loc[none_df.validation_set == 10, 'prediction_' + str(v)] = bst.predict(d_valid_2, ntree_limit=bst.best_ntree_limit+50)

# convert to logistic space before taking mean
for i in range(10):
    predictions_df['prediction_'+str(i)] = np.log(predictions_df['prediction_'+str(i)] / (1 - predictions_df['prediction_'+str(i)]))

# get roc_auc_performance on 11th validation fold
# from sklearn.metrics import roc_auc_score
# roc_auc_score(none_df.query('validation_set == 10').reordered, predictions_df.query('validation_set == 10').loc[:, 'prediction_0':'prediction_9'].mean(axis=1))

none_df['prediction'] = predictions_df.loc[:, 'prediction_0':'prediction_9'].mean(axis=1)
none_df['prediction'] = 1.0/(1.0+np.exp(-none_df['prediction']))

raw_output = none_df.loc[:,['prediction', 'eval_set', 'validation_set', 'order_id', 'product_id', 'user_id', 'user_distinct_products', 'reordered']]
raw_output.to_csv("rawpredictions/nones.csv", index=False)
