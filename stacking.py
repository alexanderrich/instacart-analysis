import numpy as np
import pandas as pd
import numba
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# function to read predictions from each model fold and compose into a
# dataframe
def read_models(model_names, namestring):
    nmodels = len(model_names)
    for i in range(nmodels):
        if i == 0:
            df = pd.read_csv(model_names[i])
            df['prediction_0'] = np.log(df.prediction / (1 - df.prediction))
            df.drop('prediction', axis=1, inplace=True)
        else:
            df_temp = pd.read_csv(model_names[i])
            df_temp.prediction = np.log(df_temp.prediction / (1 - df_temp.prediction))
            df['prediction_' + str(i)] = df_temp.prediction
        print('loaded', i)
    df['predict_'+namestring] = df.loc[:, 'prediction_0':'prediction_'+str(nmodels-1)].mean(axis=1)
    return df


df_xgb = read_models([
    'rawpredictions/xgb0.csv',
    'rawpredictions/xgb1.csv',
    'rawpredictions/xgb2.csv',
    'rawpredictions/xgb3.csv',
    'rawpredictions/xgb4.csv',
    'rawpredictions/xgb5.csv',
    'rawpredictions/xgb6.csv',
    'rawpredictions/xgb7.csv',
    'rawpredictions/xgb8.csv',
    'rawpredictions/xgb9.csv'
], 'xgb')

df_lgbm = read_models([
    'rawpredictions/lgb0.csv',
    'rawpredictions/lgb1.csv',
    'rawpredictions/lgb2.csv',
    'rawpredictions/lgb3.csv',
    'rawpredictions/lgb4.csv',
    'rawpredictions/lgb5.csv',
    'rawpredictions/lgb6.csv',
    'rawpredictions/lgb7.csv',
    'rawpredictions/lgb8.csv',
    'rawpredictions/lgb9.csv'
], 'lgbm')

df_sh1ng = read_models([
    'rawpredictions/sh1ng0.csv',
    'rawpredictions/sh1ng1.csv',
    'rawpredictions/sh1ng2.csv',
    'rawpredictions/sh1ng3.csv',
    'rawpredictions/sh1ng4.csv',
    'rawpredictions/sh1ng5.csv',
    'rawpredictions/sh1ng6.csv',
    'rawpredictions/sh1ng7.csv',
    'rawpredictions/sh1ng8.csv',
    'rawpredictions/sh1ng9.csv'
], 'sh1ng')

# merge models into one with average prediction from each model type
df = (df_xgb.drop(['prediction_'+str(s) for s in range(9)], axis=1)
      .merge(df_lgbm[['user_id', 'product_id', 'predict_lgbm']], on=['user_id', 'product_id'])
      .merge(df_sh1ng[['user_id', 'product_id', 'predict_sh1ng']], on=['user_id', 'product_id']))

model = LogisticRegression()
# train model on held-out 11th validation fold to find balance of 3 models
X_train = df.query('eval_set=="train"')[['predict_xgb', 'predict_lgbm',
                                         'predict_sh1ng']].values
y_train = df.query('eval_set=="train"')['reordered']
X_test = df.query('eval_set=="test"')[['predict_xgb', 'predict_lgbm', 'predict_sh1ng']].values
standard = StandardScaler()
X_train = standard.fit_transform(X_train)
X_test = standard.transform(X_test)
model.fit(X_train, y_train)
df['prediction'] = 0
df.loc[df.eval_set=="test", 'prediction'] = model.decision_function(X_test)
df.loc[df.eval_set=="train", 'prediction'] = model.decision_function(X_train)

# look at AUC performance of model on held out 11th validation fold (this will
# be overfit slightly, but probably not much since there are only 3 parameters)
# from sklearn.metrics import roc_auc_score
# roc_auc_score(y_train, model.decision_function(X_train))

# convert back to probabilities
df['prob'] = 1.0 / (1 + np.exp(-df.prediction))

# load predictions for whether each person purchased "none"
none_df = pd.read_csv('rawpredictions/nones.csv')
none_df['none_prob'] = none_df.prediction
df = df.merge(none_df[['order_id', 'none_prob']], on='order_id')


@numba.jit
def GFM_numba(probs, noneprob, nsamples=10000, nruns=1):
    """General F-measure Maximizer
    Implementation of algorithm described in:
    https://papers.nips.cc/paper/4389-an-exact-algorithm-for-f-measure-maximization.pdf

    The algorithm requires a joint distribution over the probability that each
    product is ordered, and the probability that no products are ordered (i.e.,
    that "None" is ordered). Because the passed None probability is calculated
    separately than the individual product probabilities, they might not
    produce a coherent joint distribution. To fix this problem, this function
    adds correlation among all the products until the probability of no
    products being ordered matches the provided "None" probability.

    The algorithm requires deriving from the joint distribution, for each
    product and each number of total products ordered x, the joint probability
    that the product will be ordered and that x total products will be ordered.
    The algorithm estimates those probabilities by sampling possible orders,
    using a user-defined number of samples.

    Args:
        probs (1d Numpy Array): an array of probabilities that each product is purchased.
        noneprob (float): probability that no products are purchased.
        nsamples (int): number of samples to use when estimating the probability distribution.
        nruns (int): number of times to run the sampling process and return optimal predictions.
            (nruns is helpful for determing the variance in predictions caused by sampling noise)

    Returns:
        2d numpy array: whether each product should be included in the order prediction,
            for each product and each run.
        1d numpy array: whether "none" should be included in the order prediction,
            for each run.
        float: expectation of F1 score, given accurate input probabilities

    """
    nprobs = probs.shape[0]
    extra_none = 0
    newprobs = probs.copy()
    # add correlation among products until probability that all are not ordered
    # matches noneprob
    while extra_none + (1-extra_none) * np.prod(1-newprobs) < noneprob:
        extra_none = extra_none + .0001
        newprobs = np.minimum(probs / (1 - extra_none), 1)
    probs = newprobs
    W = np.zeros((nprobs+1, nprobs+1))
    for i in range(nprobs+1):
        for j in range(nprobs+1):
            W[i, j] = 1.0 / (i + j + 2.0)

    expF1 = np.zeros((nruns))
    prediction = np.zeros((nruns, nprobs+1))
    for r in range(nruns):
        samples = (np.random.rand(nsamples, nprobs) < probs * np.ones((nsamples, nprobs))).astype(int)
        samples[:np.random.binomial(nsamples, extra_none),:] = 0
        samples = np.concatenate((samples, (samples.sum(1) == 0).astype(int).reshape((nsamples, 1))), axis=1)
        sums = samples.sum(1) - 1
        P = np.zeros((nprobs + 1, nprobs + 1))
        for i in range(nsamples):
            s = sums[i]
            P[s] = P[s] + samples[i]
        P = P.transpose()
        P = P / nsamples
        F = np.dot(P, W)
        m = P.shape[0]
        expectedF1s = np.zeros((m))
        for k in range(m):
            f = F[:,k]
            h = np.zeros((m))
            h[np.flip(np.argsort(f), axis=0)[:(k+1)]] = 1
            expectedF1s[k] = 2 * np.dot(h, f)
        k = np.argmax(expectedF1s)
        expF1[r] = expectedF1s[k]
        f = F[:,k]
        prediction[r, np.flip(np.argsort(f), axis=0)[:(k+1)]] = 1
    return prediction[:, :-1], prediction[:, -1], expF1


def GFM_wrapper(df, nsamples=10000, nruns=1):
    """wrapper for General F-measure Maximizer

    Args:
        df (pandas DataFrame): data frame holding predicted probabilities
        nsamples (int): number of samples to use when estimating the probability distribution.
        nruns (int): number of times to run the sampling process and return optimal predictions.
    returns:
        df with "prediction_i" and "putnone_i" columns indicating optimal market basket from each run of the GFM

    """

    probs = df['prob'].values
    noneprob = df['none_prob'].values[0]
    prediction, putnone, expF1 = GFM_numba(probs, noneprob, nsamples, nruns)
    if nruns == 1:
        df['prediction'] = prediction[0,:]
        df['putnone'] = putnone[0]
    else:
        for i in range(nruns):
            df['prediction_'+str(i)] = prediction[i,:]
            df['putnone_'+str(i)] = putnone[i]
            #df['expF1'] = expF1[0]
    return df


test_df = df.query('eval_set=="test"')[['order_id', 'product_id', 'prob', 'none_prob']].copy()
test_df = test_df.groupby('order_id').apply(GFM_wrapper, nsamples=50000, nruns=1)

writenone_df = test_df.groupby('order_id').agg({'putnone': np.mean}).reset_index()

writenone_df['nonestring'] = ''
writenone_df.loc[writenone_df.putnone == 1, 'nonestring'] = 'None'

# get predicted products for each order, and aggregate into a list of product id's
prediction_df = test_df.query('prediction == 1').copy()
prediction_df = prediction_df[['order_id', 'product_id']]
prediction_lists = prediction_df.groupby('order_id').agg(lambda x: " ".join(x.astype(str))).reset_index()
# add "None" to list for orders where None is predicted
prediction_lists = prediction_lists.merge(writenone_df[['order_id', 'nonestring']], on='order_id', how='right')
prediction_lists['products'] = prediction_lists.product_id.fillna('')
prediction_lists['products'] = prediction_lists.products + " " + prediction_lists.nonestring

# save submissions!
prediction_lists = prediction_lists[['order_id', 'products']]
prediction_lists.to_csv("submissions/submission.csv", index=False)
