import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import spacy
nlp = spacy.load('en_vectors_glove_md')

import warnings
warnings.filterwarnings('ignore')


aisles_df = pd.read_csv("data/aisles.csv")
departments_df = pd.read_csv("data/departments.csv")
products_df = pd.read_csv("data/products.csv")
orders_df = pd.read_csv("data/orders.csv")
prior_df = pd.read_csv("data/order_products__prior.csv")
train_df = pd.read_csv("data/order_products__train.csv")
products_df_merged = (products_df
                      .merge(departments_df, on="department_id")
                      .merge(aisles_df, on="aisle_id"))

# if running with argument "extra", strip off true last orders (train/test
# orders) and train to predict last order in prior orders
is_extra = len(sys.argv) == 2 and sys.argv[1] == "extra"
if is_extra:
    orders_df = orders_df.query('eval_set == "prior"')
    orders_df['max_order'] = orders_df.groupby('user_id').order_number.transform(max)
    orders_df.loc[orders_df.order_number == orders_df.max_order, 'eval_set'] = 'extratrain'
    orders_df.drop('max_order', axis=1, inplace=True)
    train_df = prior_df.loc[prior_df.order_id.isin(orders_df.query('eval_set == "extratrain"').order_id),:]
    prior_df = prior_df.loc[prior_df.order_id.isin(orders_df.query('eval_set == "prior"').order_id),:]


# add "None" as another product for feature creation. Will separate out later.
none_train_df = train_df.groupby('order_id').agg({'product_id': lambda x: "None",
                                                  'add_to_cart_order': lambda x: 0,
                                                  'reordered': np.sum}).reset_index()
none_train_df.reordered = (none_train_df.reordered == 0).astype(int)
none_train_df = none_train_df.query('reordered == 1')
none_prior_df = prior_df.groupby('order_id').agg({'product_id': lambda x: "None",
                                             'add_to_cart_order': lambda x: 0,
                                             'reordered': np.sum}).reset_index()
none_prior_df.reordered = (none_prior_df.reordered == 0).astype(int)
none_prior_df = none_prior_df.query('reordered == 1')
train_df = pd.concat([train_df, none_train_df])
prior_df = pd.concat([prior_df, none_prior_df])


# create string for each product concatenating name, aisle name, department name
products_df_merged['allwords'] = products_df_merged.product_name.str.cat(
    [products_df_merged.department, products_df_merged.aisle], sep=' ')
# use this to make product name vectors using Glove word embedding, and reduce
# dimensionality using PCA
vectors = np.array(products_df_merged
                   .allwords.apply(lambda x: nlp(x).vector).tolist())
pca = PCA(n_components=30)
pca.fit(vectors)
short_vectors = pca.transform(vectors)
short_vectors_df = pd.DataFrame(short_vectors)
short_vectors_df.columns = ["dim"+str(c) for c in short_vectors_df.columns]
short_vectors_df['product_id'] = products_df_merged.product_id

# figure out absolute date of each order within a user's order history
orders_df['absolute_date'] = orders_df.groupby("user_id").days_since_prior_order.cumsum().fillna(0)
orders_df['max_order_number'] = orders_df.groupby("user_id").order_number.transform(max)
orders_df['max_absolute_date'] = orders_df.groupby("user_id").absolute_date.transform(max)
orders_df['reverse_date'] = orders_df.max_absolute_date - orders_df.absolute_date
orders_df['reverse_order_number'] = orders_df.max_order_number - orders_df.order_number

train_df = train_df.merge(orders_df[["order_id", "user_id"]], on="order_id")
prior_df = prior_df.merge(orders_df, on="order_id")

# convert day of week and hour of day into an angle on circle representing a
# week or day
prior_df['order_dow_angle'] = (prior_df.order_dow /
                                     (prior_df.order_dow.max() + 1) * 2 * np.pi)
prior_df['order_hod_angle'] = (prior_df.order_hour_of_day /
                                             (prior_df.order_dow.max() + 1) * 2 * np.pi)


# decompose order time/day into sin and cos
prior_df['order_dow_sin'] = np.sin(prior_df.order_dow_angle)
prior_df['order_dow_cos'] = np.cos(prior_df.order_dow_angle)
prior_df['order_hod_sin'] = np.sin(prior_df.order_hod_angle)
prior_df['order_hod_cos'] = np.cos(prior_df.order_hod_angle)

# column for later aggregation of product counts
prior_df['num_products'] = 1

# Weighting of orders by distance in the past. Weight based on both days since
# the order, and orders since the order. Weight with several decay rates.
prior_df['num_products_dw_8'] = np.exp(-np.log(2)/8 * prior_df.reverse_date)
prior_df['num_products_dw_16'] = np.exp(-np.log(2)/16 * prior_df.reverse_date)
prior_df['num_products_dw_32'] = np.exp(-np.log(2)/32 * prior_df.reverse_date)
prior_df['num_products_dw_64'] = np.exp(-np.log(2)/64 * prior_df.reverse_date)
prior_df['num_products_dw_128'] = np.exp(-np.log(2)/128 * prior_df.reverse_date)
prior_df['num_products_ow_2'] = np.exp(-np.log(2)/2 * prior_df.reverse_order_number)
prior_df['num_products_ow_4'] = np.exp(-np.log(2)/4 * prior_df.reverse_order_number)
prior_df['num_products_ow_8'] = np.exp(-np.log(2)/8 * prior_df.reverse_order_number)
prior_df['num_products_ow_16'] = np.exp(-np.log(2)/16 * prior_df.reverse_order_number)
prior_df['num_products_ow_32'] = np.exp(-np.log(2)/32 * prior_df.reverse_order_number)

# Weighting of orders by distance in past using a 14-day and 30-day periodic
# cycle. This might capture biweekly or monthly patterns not captured by order
# dow.
prior_df['num_products_dsin_14'] = (1.01 + np.sin(2*np.pi*(prior_df.reverse_date/14)))/2
prior_df['num_products_dcos_14'] = (1.01 + np.cos(2*np.pi*(prior_df.reverse_date/14)))/2
prior_df['num_products_dsin_30'] = (1.01 + np.sin(2*np.pi*(prior_df.reverse_date/30)))/2
prior_df['num_products_dcos_30'] = (1.01 + np.cos(2*np.pi*(prior_df.reverse_date/30)))/2


# Create dataframes holding the proportion of orders that were made on each
# {hour, day} for each {user, user*product, product}.

# Overall proportion of orders on each day.
prior_day_idx = prior_df[['user_id', 'order_id', 'product_id']].join(pd.get_dummies(prior_df.order_dow))
days_prior = prior_day_idx.drop(['product_id', 'user_id'], axis=1).groupby('order_id').agg(np.mean).agg(np.mean)

# Proportion of orders on each day for each product
product_day_idx = (prior_day_idx.drop(['user_id', 'order_id'], axis=1)
                   .groupby("product_id").agg(np.mean).reset_index()
                   .melt(id_vars='product_id', var_name="day", value_name="product_day_proportion"))
product_day_idx.day = product_day_idx.day.astype(int)

# Proportion of orders on each day for each user
individual_day_idx = (prior_day_idx.drop(['product_id'], axis=1)
                   .groupby(['user_id', 'order_id']).agg(np.mean).reset_index()
                     .drop(['order_id'], axis=1).groupby('user_id').agg(np.mean).reset_index()
                   .melt(id_vars='user_id', var_name="day", value_name="user_day_proportion"))
individual_day_idx.day = individual_day_idx.day.astype(int)

# Proportion of orders on each day for each user*product
indprod_day_idx = (prior_day_idx.drop(['order_id'], axis=1)
                   .groupby(['user_id', 'product_id']).agg(np.mean).reset_index()
                   .melt(id_vars=['user_id', 'product_id'], var_name="day", value_name="indprod_day_proportion"))
indprod_day_idx.day = indprod_day_idx.day.astype(int)

indprod_day_idx.rename(columns={'day': 'order_dow'}, inplace=True)
product_day_idx.rename(columns={'day': 'order_dow'}, inplace=True)
individual_day_idx.rename(columns={'day': 'order_dow'}, inplace=True)

# Overall proportion of orders on each hour of day.
prior_hod_idx = prior_df[['user_id', 'order_id', 'product_id']].join(pd.get_dummies(prior_df.order_hour_of_day))

prior_hod_idx_orig = prior_hod_idx
prior_hod_idx = prior_hod_idx.copy()

# Since there are lots of hours, count an order as occurring during an hour if
# it occurred the hour before or after
for i in range(24):
    prior_hod_idx[i] = prior_hod_idx_orig[(i - 1) % 24] + prior_hod_idx_orig[i] + prior_hod_idx_orig[(i + 1) % 24]
hod_prior = prior_hod_idx.drop(['product_id', 'user_id'], axis=1).groupby('order_id').agg(np.mean).agg(np.mean)

del prior_hod_idx_orig

# Proportion of orders on each hour for each product
product_hod_idx = (prior_hod_idx.drop(['user_id', 'order_id'], axis=1)
                   .groupby("product_id").agg(np.mean).reset_index()
                   .melt(id_vars='product_id', var_name="hod", value_name="product_hod_proportion"))
product_hod_idx.hod = product_hod_idx.hod.astype(int)

# Proportion of orders on each hour for each user
individual_hod_idx = (prior_hod_idx.drop(['product_id'], axis=1)
                   .groupby(['user_id', 'order_id']).agg(np.mean).reset_index()
                     .drop(['order_id'], axis=1).groupby('user_id').agg(np.mean).reset_index()
                   .melt(id_vars='user_id', var_name="hod", value_name="user_hod_proportion"))
individual_hod_idx.hod = individual_hod_idx.hod.astype(int)

# Proportion of orders on each hour for each user*product
indprod_hod_idx = (prior_hod_idx.drop(['order_id'], axis=1)
                   .groupby(['user_id', 'product_id']).agg(np.mean).reset_index()
                   .melt(id_vars=['user_id', 'product_id'], var_name="hod", value_name="indprod_hod_proportion"))
indprod_hod_idx.hod = indprod_hod_idx.hod.astype(int)

indprod_hod_idx.rename(columns={'hod': 'order_hour_of_day'}, inplace=True)
product_hod_idx.rename(columns={'hod': 'order_hour_of_day'}, inplace=True)
individual_hod_idx.rename(columns={'hod': 'order_hour_of_day'}, inplace=True)

# calculating statistics for each product
prior_product_stats = prior_df.groupby("product_id").agg({'order_dow_sin': np.sum,
                                                          'order_dow_cos': np.sum,
                                                          'order_hod_sin': np.sum,
                                                          'order_hod_cos': np.sum,
                                                          'num_products': np.sum})
# arctan2 is used to get the circlular average of the dow/hod angle, using the
# summed sines and cosines
prior_product_stats['order_dow_angle'] = np.arctan2(prior_product_stats.order_dow_sin,
                                                    prior_product_stats.order_dow_cos)
prior_product_stats['order_hod_angle'] = np.arctan2(prior_product_stats.order_hod_sin,
                                                    prior_product_stats.order_hod_cos)

prior_product_stats.order_dow_sin = np.sin(prior_product_stats.order_dow_angle)
prior_product_stats.order_dow_cos = np.cos(prior_product_stats.order_dow_angle)
prior_product_stats.order_hod_sin = np.sin(prior_product_stats.order_hod_angle)
prior_product_stats.order_hod_cos = np.cos(prior_product_stats.order_hod_angle)
prior_product_stats.drop(['order_dow_angle', 'order_hod_angle'], axis=1, inplace=True)
prior_product_stats.reset_index(inplace=True)
prior_product_stats.columns = ['product_id', 'product_dow_sin',
                               'product_dow_cos', 'product_hod_sin',
                               'product_hod_cos', 'product_num_orders']


# calculating stats for each individual*order, used for certain other features
prior_indorder_stats = (prior_df.groupby(["user_id", "order_id"])
                        .agg({'order_dow_sin': np.sum,
                              'order_dow_cos': np.sum,
                              'order_hod_sin': np.sum,
                              'order_hod_cos': np.sum,
                              'num_products': np.sum,
                              'num_products_dw_8': np.mean,
                              'num_products_dw_16': np.mean,
                              'num_products_dw_32': np.mean,
                              'num_products_dw_64': np.mean,
                              'num_products_dw_128': np.mean,
                              'num_products_dcos_14': np.mean,
                              'num_products_dsin_14': np.mean,
                              'num_products_dcos_30': np.mean,
                              'num_products_dsin_30': np.mean,
                              'num_products_ow_2': np.mean,
                              'num_products_ow_4': np.mean,
                              'num_products_ow_8': np.mean,
                              'num_products_ow_16': np.mean,
                              'num_products_ow_32': np.mean,
                              'absolute_date': np.max,
                              'order_number': np.max}).reset_index())

# calculating stats for each individual
prior_individual_stats = (prior_indorder_stats.groupby("user_id")
                          .agg({'order_dow_sin': np.sum,
                                'order_dow_cos': np.sum,
                                'order_hod_sin': np.sum,
                                'order_hod_cos': np.sum,
                                'num_products': [np.sum, np.mean],
                                'num_products_dw_8': np.sum,
                                'num_products_dw_16': np.sum,
                                'num_products_dw_32': np.sum,
                                'num_products_dw_64': np.sum,
                                'num_products_dw_128': np.sum,
                                'num_products_dcos_14': np.sum,
                                'num_products_dsin_14': np.sum,
                                'num_products_dcos_30': np.sum,
                                'num_products_dsin_30': np.sum,
                                'num_products_ow_2': np.sum,
                                'num_products_ow_4': np.sum,
                                'num_products_ow_8': np.sum,
                                'num_products_ow_16': np.sum,
                                'num_products_ow_32': np.sum,
                                'absolute_date': np.max,
                                'order_number': np.max}))


prior_individual_stats.columns = ['order_dow_sin', 'order_dow_cos',
                                  'order_hod_sin', 'order_hod_cos',
                                  'num_products', 'mean_products',
                                  'num_products_dw_8', 'num_products_dw_16',
                                  'num_products_dw_32', 'num_products_dw_64',
                                  'num_products_dw_128', 'num_products_dcos_14',
                                  'num_products_dsin_14', 'num_products_dcos_30',
                                  'num_products_dsin_30', 'num_products_ow_2',
                                  'num_products_ow_4', 'num_products_ow_8',
                                  'num_products_ow_16', 'num_products_ow_32',
                                  'max_absolute_date', "max_order_number"]
prior_individual_stats['order_dow_angle'] = np.arctan2(prior_individual_stats.order_dow_sin,
                                                       prior_individual_stats.order_dow_cos)
prior_individual_stats['order_hod_angle'] = np.arctan2(prior_individual_stats.order_hod_sin,
                                                       prior_individual_stats.order_hod_cos)
prior_individual_stats.order_dow_sin = np.sin(prior_individual_stats.order_dow_angle)
prior_individual_stats.order_dow_cos = np.cos(prior_individual_stats.order_dow_angle)
prior_individual_stats.order_hod_sin = np.sin(prior_individual_stats.order_hod_angle)
prior_individual_stats.order_hod_cos = np.cos(prior_individual_stats.order_hod_angle)
prior_individual_stats.drop(['order_dow_angle', 'order_hod_angle'], axis=1, inplace=True)
prior_individual_stats = prior_individual_stats.reset_index()
prior_individual_stats.columns = ['user_id', 'user_dow_sin',
                                  'user_dow_cos', 'user_hod_sin',
                                  'user_hod_cos', 'user_num_products',
                                  'user_mean_products', 'user_num_products_dw_8',
                                  'user_num_products_dw_16', 'user_num_products_dw_32',
                                  'user_num_products_dw_64', 'user_num_products_dw_128',
                                  'user_num_products_dcos_14', 'user_num_products_dsin_14',
                                  'user_num_products_dcos_30', 'user_num_products_dsin_30',
                                  'user_num_products_ow_2', 'user_num_products_ow_4',
                                  'user_num_products_ow_8', 'user_num_products_ow_16',
                                  'user_num_products_ow_32', 'user_num_days',
                                  'user_num_orders']
prior_individual_stats['user_days_per_order'] = (prior_individual_stats.user_num_days /
                                                 prior_individual_stats.user_num_orders)
# calculate days between orders through a more intensive method, allowing calculation of SD
order_date_diffs = (prior_indorder_stats[['user_id', 'absolute_date']]
                    .groupby('user_id').absolute_date
                    .apply(lambda x: x.sort_values().diff()[1:]))
user_days_per_order = (order_date_diffs.reset_index()
                       .groupby('user_id').absolute_date
                       .agg([np.mean, lambda x: np.std(x, ddof=1)]).reset_index())
user_days_per_order.columns = ['user_id', 'user_days_per_order_mean',
                                'user_days_per_order_std']
prior_individual_stats = prior_individual_stats.merge(user_days_per_order, on='user_id')


# preserving needed individual*order stats
prior_indorder_stats = prior_indorder_stats[['order_id', 'num_products']]
prior_indorder_stats.columns = ['order_id', 'num_products_in_order']

# calculating stats for each individual*product
prior_indprod_stats = (
    prior_df.merge(prior_indorder_stats[['order_id', 'num_products_in_order']],
                   on='order_id')
    .merge(prior_individual_stats[['user_id', 'user_num_orders', 'user_num_days']],
           on='user_id', how='left'))
prior_indprod_stats['add_to_cart_proportion'] = (prior_indprod_stats['add_to_cart_order'] /
                                                 prior_indprod_stats['num_products_in_order'])
prior_indprod_stats['indprod_inorder_1'] = 1 * (prior_indprod_stats.order_number ==
                                                prior_indprod_stats.user_num_orders)
prior_indprod_stats['indprod_inorder_2'] = 1 * (prior_indprod_stats.order_number ==
                                                prior_indprod_stats.user_num_orders - 1)
prior_indprod_stats = (prior_indprod_stats.groupby(["user_id", "product_id"])
                       .agg({'order_dow_sin': np.sum,
                             'order_dow_cos': np.sum,
                             'order_hod_sin': np.sum,
                             'order_hod_cos': np.sum,
                             'num_products': np.sum,
                             'num_products_dw_8': np.sum,
                             'num_products_dw_16': np.sum,
                             'num_products_dw_32': np.sum,
                             'num_products_dw_64': np.sum,
                             'num_products_dw_128': np.sum,
                             'num_products_dcos_14': np.sum,
                             'num_products_dsin_14': np.sum,
                             'num_products_dcos_30': np.sum,
                             'num_products_dsin_30': np.sum,
                             'num_products_ow_2': np.sum,
                             'num_products_ow_4': np.sum,
                             'num_products_ow_8': np.sum,
                             'num_products_ow_16': np.sum,
                             'num_products_ow_32': np.sum,
                             'add_to_cart_order': np.mean,
                             'add_to_cart_proportion': np.mean,
                             'indprod_inorder_1': np.sum,
                             'indprod_inorder_2': np.sum,
                             'user_num_orders': np.mean,
                             'user_num_days': np.mean,
                             'reverse_date': np.min,
                             'reverse_order_number': np.min}).reset_index())
prior_indprod_stats['order_dow_angle'] = np.arctan2(prior_indprod_stats.order_dow_sin,
                                                    prior_indprod_stats.order_dow_cos)
prior_indprod_stats['order_hod_angle'] = np.arctan2(prior_indprod_stats.order_hod_sin,
                                                    prior_indprod_stats.order_hod_cos)
prior_indprod_stats['proportion_orders'] = (prior_indprod_stats.num_products /
                                            prior_indprod_stats.user_num_orders)
prior_indprod_stats['days_per_order'] = (prior_indprod_stats.user_num_days /
                                         prior_indprod_stats.num_products)
prior_indprod_stats.order_dow_sin = np.sin(prior_indprod_stats.order_dow_angle)
prior_indprod_stats.order_dow_cos = np.cos(prior_indprod_stats.order_dow_angle)
prior_indprod_stats.order_hod_sin = np.sin(prior_indprod_stats.order_hod_angle)
prior_indprod_stats.order_hod_cos = np.cos(prior_indprod_stats.order_hod_angle)
prior_indprod_stats.drop(['order_dow_angle', 'order_hod_angle', 'user_num_orders', 'user_num_days'],
                         axis=1, inplace=True)

prior_indprod_stats.columns = ['user_id', 'product_id', 'indprod_dow_sin',
                               'indprod_dow_cos', 'indprod_hod_sin',
                               'indprod_hod_cos', 'indprod_num_orders',
                               'indprod_num_products_dw_8', 'indprod_num_products_dw_16',
                               'indprod_num_products_dw_32', 'indprod_num_products_dw_64',
                               'indprod_num_products_dw_128', 'indprod_num_products_dcos_14',
                               'indprod_num_products_dsin_14', 'indprod_num_products_dcos_30',
                               'indprod_num_products_dsin_30', 'indprod_num_products_ow_2',
                               'indprod_num_products_ow_4', 'indprod_num_products_ow_8',
                               'indprod_num_products_ow_16', 'indprod_num_products_ow_32',
                               'indprod_add_to_cart_order', 'indprod_add_to_cart_proportion',
                               'indprod_inorder_1', 'indprod_inorder_2',
                               'indprod_days_since_last', 'indprod_orders_since_last',
                                'indprod_proportion_orders', 'indprod_days_per_order']

# use indprod means to add more product stats
product_order_proportions = (prior_indprod_stats[['user_id', 'product_id',
                                                  'indprod_proportion_orders', 'indprod_days_per_order']]
                             .groupby("product_id")
                             .agg({'indprod_proportion_orders': np.mean,
                                   'indprod_days_per_order': np.mean})).reset_index()
product_order_proportions.columns = ['product_id', 'product_proportion_orders', 'product_days_per_order']
prior_product_stats = prior_product_stats.merge(product_order_proportions, on='product_id')

# merge all features into one big data frame
prior_all_stats = (prior_indprod_stats
                   .merge(prior_individual_stats, on="user_id", how="left")
                   .merge(prior_product_stats, on='product_id', how="left"))


# calculate versions of weighted predictors regularized by individuals total
# number/timing of orders
for label in ['ow_2', 'ow_4', 'ow_8', 'ow_16', 'ow_32', 'dw_8', 'dw_16', 'dw_32',
              'dw_64', 'dw_128', 'dcos_14', 'dsin_14', 'dcos_30', 'dsin_30']:
    prior_all_stats['indprod_num_products_'+ label + '_reg'] = (prior_all_stats['indprod_num_products_' + label] /
                                                                prior_all_stats['user_num_products_' + label])
    prior_all_stats.drop(['user_num_products_' + label], axis=1, inplace=True)


# calculate and merge in stats related to the time of the last (train/test) order
orders_df_last = orders_df[orders_df.eval_set != "prior"].copy()
orders_df_last['order_dow_angle'] = (orders_df_last.order_dow /
                                     (orders_df_last.order_dow.max() + 1) * 2 * np.pi)
orders_df_last['order_hod_angle'] = (orders_df_last.order_hour_of_day /
                                     (orders_df_last.order_dow.max() + 1) * 2 * np.pi)
orders_df_last['order_dow_sin'] = np.sin(orders_df_last.order_dow_angle)
orders_df_last['order_dow_cos'] = np.cos(orders_df_last.order_dow_angle)
orders_df_last['order_hod_sin'] = np.sin(orders_df_last.order_hod_angle)
orders_df_last['order_hod_cos'] = np.cos(orders_df_last.order_hod_angle)
orders_df_last.drop(["order_number", "order_dow_angle", "order_hod_angle"], axis=1, inplace=True)
prior_all_stats = prior_all_stats.merge(orders_df_last, on="user_id", how="inner")

# merge in stats on proportion of {user, product, user*product} orders at each {hod, dow}
prior_all_stats = prior_all_stats.merge(individual_day_idx, on=['user_id', 'order_dow'])
prior_all_stats = prior_all_stats.merge(product_day_idx, on=['product_id', 'order_dow'])
prior_all_stats = prior_all_stats.merge(indprod_day_idx, on=['user_id', 'product_id', 'order_dow'])
prior_all_stats = prior_all_stats.merge(individual_hod_idx, on=['user_id', 'order_hour_of_day'])
prior_all_stats = prior_all_stats.merge(product_hod_idx, on=['product_id', 'order_hour_of_day'])
prior_all_stats = prior_all_stats.merge(indprod_hod_idx, on=['user_id', 'product_id', 'order_hour_of_day'])

del individual_hod_idx
del product_hod_idx
del indprod_hod_idx
del individual_day_idx
del product_day_idx
del indprod_day_idx
import gc
gc.collect()

# add overall hod/dow priors into df.
prior_all_stats['days_prior'] = days_prior[prior_all_stats['order_dow']].tolist()
prior_all_stats['hod_prior'] = hod_prior[prior_all_stats['order_hour_of_day']].tolist()
prior_all_stats.drop(['order_hour_of_day', 'order_dow'], axis=1, inplace=True)
# use these priors to shift the proportion estimates for each product/user/user*product a bit
prior_all_stats.eval("product_day_proportion=(product_num_orders * product_day_proportion + 30 * days_prior)/(product_num_orders+30)",
                     inplace=True)
prior_all_stats.eval("product_hod_proportion=(product_num_orders * product_hod_proportion + 15 * hod_prior)/(product_num_orders+15)",
                     inplace=True)
prior_all_stats.eval("indprod_day_proportion=(indprod_num_orders * indprod_day_proportion + 10 * days_prior)/(indprod_num_orders+10)",
                    inplace=True)
prior_all_stats.eval("indprod_hod_proportion=(indprod_num_orders * indprod_hod_proportion + 5 * hod_prior)/(indprod_num_orders+5)",
                    inplace=True)
prior_all_stats.drop(['days_prior', 'hod_prior', 'reverse_date', 'reverse_order_number'], axis=1, inplace=True)


# calculate the last order's difference from the average hod/dow angle
prior_all_stats['indprod_dow_avg_diff'] = np.cos(np.arctan2(prior_all_stats.order_dow_sin, prior_all_stats.order_dow_cos) -
                                                np.arctan2(prior_all_stats.indprod_dow_sin, prior_all_stats.indprod_dow_cos))
prior_all_stats['indprod_hod_avg_diff'] = np.cos(np.arctan2(prior_all_stats.order_hod_sin, prior_all_stats.order_hod_cos) -
                                                np.arctan2(prior_all_stats.indprod_hod_sin, prior_all_stats.indprod_hod_cos))
prior_all_stats['user_dow_avg_diff'] = np.cos(np.arctan2(prior_all_stats.order_dow_sin, prior_all_stats.order_dow_cos) -
                                                np.arctan2(prior_all_stats.user_dow_sin, prior_all_stats.user_dow_cos))
prior_all_stats['user_hod_avg_diff'] = np.cos(np.arctan2(prior_all_stats.order_hod_sin, prior_all_stats.order_hod_cos) -
                                                np.arctan2(prior_all_stats.user_hod_sin, prior_all_stats.user_hod_cos))
prior_all_stats['product_dow_avg_diff'] = np.cos(np.arctan2(prior_all_stats.order_dow_sin, prior_all_stats.order_dow_cos) -
                                                np.arctan2(prior_all_stats.product_dow_sin, prior_all_stats.product_dow_cos))
prior_all_stats['product_hod_avg_diff'] = np.cos(np.arctan2(prior_all_stats.order_hod_sin, prior_all_stats.order_hod_cos) -
                                                np.arctan2(prior_all_stats.product_hod_sin, prior_all_stats.product_hod_cos))

# calculate number of distinct products each user has ordered
prior_all_stats['user_distinct_products'] = prior_all_stats.groupby('user_id')['product_id'].transform('count')
# calculate proportion of all products ordered by a user they order at each order
prior_all_stats['user_mean_proportion_products'] = (prior_all_stats.user_mean_products
                                                    / prior_all_stats.user_distinct_products)

# merge in training data outcomes
prior_all_stats = prior_all_stats.merge(train_df[['user_id', 'product_id', 'reordered']],
                                        how="left", on=["user_id", "product_id"])
prior_all_stats.reordered = prior_all_stats.reordered.fillna(0)

#separate out None into its own DF
none_df = prior_all_stats.query("product_id == 'None'")
none_df.drop(['product_dow_sin',
              'product_dow_cos', 'product_hod_sin', 'product_hod_cos',
              'product_num_orders', 'product_proportion_orders',
              'product_days_per_order', 'product_day_proportion',
              'product_hod_proportion', 'product_dow_avg_diff',
              'product_hod_avg_diff'], axis=1, inplace=True)
none_df.drop(['indprod_add_to_cart_order', 'indprod_add_to_cart_proportion'],
             axis=1, inplace=True)
prior_all_stats = prior_all_stats.query("product_id != 'None'")


# add proportion of None's as a user-level feature
none_proportion = none_df[['user_id', 'indprod_proportion_orders']]
none_proportion.columns = ['user_id', 'user_proportion_none']
prior_all_stats = prior_all_stats.merge(none_proportion, on='user_id')


# merge in product vectors
prior_all_stats = prior_all_stats.merge(short_vectors_df, on="product_id")

# create vectors for each user describing their "average" product in product vector space
vecs = ['dim'+str(i) for i in range(30)]
pos = prior_all_stats[['user_id', 'indprod_num_orders'] + vecs].copy()
pos['indprod_sum_orders'] = pos.groupby('user_id').indprod_num_orders.transform('sum')
# add some pull towards 0 for regularization
pos.loc[:, 'dim0':'dim29'] = pos.loc[:, 'dim0':'dim29'].multiply(
    pos['indprod_num_orders']/(pos.indprod_sum_orders + 15), axis=0)
user_mean_vecs = (pos.groupby('user_id').agg('sum')
                  .drop(['indprod_num_orders', 'indprod_sum_orders'], axis=1)
                  .reset_index())
# shrink dimensionality of user vectors
pca = PCA(n_components=15)
pca.fit(user_mean_vecs.loc[:, 'dim0':'dim29'])
short_vectors = pca.transform(user_mean_vecs.loc[:, 'dim0':'dim29'])
short_vectors = pd.DataFrame(short_vectors)
short_vectors.columns = ['user_dim'+str(c) for c in range(15)]
short_vectors['user_id'] = user_mean_vecs['user_id']
prior_all_stats = prior_all_stats.merge(short_vectors, on='user_id')
none_df = none_df.merge(short_vectors, on='user_id')

# get rid of unused variables. Helped with memory issues when creating hdf.
del prior_indprod_stats
del prior_individual_stats
del prior_product_stats
del short_vectors_df
del short_vectors
del user_mean_vecs
del prior_df
del train_df
del products_df_merged
del orders_df
del orders_df_last
del prior_day_idx
del prior_hod_idx
del prior_indorder_stats
del products_df
del vectors
print(gc.collect())

# create validation sets for use in later models
all_users = prior_all_stats.loc[prior_all_stats.eval_set == "train", "user_id"].unique()
np.random.seed(1234)
np.random.shuffle(all_users)
valid_set = pd.DataFrame({'user_id': all_users,
                          'validation_set': np.arange(0, all_users.shape[0]) % 11})
prior_all_stats = prior_all_stats.merge(valid_set, on='user_id', how='left')
prior_all_stats.validation_set = prior_all_stats.validation_set.fillna(-1)
none_df = none_df.merge(valid_set, on='user_id', how='left')
none_df.validation_set = none_df.validation_set.fillna(-1)


prior_all_stats['extratrain'] = is_extra
none_df['extratrain'] = is_extra

prior_all_stats.product_id = prior_all_stats.product_id.astype(int)
# create useful key for deciding which samples to use in which training folds later on
prior_all_stats['subset_key'] = np.random.randint(0, 252, (prior_all_stats.shape[0]))

if is_extra:
    prior_all_stats.to_hdf("data/prior_all_stats_extratrain.h5", "table",
                           format='table', data_columns=['eval_set', 'validation_set'])
    none_df.to_hdf("data/none_stats_extratrain.h5", "table")
else:
    prior_all_stats.to_hdf("data/prior_all_stats.h5", "table",
                           format='table', data_columns=['eval_set', 'validation_set'])
    none_df.to_hdf("data/none_stats.h5", "table")
