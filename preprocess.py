
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#% matplotlib inline


# In[2]:

aisles_df = pd.read_csv("data/aisles.csv")
departments_df = pd.read_csv("data/departments.csv")
products_df = pd.read_csv("data/products.csv")
orders_df = pd.read_csv("data/orders.csv")
prior_df = pd.read_csv("data/order_products__prior.csv")
train_df = pd.read_csv("data/order_products__train.csv")
products_df_merged = (products_df
                      .merge(departments_df, on="department_id")
                      .merge(aisles_df, on="aisle_id"))

products_df_merged.to_hdf("data/testing.h5", "table")


# In[3]:

orders_df['absolute_date'] = orders_df.groupby("user_id").days_since_prior_order.cumsum().fillna(0)


# In[4]:

orders_df['max_order_number'] = orders_df.groupby("user_id").order_number.transform(max)
orders_df['max_absolute_date'] = orders_df.groupby("user_id").absolute_date.transform(max)


# In[5]:

orders_df['reverse_date'] = orders_df.max_absolute_date - orders_df.absolute_date
orders_df['reverse_order_number'] = orders_df.max_order_number - orders_df.order_number


# In[6]:

train_df = train_df.merge(orders_df[["order_id", "user_id"]], on="order_id")


# In[7]:

prior_df = prior_df.merge(orders_df, on="order_id")


# In[8]:

prior_df['order_dow_angle'] = (prior_df.order_dow / 
                                     (prior_df.order_dow.max() + 1) * 2 * np.pi)
prior_df['order_hod_angle'] = (prior_df.order_hour_of_day / 
                                             (prior_df.order_dow.max() + 1) * 2 * np.pi)


# In[9]:

prior_df['order_dow_sin'] = np.sin(prior_df.order_dow_angle)
prior_df['order_dow_cos'] = np.cos(prior_df.order_dow_angle)
prior_df['order_hod_sin'] = np.sin(prior_df.order_hod_angle)
prior_df['order_hod_cos'] = np.cos(prior_df.order_hod_angle)
prior_df['num_products'] = 1
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


# In[10]:

prior_product_stats = prior_df.groupby("product_id").agg({'order_dow_sin': np.sum, 
                                                                'order_dow_cos': np.sum, 
                                                                'order_hod_sin': np.sum,
                                                                'order_hod_cos': np.sum,
                                                               'num_products': np.sum})
prior_product_stats['order_dow_angle'] = np.arctan2(prior_product_stats.order_dow_sin, prior_product_stats.order_dow_cos)
prior_product_stats['order_hod_angle'] = np.arctan2(prior_product_stats.order_hod_sin, prior_product_stats.order_hod_cos)


# In[11]:

prior_product_stats.order_dow_sin = np.sin(prior_product_stats.order_dow_angle)
prior_product_stats.order_dow_cos = np.cos(prior_product_stats.order_dow_angle)
prior_product_stats.order_hod_sin = np.sin(prior_product_stats.order_hod_angle)
prior_product_stats.order_hod_cos = np.cos(prior_product_stats.order_hod_angle)
prior_product_stats.drop(['order_dow_angle', 'order_hod_angle'], axis=1, inplace=True)
prior_product_stats.reset_index(inplace=True)
prior_product_stats.columns = ['product_id', 'product_dow_sin', 'product_dow_cos', 'product_hod_sin', 'product_hod_cos', 'product_num_purchases']


# In[12]:

prior_indorder_stats = prior_df.groupby(["user_id", "order_id"]).agg({'order_dow_sin': np.sum, 
                                                                'order_dow_cos': np.sum, 
                                                                'order_hod_sin': np.sum,
                                                                'order_hod_cos': np.sum,
                                                               'num_products': np.sum,
                                                                      'absolute_date': np.max,
                                                                     'order_number': np.max}).reset_index()


# In[13]:

prior_individual_stats = prior_indorder_stats.groupby("user_id").agg({'order_dow_sin': np.sum, 
                                                                'order_dow_cos': np.sum, 
                                                                'order_hod_sin': np.sum,
                                                                'order_hod_cos': np.sum,
                                                               'num_products': [np.sum, np.mean],
                                                                'absolute_date': np.max,      
                                                                'order_number': np.max})


# In[14]:

prior_individual_stats.columns = ['order_dow_sin', 'order_dow_cos',
                                  'order_hod_sin', 'order_hod_cos', 
                                   'num_products', 'mean_products', 'max_absolute_date', "max_order_number"]


# In[15]:

prior_individual_stats['order_dow_angle'] = np.arctan2(prior_individual_stats.order_dow_sin, prior_individual_stats.order_dow_cos)
prior_individual_stats['order_hod_angle'] = np.arctan2(prior_individual_stats.order_hod_sin, prior_individual_stats.order_hod_cos)
prior_individual_stats.order_dow_sin = np.sin(prior_individual_stats.order_dow_angle)
prior_individual_stats.order_dow_cos = np.cos(prior_individual_stats.order_dow_angle)
prior_individual_stats.order_hod_sin = np.sin(prior_individual_stats.order_hod_angle)
prior_individual_stats.order_hod_cos = np.cos(prior_individual_stats.order_hod_angle)
prior_individual_stats.drop(['order_dow_angle', 'order_hod_angle'], axis=1, inplace=True)


# In[16]:

prior_individual_stats = prior_individual_stats.reset_index()


# In[17]:

prior_individual_stats.columns = ['user_id', 'user_dow_sin',
                                  'user_dow_cos', 'user_hod_sin',
                                  'user_hod_cos', 'user_num_products',
                                  'user_mean_products', 'user_num_days',
                                 'user_num_orders']
prior_individual_stats['user_days_per_order'] = prior_individual_stats.user_num_days / prior_individual_stats.user_num_orders


# In[18]:

prior_indorder_stats = prior_indorder_stats[['order_id', 'num_products']]
prior_indorder_stats.columns = ['order_id', 'num_products_in_order']


# In[19]:

prior_indprod_stats = (prior_df.merge(prior_indorder_stats[['order_id', 'num_products_in_order']], on='order_id')
                       .merge(prior_individual_stats[['user_id', 'user_num_orders', 'user_num_days']], on='user_id', how='left'))
prior_indprod_stats['add_to_cart_proportion'] = prior_indprod_stats['add_to_cart_order'] / prior_indprod_stats['num_products_in_order']
prior_indprod_stats['indprod_inorder_1'] = 1 * (prior_indprod_stats.order_number == prior_indprod_stats.user_num_orders)
prior_indprod_stats['indprod_inorder_2'] = 1 * (prior_indprod_stats.order_number == prior_indprod_stats.user_num_orders - 1)
prior_indprod_stats['indprod_inorder_3'] = 1 * (prior_indprod_stats.order_number == prior_indprod_stats.user_num_orders - 2)
prior_indprod_stats = prior_indprod_stats.groupby(["user_id", "product_id"]).agg({'order_dow_sin': np.sum, 
                                                                'order_dow_cos': np.sum, 
                                                                'order_hod_sin': np.sum,
                                                                'order_hod_cos': np.sum,
                                                               'num_products': np.sum,
                                                               'num_products_dw_8': np.sum,
                                                                'num_products_dw_16': np.sum,
                                                                'num_products_dw_32': np.sum,
                                                                'num_products_dw_64': np.sum,
                                                                'num_products_dw_128': np.sum,
                                                                'num_products_ow_2': np.sum,           
                                                                'num_products_ow_4': np.sum,
                                                                'num_products_ow_8': np.sum,                  
                                                                'num_products_ow_16': np.sum,
                                                                'num_products_ow_32': np.sum,                                   
                                                           'add_to_cart_order': np.mean,
                                                           'add_to_cart_proportion': np.mean,
                                                           'indprod_inorder_1': np.sum,
                                                           'indprod_inorder_2': np.sum,
                                                           'indprod_inorder_3': np.sum,
                                                            'user_num_orders': np.mean,
                                                            'user_num_days': np.mean,
                                                            'reverse_date': np.min,
                                                            'reverse_order_number': np.min}).reset_index()


# In[20]:

prior_indprod_stats['order_dow_angle'] = np.arctan2(prior_indprod_stats.order_dow_sin, prior_indprod_stats.order_dow_cos)
prior_indprod_stats['order_hod_angle'] = np.arctan2(prior_indprod_stats.order_hod_sin, prior_indprod_stats.order_hod_cos)
prior_indprod_stats['proportion_orders'] = prior_indprod_stats.num_products / (prior_indprod_stats.user_num_orders)
prior_indprod_stats['days_per_order'] = prior_indprod_stats.user_num_days / (prior_indprod_stats.num_products)


# In[21]:

prior_indprod_stats.order_dow_sin = np.sin(prior_indprod_stats.order_dow_angle)
prior_indprod_stats.order_dow_cos = np.cos(prior_indprod_stats.order_dow_angle)
prior_indprod_stats.order_hod_sin = np.sin(prior_indprod_stats.order_hod_angle)
prior_indprod_stats.order_hod_cos = np.cos(prior_indprod_stats.order_hod_angle)
prior_indprod_stats.drop(['order_dow_angle', 'order_hod_angle', 'user_num_orders', 'user_num_days'], axis=1, inplace=True)


# In[22]:

prior_indprod_stats.columns = ['user_id', 'product_id', 'indprod_dow_sin', 
                               'indprod_dow_cos', 'indprod_hod_sin',
                               'indprod_hod_cos', 'indprod_num_products',
                               'indprod_num_products_dw_8', 'indprod_num_products_dw_16', 
                               'indprod_num_products_dw_32', 'indprod_num_products_dw_64', 
                               'indprod_num_products_dw_128', 'indprod_num_products_ow_2',
                               'indprod_num_products_ow_4', 'indprod_num_products_ow_8',
                               'indprod_num_products_ow_16', 'indprod_num_products_ow_32',
                               'indprod_add_to_cart_order', 'indprod_add_to_cart_proportion',
                               'indprod_inorder_1', 'indprod_inorder_2', 'indprod_inorder_3',
                               'indprod_days_since_last', 'indprod_orders_since_last',
                                'indprod_proportion_orders', 'indprod_days_per_order']


# In[23]:

# use indprod means to add more product stats
product_order_proportions = (prior_indprod_stats[['user_id', 'product_id', 
                                                  'indprod_proportion_orders', 'indprod_days_per_order']]
                             .groupby("product_id")
                             .agg({'indprod_proportion_orders': np.mean,
                                  'indprod_days_per_order': np.mean})).reset_index()
product_order_proportions.columns = ['product_id', 'product_proportion_orders', 'product_days_per_order']
prior_product_stats = prior_product_stats.merge(product_order_proportions, on='product_id')


# In[24]:

prior_all_stats = prior_indprod_stats.merge(prior_individual_stats, on="user_id", how="left").merge(prior_product_stats, on='product_id', how="left")


# In[25]:

orders_df_last = orders_df[orders_df.eval_set != "prior"].copy()
orders_df_last['order_dow_angle'] = (orders_df_last.order_dow / 
                                     (orders_df_last.order_dow.max() + 1) * 2 * np.pi - np.pi)
orders_df_last['order_hod_angle'] = (orders_df_last.order_hour_of_day / 
                                             (orders_df_last.order_dow.max() + 1) * 2 * np.pi - np.pi)
orders_df_last['order_dow_sin'] = np.sin(orders_df_last.order_dow_angle)
orders_df_last['order_dow_cos'] = np.cos(orders_df_last.order_dow_angle)
orders_df_last['order_hod_sin'] = np.sin(orders_df_last.order_hod_angle)
orders_df_last['order_hod_cos'] = np.cos(orders_df_last.order_hod_angle)


# In[26]:

orders_df_last.drop(["order_number", "order_dow", "order_hour_of_day", "order_dow_angle", "order_hod_angle"], axis=1, inplace=True)


# In[27]:

prior_all_stats = prior_all_stats.merge(orders_df_last, on="user_id", how="inner")


# In[28]:

products_df_merged = products_df_merged.join(pd.get_dummies(products_df_merged.aisle))


# In[29]:

products_df_merged.drop(['product_name', 'aisle_id', 'department_id', 'department', 'aisle'], axis=1, inplace=True)


# In[30]:

prior_all_stats = prior_all_stats.merge(products_df_merged, on="product_id")

prior_all_stats['user_distinct_products'] = prior_all_stats.groupby('user_id')['product_id'].transform('count')


# In[31]:

prior_all_stats = prior_all_stats.merge(train_df[['user_id', 'product_id', 'reordered']], how="left", on=["user_id", "product_id"])
prior_all_stats.reordered = prior_all_stats.reordered.fillna(0)

prior_all_stats.to_hdf("data/prior_all_stats.h5", "table")
