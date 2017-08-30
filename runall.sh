#!/bin/bash

# create data set 
python preprocess.py
# create data set for second-to-last orders
python preprocess.py extra

# run model on "none" orders
python none_analysis.py

# run all folds of model in xgboost
for i in {0..9}
do
	  python analysis_xgb.py $i
done

# run all folds of model in lightgbm
for i in {0..9}
do
	  python analysis_lgbm.py $i
done

# link data to sh1ng model subfolder
ln -s data imba/data

cd imba

# create sh1ng data set
python create_products.py
python split_data_set.py
python orders_comsum.py
python user_product_rank.py
python create_prod2vec_dataset.py
python skip_gram_train.py
python skip_gram_get.py
python makedata_main.py

# create sh1ng data set for second-to-last orders
python convert_train.py
python create_products.py extra
python split_data_set.py extra
python orders_comsum.py extra
python user_product_rank.py extra
python makedata_main.py extra

# run all folds of sh1ng model
for i in {0..9}
do
	  python analysis_sh1ng.py $i
done

cd ..
# combine predictions and use GFM algorithm to predict baskets
python stacking.py
