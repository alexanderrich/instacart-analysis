# Instacart Market Basket Analysis, 23rd Place Solution

This repository contains scripts to produce the 23rd place solution
to Kaggle's [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) prediction competition. 

The code was run using python 3.5 and requires numpy, pandas, scikit-learn,
xgboost, lightgbm, and numba. To run the complete solution, download the data
files from kaggle into the `data` folder and execute `runall.sh`. Running the scripts will require
around 64GB RAM. 

# Data and objective

The raw data consists of several tables that, in combination, contain the
complete order histories of ~200,000 shoppers. This includes: 

* The products in each order 
* The time of day and day of week of each order
* The name and aisle/department of each product

The goal of the competition is to predict, for each individual, which products
that they have purchased before they will purchase again in their final order.
For the training set, we are given the participants' final order. For the
training set, we are given the time and day of their final order, but not the
products they ordered.

Predictions for the competition are submitted as an order ID and a list of
product IDs predicted for that order. Predictions can include `None`, a special
product ID indicating that no previously ordered products have been ordered.
`None` can be combined with other product IDs in prediction, but will of course
only occur on its own in the ground truth.

Predictions are evaluated using the average
[F1-score](https://en.wikipedia.org/wiki/F1_score) across orders.

# Preprocessing

The raw data is spread across 7 tables, and isn't amenable to standard
classification algorithms. To create a tractable data set, I built a table with
a row for each (individual, product) combination that occurs in the prior
orders, and constructed a set of features based on the attributes and order
history of the individual, the product, and the (individual, product)
combination. Along with these features was included a target column indicating
whether the individual in fact ordered the given product in their final order.

Details of the feature-engineering process can be found in `preprocess.py`, but
here are some highlights:

## Simple summary statistics
* Number of orders for {individual, product, (individual, product)}, average
  reorder proportion for product among people who order it, individual's average
  time between orders and total number of distinct items ordered, etc.

## Time of day/day of week features
* Features coding what proportion of {individual, product, (individual,
  product)} orders were on the same {hour of day, day of week} as the final
  order.
* Features coding the average order {hour of day, day of week} for the
  {individual, product, (individual, product)}. Since hour of day and day of
  week are inherently circular, I converted these values into angular space and
  took a circular average.
  
## Recency-weighted past orders
* An individual might be more likely to reorder a product that they've ordered a
  lot recently than one they ordered a lot a long time ago. To capture this time
  dependence, I created a set of recency-weighted predictors that summed the
  past (individual, product) orders with exponential decay on the order weights.
  I created these predictors with decay on both the day dimension (# of days
  since order) and order dimension (# of orders since order), and with a range
  of different decay rates. I also created weighted predictors with sinusoidal
  weights to capture bi-weekly (14 day) and monthly patterns.

## Product description vectors
* For each product, we are given the product's name and the name of it's aisle
  and department. To change this text into usable features, I concatenated the
  name, aisle and department together and used an off-the-shelf GloVe model to
  embed it into a semantically meaningful vector space. I then performed
  principal components analysis to reduce these vectors into a manageable
  30-dimensional space, and included them as features.

## User vectors
* From the product vectors, I created vectors coding the average product
  purchased by each individual, and included these as well.
  
Following feature engineering, I divided the training data into 11 validation
sets at the level of individuals.

# Extra training data

To create a larger training set, I reran the feature engineering script, only
this time discarding the *actual* final order and using the *second-to-last*
order as the target. This makes it possible to roughly double the training set.
This could in theory also be performed using the third-to-last order, etc, but
likely produces diminishing returns, particularly because the number of prior
orders used to engineer features shrinks with each discarded order.

# "None" model using xgboost

During feature engineering, I included "None" as a product by stipulating that
it had been purchased in any order where no previously-ordered products were
purchased. At the end of feature engineering, I separated the "None" rows into a
separate table, as the predictors of a None order are likely different than
those of other products. I trained a set of 10 xgboost models on the data. Each
model used all of the "extra" training data, and 9 validation folds of the
original training data. One validation fold was used to determine the
early-stopping point for the model, and the 11th fold was excluded from all 10
models. After training, the 10 models were averaged together to product
predictions for the 11th validation fold and for the test set.

# Xgboost and lightgbm models

The main data set was also modeled using gradient boosting machines. Because of
the size of the data, I wasn't able to load the entire data set (including the
"extra" training data) into memory at once. To get around this each model was
trained on a subset of the data. As with the None data, I ran 10 models on the
data. Each model was run on 50% of the "extra" training data, and 50% of data
from 9 validation folds of the original training data. As with the None model,
the 11th fold was excluded from all 10 models, and the remaining fold was used
to determine early stopping. The predictions of the 10 models were averaged
after fitting.

For most of the competition, I trained the main model using the xgboost package.
Late in the competition, I trained a set of models using lightgbm as well. This
packages uses a slightly different tree construction algorithm, and allows
categorical predictors, meaning I could include "aisle" and "department"
directly in the model. Lightgbm also ran must faster for this data set&mdash;if
I had started using it sooned, I might have been able to do more experimentation!

# Sh1ng solution features and model

In the final week of the competition, a competitor going by username sh1ng
released his entire solution code. To maintain competitiveness, I used this
model along with the models I trained on my own features. I forked the sh1ng
model and made a few small modifications so that it would include the "extra"
training data and run using the same 10-fold structure I used for my other models.

# Model averaging

After producing average model predictions for each of the three model type
(xgboost, lightgbm, and sh1ng), I combined them to form a single set of
predictions. To do this I ran a simple logistic regression using the three
models' predictions to predict the orders in the 11th validation fold, which had
been held out of training for all three model types. This produced the optimal
weights with which to combine the three models.

# Converting probabilities into predicted market baskets

Having produced predicted probabilities for each product, and for None, the
final step was to choose which items to actually include in the market basket
prediction to maximize F1 score. To do this I implemented the algorithm outlined
in the NIPS paper [An Exact Algorithm for F-Measure
Maximization](https://papers.nips.cc/paper/4389-an-exact-algorithm-for-f-measure-maximization.pdf).
While there are some simpler approximate methods for maximizing F1 scores, they
tend to assume that the labels (i.e., products) are independent. This is a
reasonable (thought certainly not fully correct) assumption for the products
except for None, but None is clearly non-independent from all of the other
products. More details of the algorithm and its implementation can be found in
`stacking.py`.
