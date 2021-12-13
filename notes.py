# How to 1-Hot Encode
categorical_variables = [
  'make',
  'engine_fuel_type',
  'transmission_type',
  'driven_wheels',
  'market_category',
  'vehicle_size',
  'vehicle_style',
]
categories = {
 category: df[category].values_counts().head().index
   for category in categorical_variables
 }

 for category, values in categories.items():
   for value in values:
     feature = f"{category}_{value}"
     df[feature] = (df[category] == value).astype(int)
     features.append(feature)


# Feature Importance
1) Diff = Mean(Global) - Mean(Group)
   if Diff > 0 less likely to churn,
   if Diff < 0 more likely ro churn,
2) Risk Ratio = Mean(Group) / Mean(Global)
   Ratio > 1 => More likely to Churn
   Ratio < 1 => Less likely to Churn


# Determing which Categorical Features are important
for col in categorical:
  df_group = df_full_train.groupby(col)['churn'].agg(['mean', 'count'])
  df_group['diff'] = df_group['mean'] - global_churn
  df_group['risk'] = df_group['mean'] / global_churn
  print(df_group)

from sklearn.metrics import mutual_info_score
def mutual_info_churn_score(series):
  return mutual_info_score(df_full_train["churn"], series)
df_full_train.apply(mutual_info_churn_score).sort_values(ascending=False)


# Determining which numerical features are important

# Correlation Coefficient (Pearsons)
numerical = ['tenure', 'monthlycharges', 'totalcharges']
df_full_train[numerical].corrwith(df_full_train['churn'])


# 1-Hot Encoding
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# Note: can pass numerical values to DictVectorizer as well
train_dicts = df_train[categorical + numerical].to_dict(orient='records')
#Method 1
dv = DictVectorizer(sparse=False)
dv.fit(train_dicts)
X_train = dv.transform(train_dicts)
#Method 2
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

dv.get_feature_names()


def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def linear_regression(x_i):
  result = w0
  for j in range(len(w)):
    result += x_i[j] * w[j]
  return result

def logistic_regression(x_i):
  score = w0
  for j in range(len(w)):
    score += x_i[j] * w[j]
  result = sigmoid(score)
  return result


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
w = model.coef_[0].round(3)
b = model.intercept_[0]
model.predict(X_train)        # Output is 1s and 0s
model.predict_proba(X_train)  # Output is probabilities (Second cal is prb of churning)

y_pred = model.predict_proba(X_train)[:, 1]
churn_decision = (y_pred >= 0.50)
df_val[churn_decision]["customerid"]

y_val = model.predict_proba(X_val)[:, 1]
churn_decision = (y_val >= 0.50)
(y_val == churn_decision).mean()  <== Good number for model score

df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
df_pred['correct'] = df_pred['prediction'] == df_pred['actual']
df_pred['correct'].mean()

dict(zip(dv.get_feature_names(), model.coef_[0].round(3)))

small = ['contract', 'tenure', 'monthlycharges']
dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')

dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)

X_train_small = dv_small.transform(dicts_train_small)
model_small = LogisticRegression()
model_small.fit(X_train_small, y_train)

w0 = model_small.intercept_[0]
w = model_small.coef_[0]

dict(zip(dv_small.get_feature_names(), w.round(3)))



customer = dicts_test[10]
X_small = dv.transform([customer])
model.predict_proba(X_small)[0, 1]
y_test[10]


# 4.2 Accuracy
from sklearn.metrics import accuracy_score
scores = []
thesholds = np.linspace(0, 1, 21)
for t in thresholds:
  #churn_decision = (y_pred >= t)
  #score = (y_val == churn_decision).mean()
  score = accuracy_score(y_val, y_pred >= t)
  scores.append(score)
  print(f"{theshold:.2f} {score:.3f}")
plt.plot(thesholds, scores)

# 4.3 Confusion Table
# Churning data had class imbalance (the data had 70% nonchurning, so
# isntead of a model, just saying no one will churn is 70% accurate)

"""
True Positive:  g(x) >= threshold & y == 1
True Negative:  g(x) <  threshold & y == 0
False Positive: g(x) >= threshold & y == 0
False Negative: g(x) <  threshold & y == 1
"""

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

tp = (predict_positive & actual_positive).sum()  # True Positive, 210
tn = (predict_negative & actual_negative).sum()  # True Negative, 922
fp = (predict_positive & actual_negative).sum()  # False Positive, 101
fn = (predict_negative & actual_positive).sum()  # False Negative, 176

"""
Confusion Table, 2x2 table

                          Predicitons
                     g(x) < t         g(x) >= t
                     Negative         Positive
NegActual y == 0     922              101
PosActual y == 1     176              210 <- This row is Recall
                                       ^- This col is Precision
"""
confusion_matrix = np.array([
  [tn, fp],
  [fn, tp],
])
(confusion_matrix / confusion_matrix.sum()).round(2)  # Normalize
# Accuracy / Score is also the sum of TN and TP

"""
4.4 Precision and Recall
"""
accuracy = (tp + tn) / (tp + tn + fp + fn)

"""
Precision = Fraction of Positive Predicitions that are correct
(67% precise, so 33% of people getting promotional email are mistakes)
"""
precision = tp / (tp + fp)

"""
Recall = Fraction of Correctly Identified Positive Examples
(54% recall, so 46% who are churning, we failed to identify them)
"""
recall = tp / (tp + fn)

"""
So accuracy of 80% but precision of 67% and recall of %54, not so good
When you have class imbalance, accuracy is not very good
"""


"""
4.5 ROC Curves (Receiver Operator Characteristics)

Interested in FPR FalsePositive Rate and TPR TruePositiveRate
"""
false_positive_rate = fp / (tn + fp)  # First row of Confusion Matrix
true_positive_rate = tp / (fn + tp)   # Second row of Confusion Matrix
# TPR === Recall

# Want to minimize false positive rate and maximioze true positive rate
# ROC curve parametrizes TPR and FPR with threshold

def tpr_fpr_dataframe(y_val, y_pred):
  scores = []
  thresholds = np.linspace(0, 1, 101)
  for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    scores.append((t, tp, tn, fp, fn))

  df = pd.DataFrame(
    data=scores,
    columns=['threshold', 'tp', 'tn', 'fp', 'fn'],
  )
  df['tpr'] = df['tp'] / (df['tp'] + df['fn'])
  df['fpr'] = df['fp'] / (df['fp'] + df['tn'])
  return df

df = tpr_fpr_dataframe(y_val, y_pred)
plt.plot(df.threshold, df['trp'], label='TPR')
plt.plot(df.threshold, df['frp'], label='FPR')

# Compare to a Random Model
np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))
df_rand = tpr_fpr_dataframe(y_val, y_rand)

plt.plot(df.threshold, df['trp'], label='TPR')  # Straight y = -x + 1 line
plt.plot(df.threshold, df['frp'], label='FPR')

"""
Ideal Model
"""
num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal_pred = np.linspace(0, 1, len(y_val))
t = 1 - y_val.mean()
(y_idea_pred >= t) == y_idea).mean()

df_ideal = tpr_fpr_dataframe(y_idea, y_ideal_pred)
df_idea[::10]
plt.plot(df_ideal.threshold, df_ideal['trp'], label='TPR')
plt.plot(df_ideal.threshold, df_ideal['frp'], label='FPR')


plt.figure(figsize=(6, 6))
plt.plot(df['fpr'], df['tpr'], label='model')
#plt.plot(df_rand['fpr'], df_rand['tpr'], label='random')
plt.plot([0, 1], [0, 1], label='random')
plt.plot(df_ideal['fpr'], df_ideal['tpr'], label='ideal')

plt.xlabel('FPR')
plt.ylabel('TPR')

# Want our model far away from y=x and close to top left corner
# (top left corner == ideal model)

"""
4.6 ROC Curves with sklearn
4.7 AUC (Area under ROC curve)
KEY TAKEAWAY
Ideal Model AUC == 1
Random Model AUC == 0.5

"""
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
plt.plot(fpr, tpr, label='model')
auc(fpr, tpr)

from sklearn.metrics import roc_auc_score
roc_auc_score(y_val, y_pred)

# roc_auc_score == Prob(randomly selcted pos > randomly selected neg)
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]
import random
n = 10000
success = 0
for i in range(n)
  pos_ind = random.randint(0, len(pos) - 1)
  neg_ind = random.randint(0, len(neg) - 1)
  if (pos[pos_ind] > neg[neg_ind]):
    success += 1

our_auc_score = success / n

# Using numpy
n = 10000
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

(pos[pos_ind] > neg[neg_ind]).mean()

"""
4.7 (K-Fold) Cross Validation
Parameter tuning > selecting best parameter

Data = [val, train, test]
Full Train = [val, train]
Split Full Train into k=3 parts
Train on 1,2, val on 3 (calculate AUC on 3)
Train on 1,3, val on 2 (calculate AUC on 2)
Train on 2,3, val on 1 (calculate AUC on 1)
auc1, auc2, auc2 => mean auc => std auc (std will show how stable model is)

"""
def train(df_train, y_train, C=1.0):
  dicts = df_train[categorical + numerical].to_dict()
  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(dicts)
  model = LogisticRegresion(C=C, max_iter=1000)
  model.fit(X_train, y_train)
  return dv, model

dv, model = train(df_train, y_train)

def predict(df, dv, model):
  dicts = df[categorial + numerical].to_dict(orient='records')
  X = dv.transform(dicts)
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred

y_pred = predict(df_val, dv, model)


from sklearn.model_selection import KFold
from tqdm.auto import tqdm  # Times loops

n_splits = 5
# Takes about 2 mins
for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
  scores = []
  kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
  for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

  print(f"{C}: {np.mean(scores):3.f} +/- {np.std(scores):.3f}")

# Best C  is now 1
# Train final model
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)
auc


"""
5 Deployment

5.1 Overview
5.2 Saving and loading the model
"""
# Pick to store and load DictVectorizor and Model
X = dv.transform([customer])
model.predict_proba(X)[0, 1]  # i = probability not churn, j = probability will churn

# 5.3, 5.4 Intro to Flask
from flask import Flask


app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
  return "PONG"

# 5.5 Pipenv
# pipenv shell to activate

# 5.6 Docker
# hub.docker.com/_python
#   sudo update-alternatives --config iptables
#       Select 1
#   sudo service docker start
#   sudo groupadd docker
#   sudo usermod -aG docker ${USER}
#   su -s ${USER}
#   docker run -it --rm python:3.8.12-slim

# 6 Decision Trees
# 6.1 Credit Risk Scoring Intro
# 6.2 Data Cleaning and Prep
df['status'] = df['status'].map({1: 'ok', 2: 'default', 0: 'unknown'})
# THIS IS FROM HW4
# .. other categorical
df.describe()  # Reveals 99999999 = nan
for c in ['income', 'assets', 'debt']:
  df[c].replace(to_replace=df[c].max(), value=np.nan)

df['status'].value_counts()
df.loc[:, df['status'] != 'unknown'].reset_index(drop=True)

fro sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = (df_train['status'] == 'default').astype(int).values
y_val = (df_val['status'] == 'default').astype(int).values
y_test = (df_test['status'] == 'default').astype(int).values

del df_train['status']
del df_val['status']
del df_text['status']






# 6.3 Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

train_dicts = df_train.fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_pred = dt.predict_proba(X_val)[:, 1]  # Two columns, proba False|True
roc_auc_score(y_val, y_pred)
# == 0.65484 (Bad)

# Check for overfitting
y_prd = dt.predict_proba(X_train)[:, 1]
roc_auc_score(y_train, y_pred)
# == 1.0 (Perfect) -> Overfitting (due to infinite depth, memorized answers)


# Restrict to 3 levels
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
# 0.776

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
# 0.739

# Look inside tree
from sklearn.tree import export_text
print(export_text(dt, feature_names=dv.get_feature_names()))

# 6.5 Decision Trees Parameter Tuning

for d in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
  dt = DecisionTreeClassifier(max_depth=d)
  dt.fit(X_train, y_train)
  y_pred = dt.predict_proba(X_val)[:, 1]
  auc = roc_auc_score(y_val, y_pred)
  print(d, auc)

#4, 5, 6 > 76%
scores = []
for d in [4, 5, 6]:
  for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
    dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((d, s, auc))

df_scores = pd.DataFrame(
  data=scores,
  columns=['max_depth', 'min_samples_leaf', 'auc'],
)
df_scores.sort_values(by='auc', ascending=False).head()
df_scores_pivot = df_scores.pivot(
  index='min_samples_leaf',
  columns='max_depth',
  values='auc',
)
sns.heatmap(df_scores_pivot, annot=True, fmt='.3f')

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

# 6.6 Ensembles and random forest
from sklearn.ensemble import RandomForestClassifier

# n_estimators == "number of experts" or "number of trees"
scores = []

# Find best max_depth
for d in [5, 10, 15]:
  for n in range(10, 201, 10):
    rf = RandomForestClassifier(
      n_estimators=n,
      max_depth=d,
      random_state=1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((d, n, auc))

df_scores = pd.DataFrame(
  data=scores,
  columns=['max_depth', 'n_estimators', 'auc'],
)

for d in [5, 10, 15]:
  df_subset = df_scores[df_scores['max_depth'] == d]
  plt.plot(
    df_subset['n_estimators'],
    df_subset['auc'],
    label=f'max_depth={d}',
  )
max_depth = 10  # Best based on graph

# Find best leaf size
for s in [1, 3, 5, 10, 50]:
  for n in range(10, 201, 10):
    rf = RandomForestClassifier(
      n_estimators=n,
      max_depth=max_depth,
      min_samples_leaf=s,
      random_state=1,
    )
    rf.fit(X_train, y_train)
    rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((s, n, auc))

df_scores = pd.DataFrame(
  data=scores,
  columns=['min_samples_leaf', 'n_estimators', 'auc'],
)

for s in [1, 3, 5, 10, 50]:
  df_subset = df_scores[df_scores['min_samples_leaf'] == s]
  plt.plot(
    df_subset['n_estimators'],
    df_subset['auc'],
    label=f'min_samples_leaf={d}',
  )

# 1, 3, 5 decide best s = 3
min_samples_leaf = 3
n = 100
rf = RandomForestClassifier(
  n_estimators=n,
  max_depth=max_depth,
  min_samples_leaf=min_samples_leaf,
  random_state=1,
  n_jobs=-1,
)
rf.fit(X_train, y_train)
rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
scores.append((s, n, auc))


# 6.7 Gradient Boosting and XGBoost
pip install xgboost

import xgboost as xgb

features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
dtest = xgb.DMatrix(X_test, feature_names=features)  # No label

xgb_params = {
  'eta': 0.3,             # Learning rate
  'max_depth': 6,
  'min_child_weight': 1,  # Same as min_samples_leaf

  'objective': 'binary:logistic', # Binary Classification, using logistic models
  'eval_metric': 'auc',
  'nthreads': 6,  # num cores

  'seed': 1,
  'verbosity': 1,
}

watchlist = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(
  xgb_params,
  dtrain,
  num_boost_round=10,
  evals=watchlist,
  verbose_eval=5,
)
y_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_pred)

# AUC goes to 1.0 on train, so the model overfits


# 6.8 XGBoost Parameter Tuning


# 6.9 Selecting the Best Model


# 8.0 Neural Netowrks and Deep Learning
# 8.1 Fashion Classification (Image Data)
# Multiclass Classificaiton from Images

# 8.2 TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

path = '../clothing-dataset-small/train/t-shirt'
name = '[HASH]'
img = load_img(f'{path}/{name}', target_size=(299, 299))  # PIL class
x = np.array(img)
x.shape # (299, 299, 3) array (3 = RGB) uint8 (unsigned, so 0-255)

# 8.3 Pretrained convolutional neural networks
# https://keras.io/api/applications/


from tensorflow.keras.applications.xception import Xception
model = Xception(
  weights='imagenet',
  input_shape=(299, 299, 3),
)
X = np.array([x]) # ([x, y, z]) for three images, etc
model.predict(X)  # Output is zero, bc no preprocessing

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
X = preprocess_input(X)  # Output not in 0-255, but [-1, 1], which model needs

pred = model.predict(X)
pred.shape # (1, 1000)
decode_prediction(pred)A


# 8.4 Convolutional Neural Networks
"""
Mostly used for images
Made up of layers, two main ones are convolutional and dense
Convolutional: Make up of "filters" of basic shapes that slide
around the image and score the similarity between the basic shape and a
cell-subset of the image. This produces a "Feature Map" for each basic
shape (aka Feature)

Pooling layers reduce the number of features in convolutional layers

Output of Convolutional Layers is a vector. This can be the input of a
dense layer, role of dense layer is to make final predicition.

Binary Classification:
Get weights from model, then vector * weights => summed => sigmoid
for logisitic/single classification => outout is prob of being tshirt


Multiclass Classification
Three models, three weights (for shirts, t-shirt, dress)
Calculate summed weights for each model, then we have probabilities of
which of the tree it is. Sigmoid for multiple classes is called softmax
this output will gives you three proba numbers.

"""

# 8.5 Transfer Learning
"""
Use already trained model on imagenet

The part where image is turned into vector by conv layers
is quite generic, and can be re-used for different models.
It also requires lots of images and good training, but not
specific to dataset (but Dense layer is specific to the problem)

Keep conv layers from image net, but make our own dense ones.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_ds = train_gen.flow_from_directory('./clothing-dataset-small/train/', target_size=(150, 150), batch_size=32)

# Batch is (32, 150, 150, 3) (RGB)

train_ds.class_indices # Class names come from folder names
X, y = next(train_ds)
X.shape (32, 150, 150, 3)
y[:5] # Each row is a 1-hot classification according to the available classes


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = train_gen.flow_from_directory(
  './clothing-dataset-small/validation/',
  target_size=(150, 150),
  batch_size=32,
)

base_model = Xception(
  weights='imagenet',
  include_top=False,  # only include convolutional layers, not dense ones
  input_shape=(150, 150, 3)
)
base_model.trainable = False  # Don't want to train convo layers

# Create new top/dense layer
inputs = keras.Input(shape=(150, 150, 3))
base = base_model(inputs)

# WHAT DO TO HERE? use pooling (see below)
outputs = base

model = keras.Model(inputs, outputs)
preds = model.predict(X)

preds.shape # (32, 5, 5, 2048) (batch is 32, so the output is (5, 5, 2048)
# Want to turn this (5, 5, 2048) into a 1-d vector
# Use (2d-average) pooling to do this

inputs = keras.Input(shape=(150, 150, 3))
base = base_model(inputs)

pooling = keras.layers.GlobalAveragePooling2D()
vectors = pooling(base)

outputs = vectors

model = keras.Model(inputs, outputs)
preds = model.predict(X)
preds.shape # (32, 2048) 32 batch, 1d due to pooling

# Now we need dense layer to turn 2048-d vector into output
inputs = keras.Input(shape=(150, 150, 3))
base = base_model(inputs, training=False)

vectors = keras.layers.GlobalAveragePooling2D()(base)

outputs = keras.layers.Dense(10)(vectors)  # 10 from # of multiclass
# To apply softmax afterwards, use
#outputs = keras.layers.Dense(10, activate='softmax')(vectors)  # 10 from # of multiclass

model = keras.Model(inputs, outputs)
preds = model.predict(X)
preds.shape # (32, 10) 32 batch


# Now we need to train the model, using an optimizer to get weights
learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.CategoricalCrossEntropy(
  from_logits=True,  # Set to True if Dense layer has Softmax activation
)
#keras.losses.BinaryCrossEntropy()
#keras.losses.MeanSquaredError() # For regression

# Applying Softmax (sigmoid for multiclass) is called Activation

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=val_ds)

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')


# 8.6 Adjusting the learning rate
"""
Learning rate == "how fast you can read" => faster == skimming text
                                            slower == taking notes, etc

"""
def make_model(learning_rate=0.01)
  base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3),
  )
  base_model.trainable = False

  # Make model architecture
  inputs = keras.Input(shape=(150, 150, 3))
  base = base_model(inputs, training=False)
  vectors = keras.layers.GlobalAveragePooling2D()(base)
  outputs = keras.layers.Dense(10)(vectors)
  model = keras.Model(inputs, outputs)
  ##

  learning_rate = 0.01
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  loss = keras.losses.CategoricalCrossEntropy(from_logits=True)

  model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'],
  )

  return model

scores = {}
for lr in [0.0001, 0.001, 0.01, 0.1]:
  print(lr)
  model = make_model(learning_rate=lr)
  history = model.fit(train_ds, epochs=10, validation_data=val_ds)
  scores[lr] = history.history
  print()
  print()

del scores[0.1]
del scores[0.0001]
for lr, hist in scores.items():
  plt.plot(history['val_accuracy'], label='val')
  plt.xticks(np.arange(10))
  plt.legend()

# Go with 0.001

# 8.7 Checkpointing
"""
Getting best model across eppochs (last one is not necessarily the
best one). Done using callbacks after each epovchs

"""

checkpoint = keras.callbacks.ModelCheckpoint(
  'xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
  save_best_only=True,
  monitor='val_accuracy',
  mode='max',
) 

learning_rate = 0.001
model = make_model(learning_rate=lr)
history = model.fit(
  train_ds,
  epochs=10,
  validation_data=val_ds,
  callbacks=[checkpoint],
)

# 8.8 Adding more (Dense) Layers

"""
Activation Functions
  o Output
    o Sigmoid
    o Softmax 
  o Intermediate
    o ReLU ( Rectified Linear Unit)
    o ...
"""

def make_model(learning_rate=0.01, size_inner=10):
  base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3),
  )
  base_model.trainable = False

  # Make model architecture
  inputs = keras.Input(shape=(150, 150, 3))
  base = base_model(inputs, training=False)
  vectors = keras.layers.GlobalAveragePooling2D()(base)

  inner = keras.layers.Dense(size_inner, activation='relu')(vectors)

  outputs = keras.layers.Dense(10)(inner)
  model = keras.Model(inputs, outputs)
  ##

  learning_rate = 0.01
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  loss = keras.losses.CategoricalCrossEntropy(from_logits=True)

  model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'],
  )

  return model

learning_rate = 0.001
scores = {}
for size in [10, 100, 1000]:
  model = make_model(learning_rate=learning_rate, size_inner=size)
  history = model.fit(train_ds, epochs=10, valdation_data=val_ds)
  scores[size] = history.history

# 8.9 Regularization and dropout
"""
Randomly hide part of the image so thing like logos don't
get turned into signifiers for being a shirt (since that logo
could be on a hat, etc)
"""


def make_model(learning_rate=0.01, size_inner=10, droprate=0.2):
  base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3),
  )
  base_model.trainable = False

  # Make model architecture
  inputs = keras.Input(shape=(150, 150, 3))
  base = base_model(inputs, training=False)
  vectors = keras.layers.GlobalAveragePooling2D()(base)

  inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
  drop = keras.layers.Dropout(droprate)(inner)

  outputs = keras.layers.Dense(10)(drop)
  model = keras.Model(inputs, outputs)
  ##

  learning_rate = 0.01
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  loss = keras.losses.CategoricalCrossEntropy(from_logits=True)

  model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'],
  )

  return model

learning_rate = 0.001
size = 100
scores = {}

for droprate in [0.0, 0.2, 0.5, 0.8]:
  model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate,
  )
  history = model.fit(train_ds, epochs=30, valdation_data=val_ds)
  scores[size] = history.history


# 8.10 Data Augmentation
"""
Create new data from old
  o vert & horiz flip image
  o rotate image
  o height shift
  o width shift
  o shear
  o zoom in/out
  o brightness/contract
"""
train_gen = ImageDataGenerator(
  preprocessing_function=preprocess_input
  rotation_range=30,
  width_shift_range=10.0,
  height_shift_range=10.0,
  shear_range=10,
  zoom_range=0.1,
  horizontal_flip=False,
  vertical_flip=True,
)
train_ds = train_gen.flow_from_directory(
  './clothing-dataset-small/train',
  target_size=(150, 150),
  batch_size=32,
)

val_gen = ImageDataGenerator(
  preprocessing_function=preprocess_input
)

# Do not apply augmentation to validation_ds
val_ds = val_gen.flow_from_directory(
  './clothing-dataset-small/validation',
  target_size=(150, 150),
  batch_size=32,
  shuffle=False,
)
learning_rate = 0.001
size = 100
droprate = 0.2

for droprate in [0.0, 0.2, 0.5, 0.8]:
  model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate,
  )
  history = model.fit(train_ds, epochs=50, valdation_data=val_ds)
  scores[size] = history.history



# 8.11 Training a larger model

def make_model(input_size=150, learning_rate=0.01, size_inner=10, droprate=0.2):
  base_model = Xception(
    weights='imagenet',l
    include_top=False,
    input_shape=(input_size, input_size, 3),
  )
  base_model.trainable = False

  # Make model architecture
  inputs = keras.Input(shape=(150, 150, 3))
  base = base_model(inputs, training=False)
  vectors = keras.layers.GlobalAveragePooling2D()(base)

  inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
  drop = keras.layers.Dropout(droprate)(inner)

  outputs = keras.layers.Dense(10)(drop)
  model = keras.Model(inputs, outputs)
  ##

  learning_rate = 0.01
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  loss = keras.losses.CategoricalCrossEntropy(from_logits=True)

  model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'],
  )

  return model

# 8.12 Using the model
import tensorflow as tf
from tensorflow import feras
model = keras.models.load_model('xception_v4_1_13_0.903.h5')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
  './clothing-dataset-small/test',
  target_size=(299, 299),
  batch_size=32,
  shuffle=False,
)

model.evaluate(test_ds) # Returns tuple (cat_cross_entropy, accuracy)
path = 'clothing-dataset-small/test/pants/<uuid>'
img = load_img(path, target_size=(299, 299))
x = nparray(img)
X = np.array([x])
X = preprocess_input(X)
pred = model.predict(X)
pred[0]

classes = [
  'pant',
  'shirt',
  'etc',
]
dict(zip(classes, pred[0]))



# 9 SERVERLESS #

# 9.1 Introduction
# 9.2 AWS Lambda
# 9.3 TensorFlow Lite
"""
Only inference (prediciton) not model training.

"""

pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
import tensorflow_runtime as tflite
from keras_image_helper import create_preprocessor


interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()  # Set weights

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(299, 299))

classes = [
  'dress',
  'hat',
  ...
]

def predict(url):
  X = preprocessor.from_url(url)

  interpreter.set_tensor(input_index, X)
  interpreter.invoke()
  preds = interpretor.get_tensor(output_index)

  float_predictions = preds[0].tolist()

  return dict(zip(classes, float_predictions))

# 9.4 Preparing for AWS Lambda
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result


# 9.5 Prep Docker Image

#Dockerfile
FROM public.ecr.aws/lambda/python:3.8

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl
COPY clothing-model.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]


$ docker build -t clothing-model . 
$ docker run -it --rm clothing-model:latest -p :8080:8080


import requests
url = 'http://localhost:8080/2015-03-31/functions/invocations/'
data = {
  'url': 'http://bit.ly/mlbookcamp-pants/',
}
response = requets.post(url, json=data).json()

# 9.6 Creating the Lambda Function
# Publish Docker contianer to Amazon Elastic Container Registry
$ pip install aswcli
$ aws configure
$ aws ecr create-repository --repository-name clothing-tflite-images

Copy URI

$ aws ecr get-login --no-include-email
<run the output of this ocmmand>

