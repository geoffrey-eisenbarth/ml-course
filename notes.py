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

