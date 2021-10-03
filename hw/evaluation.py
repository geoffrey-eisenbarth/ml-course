import numpy as np
import pandas as pd
import plotext as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold

# import matplotlib.pyplot as plt
# import seaborn as sns


def get_data() -> pd.DataFrame:
  df = pd.read_csv(
    filepath_or_buffer=(
      'https://raw.githubusercontent.com/alexeygrigorev/'
      'mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv'
    ),
  )
  return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

  # Get consistent column names
  df.columns = df.columns.str.lower()

  # Decode features
  df['status'] = df['status'].map({
      1: 'ok',
      2: 'default',
      0: 'unk'
  })

  df['home'] = df['home'].map({
      1: 'rent',
      2: 'owner',
      3: 'private',
      4: 'ignore',
      5: 'parents',
      6: 'other',
      0: 'unk'
  })

  df['marital'] = df['marital'].map({
      1: 'single',
      2: 'married',
      3: 'widow',
      4: 'separated',
      5: 'divorced',
      0: 'unk'
  })

  df['records'] = df['records'].map({
      1: 'no',
      2: 'yes',
      0: 'unk'
  })

  df['job'] = df['job'].map({
      1: 'fixed',
      2: 'partime',
      3: 'freelance',
      4: 'others',
      0: 'unk'
  })

  # Prepare numerical values
  for col in ['income', 'assets', 'debt']:
    df[col] = df[col].replace(to_replace=99999999, value=0)

  # Remove clients with unknown default status
  df = df[df['status'] != 'unk'].reset_index(drop=True)

  # Create target variable
  df['default'] = (df['status'] == 'default').astype(int)
  del df['status']

  return df


df = clean_data(get_data())

categorical = list(df.dtypes[df.dtypes == 'object'].index)
numerical = list(df.dtypes[df.dtypes != 'object'].index)

df_full_train, df_test = train_test_split(
  df,
  test_size=0.20,
  random_state=1,
)
df_train, df_val = train_test_split(
  df_full_train,
  test_size=0.25,
  random_state=1,
)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Remove targret variable from training data
target = 'default'
y_full_train = df_full_train[target].values
y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values
numerical = [col for col in numerical if col != target]

# Question 1
num = 1
scores = {}
for column in numerical:
  if column in ['seniority', 'time', 'income', 'debt']:
    auc = roc_auc_score(y_train, df_train[column])
    if auc < 0.5:
      auc = roc_auc_score(y_train, -df_train[column])
    scores[column] = round(auc, 4)
ans = scores
print(f"Question {num}: {ans}")

features = ['seniority', 'income', 'assets', 'records', 'job', 'home']


def train(df_train, y_train, C=1.0):
  dicts = df_train[features].to_dict(orient='records')
  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(dicts)
  model = LogisticRegression(
    solver='liblinear',
    max_iter=1000,
    C=C,
  )
  model.fit(X_train, y_train)
  return dv, model


def predict(df, dv, model):
  dicts = df[features].to_dict(orient='records')
  X = dv.transform(dicts)
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred


# Question 2
num = 2
dv, model = train(df_train, y_train)
y_pred = predict(df_val, dv, model)
ans = round(roc_auc_score(y_val, y_pred), 3)
print(f"Question {num}: {ans}")


# Question 3
num = 3

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

  accuracy = (tp + tn) / (tp + tn + fp + fn)

  precision = tp / (tp + fp)
  recall = tp / (tp + fn)

  fpr = fp / (tn + fp)
  tpr = tp / (fn + tp)

  scores.append([round(accuracy, 4), round(precision, 4)])

scores = pd.DataFrame(
  scores,
  columns=['accuracy', 'precision'],
  index=thresholds,
).ffill()

plt.clear_plot()
plt.plot(scores['accuracy'], label='accuracy')
plt.plot(scores['precision'], label='precision')
plt.xticks(np.arange(0, 100, 10))
plt.show()

ans = 0.8
print(f"Question {num}: {ans}")


# Question 4
num = 4

scores = []
for t in thresholds:
  actual_positive = (y_val == 1)
  actual_negative = (y_val == 0)
  predict_positive = (y_pred >= t)
  predict_negative = (y_pred < t)

  tp = (predict_positive & actual_positive).sum()
  tn = (predict_negative & actual_negative).sum()
  fp = (predict_positive & actual_negative).sum()
  fn = (predict_negative & actual_positive).sum()

  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1 = 2 * precision * recall / (precision + recall)

  scores.append(f1)

scores = pd.Series(
  data=scores,
  index=thresholds,
  name='f1',
).sort_values(ascending=False)
ans = scores.index[0]

print(f"Question {num}: {ans}")


# Question 5
num = 5

scores = []
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
for train_idx, val_idx in kfold.split(df_full_train):
  df_train = df_full_train.iloc[train_idx]
  df_val = df_full_train.iloc[val_idx]

  y_train = df_train[target].values
  y_val = df_val[target].values

  dv, model = train(df_train, y_train)
  y_pred = predict(df_val, dv, model)

  auc = roc_auc_score(y_val, y_pred)
  scores.append(auc)

ans = f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
print(f"Question {num}: {ans}")


# Question 6
num = 6
scores = {}
for C in [0.01, 0.1, 1, 10]:
  aucs = []
  kfold = KFold(n_splits=5, shuffle=True, random_state=1)
  for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train[target].values
    y_val = df_val[target].values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    aucs.append(roc_auc_score(y_val, y_pred))

  scores[C] = f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}"

ans = scores
print(f"Question {num}: {ans}")
