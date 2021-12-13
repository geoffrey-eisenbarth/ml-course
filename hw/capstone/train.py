import pandas as pd
import os
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


def get_data() -> (pd.DataFrame, pd.DataFrame):

  # Import data
  df_full_train = pd.read_csv('data/capstone/Video_games_esrb_rating.csv')
  df_test = pd.read_csv('data/capstone/test_esrb.csv')

  # Clean data
  df_full_train = df_full_train.rename(columns={
    'strong_janguage': 'strong_language',
  })
  df_full_train = df_full_train.drop('title', axis=1)
  df_test = df_test.rename(columns={
    'strong_janguage': 'strong_language',
  })
  df_test = df_test.drop('title', axis=1)

  return df_full_train, df_test


# Get data and set target/features
df_full_train, df_test = get_data()
target = 'esrb_rating'
features = [col for col in df_full_train if col != target]

# Prepare data
y_full_train = df_full_train.pop(target).values
y_test = df_test.pop(target).values
train_dicts = df_full_train[features].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(train_dicts)

test_dicts = df_test[features].to_dict(orient='records')
X_test = dv.fit_transform(test_dicts)

# Use LogisticRegression model with One-Vs-Rest, no penalty, lbfgs solver
model = LogisticRegression(
    random_state=0,
    class_weight=None,
    multi_class='ovr',
    penalty='none',
    solver='newton-cg',
)
model.fit(X_full_train, y_full_train)
score = model.score(X_test, y_test)
print(f'Score: {score:.2%}')

# Pickle model files into binaries
filepath = os.path.join('servers/capstone/', 'dv.bin')
with open(filepath, 'wb') as f:
  pickle.dump(dv, f)
print(f'DictVectorizer saved to {filepath}')

filepath = os.path.join('servers/capstone', 'model.bin')
with open(filepath, 'wb') as f:
  pickle.dump(model, f)
print(f'Model saved to {filepath}')

