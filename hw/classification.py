import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mutual_info_score, mean_squared_error
from sklearn.model_selection import train_test_split


def read_data() -> pd.DataFrame:
  df = pd.read_csv(
    filepath_or_buffer=(
      'https://raw.githubusercontent.com/alexeygrigorev/'
      'datasets/master/AB_NYC_2019.csv'
    ),
    usecols=[
      'neighbourhood_group',
      'room_type',
      'latitude',
      'longitude',
      'price',
      'minimum_nights',
      'number_of_reviews',
      'reviews_per_month',
      'calculated_host_listings_count',
      'availability_365',
    ],
  )
  df = df.fillna(0)
  return df

# Question 1
num = 1
df = read_data()
ans = df['neighbourhood_group'].mode().iloc[0]
print(f"Question {num}: {ans}")


# Spit the data
def split_data(
  df: pd.DataFrame,
  split: str = 'train',
  logistic=True,
  seed: int = 42,
) -> (
  pd.DataFrame,
  np.ndarray,
):

  # Do not edit original DataFrame
  df = df.copy()

  # Convert target value to binary variable
  target = 'price'
  if logistic:
    df['price'] = (df['price'] >= 152).astype(int)
  else:
    df['price'] = np.log1p(df['price'])

  # Split
  df_full_train, df_test = train_test_split(
    df,
    test_size=0.20,
    random_state=seed,
  )
  df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=seed,
  )

  df_full_train = df_full_train.reset_index(drop=True)
  df_train = df_train.reset_index(drop=True)
  df_val = df_val.reset_index(drop=True)
  df_test = df_test.reset_index(drop=True)

  # Remove targret variable from training data
  y_full_train = df_full_train.pop(target).values
  y_train = df_train.pop(target).values
  y_val = df_val.pop(target).values
  y_test = df_test.pop(target).values

  if split == 'full_train':
    result = df_full_train, y_full_train
  elif split == 'train':
    result = df_train, y_train
  elif split == 'val':
    result = df_val, y_val
  elif split == 'test':
    result = df_test, y_test
  else:
    raise ValueError("Unrecognized value for `split`.")

  return result


# Question 2
num = 2
df_train, y_train = split_data(df)
numerical = list(df_train.dtypes.loc[lambda x: x != 'object'].index)
correlation = df[numerical].corr().replace(1, np.nan).max().sort_values()
ans = list(correlation.iloc[-2:].index)
print(f"Question {num}: {ans}")


# Question 3
num = 3
categorical = list(df_train.dtypes.loc[lambda x: x == 'object'].index)

ans, ans_score = None, 0
for var in categorical:
  score = mutual_info_score(y_train, df_train[var])
  if score > ans_score:
    ans, ans_score = var, round(score, 2)
print(f"Question {num}: {ans} ({ans_score})")


def score_logistic_model(df: pd.DataFrame, drop_feature: str = '') -> float:

  df_train, y_train = split_data(df, split='train')

  features = [f for f in df_train if f != drop_feature]

  train_dicts = df_train[features].to_dict(orient='records')
  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(train_dicts)

  model = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
  model.fit(X_train, y_train)

  df_val, y_val = split_data(df, split='val')

  val_dicts = df_val[features].to_dict(orient='records')
  X_val = dv.fit_transform(val_dicts)

  y_pred = model.predict_proba(X_val)
  proba_false, proba_true = y_pred[:, 0], y_pred[:, 1]

  y_pred = (proba_true >= 0.5).astype(int)
  accuracy = (y_pred == y_val).mean()

  return round(accuracy, 6)


# Question 4
num = 4
ans = score_logistic_model(df)
print(f"Question {num}: {ans}")


# Question 5
num = 5
diffs = {}
features = [
  'neighbourhood_group',
  'room_type',
  'number_of_reviews',
  'reviews_per_month',
]
for feature in features:
  score = score_logistic_model(df, drop_feature=feature)
  diffs[feature] = round(ans - score, 6)
ans = diffs
print(f"Question {num}: {ans}")


def score_ridge_model(df: pd.DataFrame, alpha=0) -> float:

  df_train, y_train = split_data(df, split='train')

  train_dicts = df_train.to_dict(orient='records')
  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(train_dicts)

  model = Ridge(alpha=alpha)
  model.fit(X_train, y_train)

  df_val, y_val = split_data(df, split='val')

  val_dicts = df_val.to_dict(orient='records')
  X_val = dv.fit_transform(val_dicts)

  y_pred = model.predict(X_val)

  rmse = mean_squared_error(y_val, y_pred, squared=False)
  return round(rmse, 3)


# Question 6
num = 6
rmse = {}
for alpha in [0, 0.01, 0.1, 1, 10]:
  rmse[alpha] = score_ridge_model(df, alpha=alpha)
ans = rmse
print(f"Question {num}: {ans}")
