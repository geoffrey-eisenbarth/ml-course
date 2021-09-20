import numpy as np
import pandas as pd


def read_data() -> pd.DataFrame:
  df = pd.read_csv(
    filepath_or_buffer=(
      'https://raw.githubusercontent.com/alexeygrigorev/'
      'datasets/master/AB_NYC_2019.csv'
    ),
    usecols=[
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
  return df


# Question 1
num = 1
df = read_data()
ans = df.isnull().sum().loc[lambda x: x > 0][0]
print(f"Question {num}: {ans}")


# Question 2
num = 2
ans = df['minimum_nights'].describe().loc['50%']
print(f"Question {num}: {ans}")


# Spit the data
def split_data(df: pd.DataFrame, split: str = 'train', seed: int = 42) -> (
  pd.DataFrame,
  np.ndarray,
):

  # Do not edit original DataFrame
  df = df.copy()

  np.random.seed(seed)
  n = len(df)
  idx = np.arange(n)
  np.random.shuffle(idx)
  n_val = n_test = int(0.20 * n)
  n_train = n - n_val - n_test

  if split == 'train':
    df = df.iloc[idx[:n_train]].reset_index(drop=True)
  elif split == 'val':
    df = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
  elif split == 'test':
    df = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)
  elif split == 'full_train':
    df = df.iloc[idx[:n_train + n_val]].reset_index(drop=True)
  else:
    raise ValueError("`split` must be one of 'train,' 'val,' 'test,' or 'full_train.'")

  # Make sure target value is not in the dataset (via `pop`)
  target = 'price'
  y = np.log1p(df.pop(target).values)

  return df, y


def prepare_X(df: pd.DataFrame, fillna: str = 'zero') -> np.ndarray:
  df = df.copy()

  features = df.columns
  df = df[features]

  if fillna == 'zero':
    X = df.fillna(0).values
  elif fillna == 'mean':
    X = df.fillna(df.mean()).values
  else:
    raise NotImplementedError("`fillna` must be one of 'zero' or 'mean'")

  return X


def train_linear_regression(X, y, r=0.001):
  ones = np.ones(X.shape[0])
  X = np.column_stack([ones, X])

  gram = X.T.dot(X) + np.eye(X.shape[1]) * r

  w = np.linalg.inv(gram).dot(X.T).dot(y)
  w0 = w[0]
  w = w[1:]
  return w0, w


def rmse(y, y_hat):
  error = ((y - y_hat) ** 2).mean()
  mse = (error ** 2).mean()
  rmse = np.sqrt(mse)
  return round(rmse, 2)


# Question 3
num = 3

df_train, y_train = split_data(df, split='train', seed=42)

models = {}
for fill_method in ['zero', 'mean']:
  X_train = prepare_X(df_train, fillna=fill_method)
  w0, w = train_linear_regression(X_train, y_train, r=0)
  y_hat = w0 + X_train.dot(w)
  models[fill_method] = rmse(y_train, y_hat)

ans = models
print(f"Question {num}: {ans}")


# Question 4
num = 4

df_val, y_val = split_data(df, split='val', seed=42)
X_val = prepare_X(df_val, fillna='zero')

rs = {}
for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
  X_train = prepare_X(df_train, fillna='zero')
  w0, w = train_linear_regression(X_train, y_train, r=r)

  y_hat = w0 + X_val.dot(w)
  rs[r] = rmse(y_val, y_hat)

ans = rs
print(f"Question {num}: {ans}")

# Question 5
num = 5

scores = []
for seed in range(10):
  df_train, y_train = split_data(df, split='train', seed=seed)
  df_val, y_val = split_data(df, split='val', seed=seed)

  X_train = prepare_X(df_train, fillna='zero')
  X_val = prepare_X(df_val, fillna='zero')

  w0, w = train_linear_regression(X_train, y_train, r=0)
  y_hat = w0 + X_val.dot(w)

  scores.append(rmse(y_val, y_hat))

print(scores)
ans = round(np.std(scores), 3)
print(f"Question {num}: {ans}")
print("Closest HW answer was 0.008")

# Question 6
num = 6
df_full_train, y_full_train = split_data(df, split='full_train', seed=9)
df_test, y_test = split_data(df, split='test', seed=9)

X_full_train = prepare_X(df_full_train, fillna='zero')
X_test = prepare_X(df_test, fillna='zero')

w0, w = train_linear_regression(X_full_train, y_full_train, r=0.001)
y_hat = w0 + X_test.dot(w)

ans = rmse(y_test, y_hat)
print(f"Question {num}: {ans}")
print("Closest HW answer was 0.39")
