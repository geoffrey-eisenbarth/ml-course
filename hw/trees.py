import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def get_data() -> pd.DataFrame:
  df = pd.read_csv(
    filepath_or_buffer=(
      'https://raw.githubusercontent.com/alexeygrigorev/'
      'datasets/master/AB_NYC_2019.csv'
    ),
    header=0,
    usecols=[
      'neighbourhood_group',
      'room_type',
      'latitude',
      'longitude',
      'minimum_nights',
      'number_of_reviews',
      'reviews_per_month',
      'calculated_host_listings_count',
      'availability_365',
      'price',
    ],
  )
  df = df.fillna(0)
  df['price'] = np.log1p(df['price'])
  return df


# Split data
df = get_data()
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
target = 'price'
y_full_train = df_full_train.pop(target).values
y_train = df_train.pop(target).values
y_val = df_val.pop(target).values
y_test = df_test.pop(target).values


# Convert DataFrames into feature matrices
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(df_full_train.to_dict(orient='records'))
X_train = dv.fit_transform(df_train.to_dict(orient='records'))
X_val = dv.fit_transform(df_val.to_dict(orient='records'))
X_test = dv.fit_transform(df_test.to_dict(orient='records'))


# Question 1
num = 1

dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)
text = export_text(dt, feature_names=dv.get_feature_names())
print(text)

ans = 'room_type'
print(f"Question {num}: {ans}")


# Question 2
num = 2

rf = RandomForestRegressor(
  n_estimators=10,
  random_state=1,
  n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

ans = mean_squared_error(y_val, y_pred, squared=False)
print(f"Question {num}: {ans:.3f}")


# Question 3
num = 3

scores = []
for n in range(10, 201, 10):
  rf = RandomForestRegressor(
    n_estimators=n,
    random_state=1,
    n_jobs=-1,
  )
  rf.fit(X_train, y_train)
  y_pred = rf.predict(X_val)
  rmse = round(mean_squared_error(y_val, y_pred, squared=False), 3)
  scores.append((n, rmse))

scores = pd.DataFrame(scores, columns=['n', 'rmse'])

ans = 120
print(f"Question {num}: {ans}")


# Question 4
num = 4

scores = []
for d in [10, 15, 20, 25]:
  for n in range(10, 201, 10):
    rf = RandomForestRegressor(
      n_estimators=n,
      max_depth=d,
      random_state=1,
      n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = round(mean_squared_error(y_val, y_pred, squared=False), 3)
    scores.append((d, n, rmse))
scores = pd.DataFrame(scores, columns=['d', 'n', 'rmse'])

ans = scores.sort_values('rmse')['d'].iloc[0]
print(f"Question {num}: {ans}")  # 15


# Question Bonus
num = "bonus"

run_bonus = False
if run_bonus:
  scores = []
  for seed in range(5):
    for d in [10, 15, 20, 25]:
      for n in range(10, 201, 10):
        rf = RandomForestRegressor(
          n_estimators=n,
          max_depth=d,
          random_state=seed,
          n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = round(mean_squared_error(y_val, y_pred, squared=False), 3)
        scores.append((seed, d, n, rmse))
  scores = pd.DataFrame(scores, columns=['seed', 'd', 'n', 'rmse'])

ans = 'No'
print(f"Question {num}: {ans}")


# Question 5
num = 5

rf = RandomForestRegressor(
  n_estimators=10,
  max_depth=20,
  random_state=1,
  n_jobs=-1,
)
rf.fit(X_train, y_train)

feature_rank = (
  pd
  .DataFrame(
    data=zip(dv.get_feature_names(), rf.feature_importances_),
    columns=['feature', 'importance'],
  )
  .sort_values('importance', ascending=False)
  .reset_index(drop=True)
)

ans = feature_rank['feature'].iloc[0]
print(f"Question {num}: {ans}")


# Question 6
num = 6

features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]

scores = []
for eta in [0.3, 0.1, 0.01]:
  xgb_params = {
    'eta': eta,             # Learning rate
    'max_depth': 6,         # Number of levels in the tree
    'min_child_weight': 1,  # Same as min_samples_leaf

    'objective': 'reg:squarederror',
    'nthread': 6,

    'seed': 1,
    'verbosity': 1,
  }

  model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=100,
    evals=watchlist,
    verbose_eval=5,
  )
  y_pred = model.predict(dval)
  rmse = round(mean_squared_error(y_val, y_pred, squared=False), 3)
  scores.append((eta, rmse))

ans = scores
print(f"Question {num}: {ans}")
