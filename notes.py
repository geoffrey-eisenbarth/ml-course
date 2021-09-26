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

