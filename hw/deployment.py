import pickle
import requests

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# Question 1
num = 1
ans = "pipenv, version 11.9.0"
print(f"Question {num}: {ans}")

# Question 2
num = 2
hash = 'sha256:121f78d6564000dc5e968394f45aac87981fcaaf2be40cfcd8f07b2baa1e1829'
ans = f"Scikit-Learn first hash is {hash}"
print(f"Question {num}: {ans}")


def load_model() -> (DictVectorizer, LogisticRegression):
  # Download DictVectorizer and Model
  url = (
    'https://raw.githubusercontent.com/alexeygrigorev/'
    'mlbookcamp-code/master/course-zoomcamp/05-deployment/homework'
  )
  with requests.get(f"{url}/dv.bin") as r:
    dv = pickle.loads(r.content)
  with requests.get(f"{url}/model1.bin") as r:
    model = pickle.loads(r.content)
  return dv, model


def predict(customer):
  X = dv.transform([customer])
  not_churn, will_churn = model.predict_proba(X)[0]
  return will_churn


# Question 3
num = 3
dv, model = load_model()
customer = {
  "contract": "two_year",
  "tenure": 12,
  "monthlycharges": 19.7,
}
ans = predict(customer)
print(f"Question {num}: {ans}")


# Question 4
# Flask app set up in servers/geoffrey/predict.py
# Using servers/geoffrey/Dockerfile and gunicorn port 9696
num = 4
url = 'http://0.0.0.0:9696/predict'
customer = {
  "contract": "two_year",
  "tenure": 1,
  "monthlycharges": 10,
}
with requests.post(url, json=customer) as response:
  response = response.json()
  ans = response['churn_probability']
print(f"Question {num}: {ans}")

# Question 5
num = 5
ans = 'f0f43f7bc6e0'
print(f"Question {num}: {ans}")

# Question 6
# Flask app set up in servers/alexey/predict.py
# Using servers/alexey/Dockerfile and gunicorn port 9697
num = 5
url = 'http://0.0.0.0:9697/predict'
customer = {
  "contract": "two_year",
  "tenure": 12,
  "monthlycharges": 10,
}
with requests.post(url, json=customer) as response:
  response = response.json()
  ans = response['churn_probability']
print(f"Question {num}: {ans}")
