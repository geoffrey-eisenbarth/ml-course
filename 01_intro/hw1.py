import numpy as np
import pandas as pd


# Question 1
num = 1
ans = np.__version__
print(f"Question {num}: {ans}")

# Question 2
num = 2
ans = pd.__version__
print(f"Question {num}: {ans}")

# Question 3
num = 3
url = (
  'https://raw.githubusercontent.com/'
  'alexeygrigorev/mlbookcamp-code/master/'
  'chapter-02-car-price/data.csv'
)
df = pd.read_csv(url)
ans = df.loc[df['Make'] == 'BMW', 'MSRP'].mean()
print(f"Question {num}: {ans}")

# Question 4
num = 4
ans = df.loc[df['Year'] > 2014, 'Engine HP'].isnull().sum()
print(f"Question {num}: {ans}")

# Question 5
num = 5
avg1 = df['Engine HP'].mean()
avg2 = df['Engine HP'].fillna(avg1).mean()
ans = (avg1 != avg2)
print(f"Question {num}: {ans}")

# Question 6
num = 6
rows = (df['Make'] == 'Rolls-Royce')
cols = ['Engine HP', 'Engine Cylinders', 'highway MPG']
X = df.loc[rows, cols].drop_duplicates().values
XTX = X.T.dot(X)
ans = np.linalg.inv(XTX).sum()
print(f"Question {num}: {ans}")

# Question 7
num = 7
y = [1000, 1100, 900, 1200, 1000, 850, 1300]
w = np.linalg.inv(XTX).dot(X.T).dot(y)
ans = w[0]
print(f"Question {num}: {ans}")
