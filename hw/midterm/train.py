import numpy as np
import pandas as pd
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


FOLDER = 'servers/midterm/'


def fetch_data() -> pd.DataFrame:
    """Read in source data, clean, and engineer features.

    Source: https://www.kaggle.com/muhammetvarl/laptop-price

    """
    df = pd.read_csv(
        'hw/midterm/laptop_data_prices.zip',
        encoding='latin1',
    )

    # Standardize feature names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Feature Engineering: Split `memory` into multiple features
    memories = (
        df['memory']
        .str.split('+', expand=True)
        .rename(columns={0: 'memory1', 1: 'memory2'})
    )
    for column in memories.columns:
        memory = (
            memories[column]
            .str.strip()
            .str.split(' ', expand=True)
        )
        df[column] = memory[0].replace({'1.0TB': '1TB'}).fillna("N/A")
        df[f'{column}_type'] = memory[1].fillna("N/A")
    df = df.drop('memory', axis=1)

    # Feature Engineering: Split cpu into chipname and speed
    df['cpu_type'] = (
        df['cpu']
        .str.extract(r'(?P<cpu>.*\s*) [0-9\.]*GHz')
    )
    df['cpu_ghz'] = (
        df['cpu']
        .str.extract(r' (?P<ghz>[0-9\.]*)GHz')
        .astype(float)
    )
    df = df.drop('cpu', axis=1)

    # Feature Engineering: Split screentype into screen type, size, touchscreen
    df['screentype'] = (
        df['screenresolution']
        .str.extract(r'(.*) [0-9]*x[0-9]*')
        .fillna('N/A')
    )
    df['screentype'] = (
        df['screentype']
        .str.replace("IPS Panel Touchscreen / ", "")
        .str.replace(" / Touchscreen", "")
        .str.replace("Touchscreen / ", "")
    )
    df['screensize'] = (
        df['screenresolution']
        .str.extract(r'([0-9]*x[0-9]*)')
    )
    df['touchscreen'] = (
        df['screentype']
        .str.contains('Touchscreen')
        .astype(int)
    )
    df = df.drop('screenresolution', axis=1)

    # Convert weight to numerical feature
    df['weight'] = df['weight'].str.slice(0, -2).astype(float)

    # Add log transforms
    df['ln_price_euros'] = np.log(df['price_euros'])
    df['ln_weight'] = np.log(df['weight'])

    return df


def split_data(df: pd.DataFrame, split='train'):
    """Prepare X and y data for model.

    Notes
    -----
    Model features chosen based on feature importance analysis in the
    Jupyter notebook.

    """

    target = "ln_price_euros"
    features = [
        'typename',
        'memory1_type',
        'ln_weight',
        'ram',
        'memory1',
        'inches',
        'cpu_ghz',
        'gpu',
        'company',
        'cpu_type',
        'screensize',
        'screentype',
    ]

    # Set features
    df = df.copy()[[target] + features]

    # Split data
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

    # Extract target and encode
    y_full_train = df_full_train.pop(target).values
    y_train = df_train.pop(target).values
    y_val = df_val.pop(target).values
    y_test = df_test.pop(target).values

    # Fit and pickle DictVectorizer
    dv = DictVectorizer(sparse=False)
    dv.fit(df.to_dict(orient='records'))

    filepath = os.path.join(FOLDER, 'dv.bin')
    with open(filepath, 'wb') as f:
        pickle.dump(dv, f)
    print(f'DictVectorizer saved to {filepath}')

    # Convert DataFrames to feature matrices
    X_full_train = dv.transform(df_full_train.to_dict(orient='records'))
    X_train = dv.transform(df_train.to_dict(orient='records'))
    X_val = dv.transform(df_val.to_dict(orient='records'))
    X_test = dv.transform(df_test.to_dict(orient='records'))

    if split == 'train':
        X, y = X_train, y_train
    elif split == 'val':
        X, y = X_val, y_val
    elif split == 'full_train':
        X, y = X_full_train, y_full_train
    elif split == 'test':
        X, y = X_test, y_test
    return X, y


def train_model(X, y):
    """Trains the final model.

    Notes
    -----
    Model parameters come from model tuning in Jupyter notebook.

    """

    D, N = 20, 50
    rf = RandomForestRegressor(
        n_estimators=N,
        max_depth=D,
        random_state=1,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


# Train model on full training split
df = fetch_data()
X, y = split_data(df, split='full_train')
model = train_model(X, y)

# Test model on testing split
X_test, y_test = split_data(df, split='test')
y_pred = model.predict(X_test)
rmse = round(mean_squared_error(y_test, y_pred, squared=False), 3)

# Save model to file
filepath = os.path.join(FOLDER, 'model.bin')
with open(filepath, 'wb') as f:
    pickle.dump(model, f)

print(f'Model saved to {filepath} (RMSE {rmse})')
